import gc
import re
import threading
from pathlib import Path

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from backend.inference.model_loader import get_model, get_tokenizer, DEVICE


THINKING_STRIP_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

_active_abort: threading.Event | None = None
_abort_lock = threading.Lock()


class _EventStop(StoppingCriteria):
  def __init__(self, event: threading.Event):
    self._event = event

  def __call__(self, input_ids, scores, **kwargs) -> bool:
    return self._event.is_set()


def _tokenize_input(prompt_text: str, enable_thinking: bool):
  tokenizer = get_tokenizer()
  messages = [{"role": "user", "content": prompt_text}]

  def _apply(extra_kwargs):
    result = tokenizer.apply_chat_template(
      conversation=messages,
      add_generation_prompt=True,
      return_tensors="pt",
      **extra_kwargs,
    )
    ids = result["input_ids"] if hasattr(result, "__getitem__") and not isinstance(result, torch.Tensor) else result
    return ids.to(DEVICE)

  try:
    return _apply({"enable_thinking": enable_thinking})
  except TypeError:
    return _apply({})


def _capture_and_save_hidden_states(input_ids, hidden_states_key: str, runs_dir: Path, run_id: str) -> None:
  model = get_model()
  with torch.inference_mode():
    output = model(input_ids, output_hidden_states=True, use_cache=False)
  n_layers = model.config.num_hidden_layers
  hidden_states = np.array([
    output.hidden_states[layer_idx][0, -1, :].cpu().float().numpy()
    for layer_idx in range(n_layers + 1)
  ], dtype=np.float32)
  del output
  torch.cuda.empty_cache()

  state_dir = runs_dir / run_id
  state_dir.mkdir(parents=True, exist_ok=True)
  np.save(str(state_dir / f"{hidden_states_key}.npy"), hidden_states)


def _claim_abort_event() -> threading.Event:
  """Signal any in-flight generation to stop, then return a fresh event for this call."""
  global _active_abort
  with _abort_lock:
    if _active_abort is not None:
      _active_abort.set()
    event = threading.Event()
    _active_abort = event
    return event


def stream_prompt(
  prompt_text: str,
  mode: str,
  run_id: str,
  hidden_states_key: str,
  runs_dir: Path,
):
  """Yields dict events: {'type': 'ready'|'token'|'done'|'aborted', ...}."""
  model = get_model()
  tokenizer = get_tokenizer()
  enable_thinking = mode == 'thinking'

  input_ids = _tokenize_input(prompt_text, enable_thinking)
  _capture_and_save_hidden_states(input_ids, hidden_states_key, runs_dir, run_id)
  yield {"type": "ready", "hidden_states_key": hidden_states_key}

  abort_event = _claim_abort_event()
  streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60)

  def _run_generate():
    with torch.inference_mode():
      model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=False,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([_EventStop(abort_event)]),
      )

  worker = threading.Thread(target=_run_generate, daemon=True)
  worker.start()

  collected = ""
  try:
    for token in streamer:
      collected += token
      yield {"type": "token", "text": token}
  except Exception as err:
    yield {"type": "error", "error": str(err)}
  finally:
    worker.join(timeout=5)
    torch.cuda.empty_cache()
    gc.collect()

  stripped = THINKING_STRIP_RE.sub('', collected).strip()
  if abort_event.is_set():
    yield {"type": "aborted", "response": stripped}
  else:
    yield {"type": "done", "response": stripped}


# Retained for any non-streaming callers (compute pipeline, tests)
def run_prompt(
  prompt_text: str,
  mode: str,
  run_id: str,
  hidden_states_key: str,
  runs_dir: Path,
) -> str:
  full = ""
  for event in stream_prompt(prompt_text, mode, run_id, hidden_states_key, runs_dir):
    if event["type"] == "token":
      full += event["text"]
    elif event["type"] in ("done", "aborted"):
      return event["response"]
  return THINKING_STRIP_RE.sub('', full).strip()
