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
_active_worker: threading.Thread | None = None
_worker_lock = threading.Lock()


def abort_and_join_generation(timeout: float = 5.0) -> None:
  with _abort_lock:
    if _active_abort is not None:
      _active_abort.set()
  with _worker_lock:
    worker = _active_worker
  if worker is not None and worker.is_alive():
    worker.join(timeout=timeout)


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
      return_dict=True,
      **extra_kwargs,
    )
    return result["input_ids"].to(DEVICE), result["attention_mask"].to(DEVICE)

  try:
    return _apply({"enable_thinking": enable_thinking})
  except TypeError:
    return _apply({})


def _capture_and_save_hidden_states(input_ids, hidden_states_key: str, runs_dir: Path, run_id: str, attention_mask=None) -> None:
  model = get_model()
  with torch.inference_mode():
    output = model(input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
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
  skip_hidden_states: bool = False,
):
  """Yields dict events: {'type': 'ready'|'token'|'done'|'aborted', ...}."""
  model = get_model()
  tokenizer = get_tokenizer()
  enable_thinking = mode == 'thinking'

  input_ids, attention_mask = _tokenize_input(prompt_text, enable_thinking)
  if not skip_hidden_states:
    _capture_and_save_hidden_states(input_ids, hidden_states_key, runs_dir, run_id, attention_mask)
  yield {"type": "ready", "hidden_states_key": hidden_states_key}

  abort_event = _claim_abort_event()
  streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60)

  def _run_generate():
    with torch.inference_mode():
      model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        do_sample=False,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([_EventStop(abort_event)]),
      )

  global _active_worker
  worker = threading.Thread(target=_run_generate, daemon=True)
  with _worker_lock:
    _active_worker = worker
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
    with _worker_lock:
      if _active_worker is worker:
        _active_worker = None
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
  skip_hidden_states: bool = False,
) -> str:
  full = ""
  for event in stream_prompt(prompt_text, mode, run_id, hidden_states_key, runs_dir, skip_hidden_states):
    if event["type"] == "token":
      full += event["text"]
    elif event["type"] in ("done", "aborted"):
      return event["response"]
  return THINKING_STRIP_RE.sub('', full).strip()
