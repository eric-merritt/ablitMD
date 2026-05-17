import gc
import re
from pathlib import Path

import numpy as np
import torch

from backend.inference.model_loader import get_model, get_tokenizer, DEVICE


THINKING_STRIP_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


@torch.inference_mode()
def run_prompt(
  prompt_text: str,
  mode: str,
  run_id: str,
  hidden_states_key: str,
  runs_dir: Path,
) -> str:
  model = get_model()
  tokenizer = get_tokenizer()
  enable_thinking = mode == 'thinking'

  messages = [{"role": "user", "content": prompt_text}]

  try:
    input_ids = tokenizer.apply_chat_template(
      conversation=messages,
      add_generation_prompt=True,
      return_tensors="pt",
      enable_thinking=enable_thinking,
    ).to(DEVICE)
  except TypeError:
    input_ids = tokenizer.apply_chat_template(
      conversation=messages,
      add_generation_prompt=True,
      return_tensors="pt",
    ).to(DEVICE)

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

  try:
    gen_output = model.generate(
      input_ids,
      max_new_tokens=512,
      do_sample=False,
    )
  except Exception:
    gen_output = model.generate(input_ids, max_new_tokens=512, do_sample=False)

  response = tokenizer.decode(gen_output[0][input_ids.shape[1]:], skip_special_tokens=True)
  response = THINKING_STRIP_RE.sub('', response).strip()

  del gen_output
  torch.cuda.empty_cache()
  gc.collect()

  return response
