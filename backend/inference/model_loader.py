import gc
import os
import threading
import torch
import tqdm.auto
from transformers import AutoModelForCausalLM, AutoTokenizer


_loaded_model_id: str | None = None
_model = None
_tokenizer = None
_model_dirty = False
_load_lock = threading.Lock()
_load_progress: float = 0.0
DEVICE = "cuda:0"
MODELS_DIR = os.environ.get("ABLIT_MODELS_DIR", "/workspace/models")
BAKE_DIR = os.environ.get("ABLIT_BAKE_DIR", MODELS_DIR)


def get_load_progress() -> float:
  return _load_progress


def get_loaded_model_id() -> str | None:
  return _loaded_model_id


def set_model_dirty(value: bool = True) -> None:
  """Mark the resident weights as ablated, so the next load reloads from disk."""
  global _model_dirty
  _model_dirty = value


def is_model_dirty() -> bool:
  return _model_dirty


# Local checkouts to load from disk instead of HF, keyed by HF model id.
# Guarded by isdir below, so these stay inert where the path is absent (e.g. runpod).
LOCAL_MODEL_PATHS = {
  "Qwen/Qwen3.5-9B": os.path.expanduser("~/models/Qwen/Qwen3.5-9B-Base"),
}


def _resolve_model_path(model_id: str) -> str:
  override = LOCAL_MODEL_PATHS.get(model_id)
  if override and os.path.isdir(override):
    return override
  model_name = model_id.split("/")[-1]
  local = os.path.join(MODELS_DIR, model_name)
  if os.path.isdir(local):
    return local
  return model_id


class _ProgressTqdm(tqdm.auto.tqdm):
  def update(self, n=1):
    super().update(n)
    global _load_progress
    if self.total:
      _load_progress = self.n / self.total


def _patch_tqdm():
  orig = tqdm.auto.tqdm
  tqdm.auto.tqdm = _ProgressTqdm
  return orig


def _restore_tqdm(orig):
  global _load_progress
  tqdm.auto.tqdm = orig
  _load_progress = 1.0


def _strip_to_text_model(model) -> None:
  """Drop the vision tower and MTP head, leaving a pure text causal LM.

  Qwen3.5+ checkpoints ship as ForConditionalGeneration: a `model.visual`
  vision tower plus a top-level `mtp` multi-token-prediction head. For text
  abliteration we want only `model.language_model` + `lm_head`. The MTP head
  also perturbs the next-token logits during generate, so removing it is what
  lets greedy decoding produce real tokens instead of an immediate stop.

  Guarded by getattr, so this is a no-op on already-text checkpoints."""
  inner = getattr(model, "model", None)
  if inner is not None and getattr(inner, "visual", None) is not None:
    inner.visual = None
  if getattr(model, "mtp", None) is not None:
    model.mtp = None
  gc.collect()
  torch.cuda.empty_cache()


def _load_pretrained(source_path: str):
  orig = _patch_tqdm()
  try:
    model = AutoModelForCausalLM.from_pretrained(
      source_path,
      device_map="auto",
      torch_dtype=torch.bfloat16,
      attn_implementation="flash_attention_2",
    )
    _strip_to_text_model(model)
    return model
  finally:
    _restore_tqdm(orig)


def load_model(model_id: str, api_model_id: str) -> None:
  global _model, _tokenizer, _loaded_model_id, _load_progress
  with _load_lock:
    if _loaded_model_id == model_id and not _model_dirty:
      return
    _do_unload()
    _load_progress = 0.0
    source_path = _resolve_model_path(api_model_id)
    print(f"[model_loader] loading {source_path} (bf16)", flush=True)
    _model = _load_pretrained(source_path)
    _model.eval()
    _tokenizer = AutoTokenizer.from_pretrained(source_path)
    _loaded_model_id = model_id


def _do_unload() -> None:
  global _model, _tokenizer, _loaded_model_id, _model_dirty
  if _model is not None:
    for param in _model.parameters():
      param.data = torch.empty(0)
    del _model
    _model = None
  if _tokenizer is not None:
    del _tokenizer
    _tokenizer = None
  _loaded_model_id = None
  _model_dirty = False
  gc.collect()
  torch.cuda.empty_cache()


def unload_model() -> None:
  with _load_lock:
    _do_unload()


def get_model():
  if _model is None:
    raise RuntimeError("No model loaded. Call /load first.")
  return _model


def get_tokenizer():
  if _tokenizer is None:
    raise RuntimeError("No tokenizer loaded. Call /load first.")
  return _tokenizer
