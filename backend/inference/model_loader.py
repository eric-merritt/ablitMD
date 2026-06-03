import gc
import os
import threading
import torch
import tqdm.auto
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


_loaded_model_id: str | None = None
_model = None
_tokenizer = None
_load_lock = threading.Lock()
_load_progress: float = 0.0
DEVICE = "cuda:0"
MODELS_DIR = "/home/ermer/models/Qwen"
NF4_SUFFIX = "-nf4"

MAX_MEMORY = {0: "13500MiB", "cpu": "64GiB"}


def get_load_progress() -> float:
  return _load_progress


def get_loaded_model_id() -> str | None:
  return _loaded_model_id


def _resolve_model_path(model_id: str) -> str:
  model_name = model_id.split("/")[-1]
  local = os.path.join(MODELS_DIR, model_name)
  if os.path.isdir(local):
    return local
  return model_id


def _cache_path(model_id: str) -> str:
  return os.path.join(MODELS_DIR, model_id.split("/")[-1] + NF4_SUFFIX)


class _ProgressTqdm(tqdm.auto.tqdm):
  def update(self, n=1):
    super().update(n)
    global _load_progress
    if self.total:
      _load_progress = self.n / self.total


def _patch_tqdm():
  _orig = tqdm.auto.tqdm
  tqdm.auto.tqdm = _ProgressTqdm
  return _orig


def _restore_tqdm(orig):
  global _load_progress
  tqdm.auto.tqdm = orig
  _load_progress = 1.0


def _patch_params4bit_compat():
  from bitsandbytes.nn import Params4bit
  _orig_new = Params4bit.__new__
  def _new(cls, *args, **kwargs):
    kwargs.pop('_is_hf_initialized', None)
    return _orig_new(cls, *args, **kwargs)
  Params4bit.__new__ = staticmethod(_new)


def _from_cache(cache_dir: str):
  _patch_params4bit_compat()
  orig = _patch_tqdm()
  try:
    return AutoModelForCausalLM.from_pretrained(
      cache_dir,
      device_map="auto",
      max_memory=MAX_MEMORY,
      torch_dtype=torch.bfloat16,
      attn_implementation="flash_attention_2",
    )
  finally:
    _restore_tqdm(orig)


def _quantize_and_cache(source_path: str, cache_dir: str):
  quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
  )
  orig = _patch_tqdm()
  try:
    model = AutoModelForCausalLM.from_pretrained(
      source_path,
      quantization_config=quant_cfg,
      device_map="auto",
      max_memory=MAX_MEMORY,
      torch_dtype=torch.bfloat16,
      attn_implementation="flash_attention_2",
    )
  finally:
    _restore_tqdm(orig)
  print(f"[model_loader] saving NF4 cache to {cache_dir}", flush=True)
  model.save_pretrained(cache_dir)
  return model


def load_model(model_id: str, api_model_id: str) -> None:
  global _model, _tokenizer, _loaded_model_id, _load_progress
  with _load_lock:
    if _loaded_model_id == model_id:
      return
    _do_unload()
    _load_progress = 0.0
    cache_dir = _cache_path(api_model_id)
    if os.path.isdir(cache_dir):
      print(f"[model_loader] loading from NF4 cache: {cache_dir}", flush=True)
      _model = _from_cache(cache_dir)
    else:
      source_path = _resolve_model_path(api_model_id)
      print(f"[model_loader] quantizing {source_path} → {cache_dir}", flush=True)
      _model = _quantize_and_cache(source_path, cache_dir)
    _model.eval()
    _tokenizer = AutoTokenizer.from_pretrained(
      cache_dir if os.path.isdir(cache_dir) else _resolve_model_path(api_model_id)
    )
    _loaded_model_id = model_id


def _do_unload() -> None:
  global _model, _tokenizer, _loaded_model_id
  if _model is not None:
    for param in _model.parameters():
      param.data = torch.empty(0)
    del _model
    _model = None
  if _tokenizer is not None:
    del _tokenizer
    _tokenizer = None
  _loaded_model_id = None
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
