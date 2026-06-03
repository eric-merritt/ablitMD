import gc
import os
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


_loaded_model_id: str | None = None
_model = None
_tokenizer = None
_load_lock = threading.Lock()
DEVICE = "cuda:0"
MODELS_DIR = "/home/ermer/models/Qwen"


def _resolve_model_path(model_id: str) -> str:
  model_name = model_id.split("/")[-1]
  local = os.path.join(MODELS_DIR, model_name)
  if os.path.isdir(local):
    return local
  return model_id


def get_loaded_model_id() -> str | None:
  return _loaded_model_id


def load_model(model_id: str, api_model_id: str) -> None:
  global _model, _tokenizer, _loaded_model_id
  with _load_lock:
    if _loaded_model_id == model_id:
      return
    _do_unload()
    path = _resolve_model_path(api_model_id)
    quant_cfg = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      llm_int8_enable_fp32_cpu_offload=True,
    )
    _model = AutoModelForCausalLM.from_pretrained(
      path,
      quantization_config=quant_cfg,
      device_map="auto",
      max_memory={0: "13500MiB", "cpu": "64GiB"},
      torch_dtype=torch.bfloat16,
      attn_implementation="flash_attention_2",
    )
    _model.eval()
    _tokenizer = AutoTokenizer.from_pretrained(path)
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
