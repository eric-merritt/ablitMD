import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_loaded_model_id: str | None = None
_model = None
_tokenizer = None
DEVICE = "cuda:0"


def get_loaded_model_id() -> str | None:
  return _loaded_model_id


def load_model(model_id: str, api_model_id: str) -> None:
  global _loaded_model_id, _model, _tokenizer

  if _loaded_model_id == model_id:
    return

  unload_model()

  print(f"Loading model {api_model_id} on {DEVICE}...")
  _model = AutoModelForCausalLM.from_pretrained(
    api_model_id,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    trust_remote_code=True,
  )
  _tokenizer = AutoTokenizer.from_pretrained(api_model_id, trust_remote_code=True)
  _loaded_model_id = model_id
  print(f"Model loaded: {model_id}")


def unload_model() -> None:
  global _loaded_model_id, _model, _tokenizer
  if _model is not None:
    del _model
    del _tokenizer
    torch.cuda.empty_cache()
    _model = None
    _tokenizer = None
    _loaded_model_id = None


def get_model():
  if _model is None:
    raise RuntimeError("No model loaded. Call /load first.")
  return _model


def get_tokenizer():
  if _tokenizer is None:
    raise RuntimeError("No tokenizer loaded. Call /load first.")
  return _tokenizer
