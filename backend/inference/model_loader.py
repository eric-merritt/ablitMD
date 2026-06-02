import gc
import os
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_loaded_model_id: str | None = None
_model = None
_tokenizer = None
_load_lock = threading.Lock()
DEVICE = "cuda:0"
LOCAL_MODELS_DIRS = ["/workspace/models", "/home/ermer/models/Qwen"]


def _resolve_model_path(api_model_id: str) -> str:
    model_name = api_model_id.split("/")[-1]
    for base_dir in LOCAL_MODELS_DIRS:
        local = os.path.join(base_dir, model_name)
        if os.path.isdir(local):
            return local
    return api_model_id


def get_loaded_model_id() -> str | None:
    return _loaded_model_id


def load_model(model_id: str, api_model_id: str) -> None:
    global _loaded_model_id, _model, _tokenizer

    with _load_lock:
        if _loaded_model_id == model_id:
            return

        unload_model()

        model_path = _resolve_model_path(api_model_id)
        print(f"Loading model {model_path} on {DEVICE}...")
        try:
            _model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=DEVICE,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            _tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            if _tokenizer.pad_token_id is None:
                _tokenizer.pad_token_id = _tokenizer.eos_token_id
            _loaded_model_id = model_id
            print(f"Model loaded: {model_id}")
        except Exception:
            _model = None
            _tokenizer = None
            _loaded_model_id = None
            gc.collect()
            torch.cuda.empty_cache()
            raise


def unload_model() -> None:
    global _loaded_model_id, _model, _tokenizer
    if _model is not None:
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(_model, recurse=True)
        _model.cpu()
        del _model
        del _tokenizer
        gc.collect()
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
