import gc
import os
import threading
import torch
import tqdm.auto
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LOCAL_MODEL_PATH = os.path.expanduser("~/models/Qwen/Qwen3.8-27B-Base/")

def get_model():
    model_path = f"{LOCAL_MODEL_PATH}"

    # INT8 weight-only quantization via bitsandbytes
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )

    return model




def get_tokenizer():
    """Loads and returns the local tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    return tokenizer


# Expose the active execution device dynamically
# (Usually torch.device('cuda:0') under device_map='auto')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_loaded_model_id: str | None = None
_model = None
_tokenizer = None
_model_dirty = False
_load_lock = threading.Lock()
_load_progress: float = 0.0
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


def load_model(model_id: str, api_model_id: str) -> None:
    """Load model_id (api_model_id on disk) into the resident slot.

    Reuses the existing resident model unless it's a different id or was
    marked dirty by an in-place ablation."""
    global _model, _tokenizer, _loaded_model_id, _load_progress
    with _load_lock:
        if _loaded_model_id == model_id and not _model_dirty:
            return
        unload_model()
        _load_progress = 0.0
        print(f"[model_loader] loading {LOCAL_MODEL_PATH} (int8 bnb, device_map=auto)", flush=True)
        orig = _patch_tqdm()
        try:
            _model = get_model()
            _model.eval()
            _tokenizer = get_tokenizer()
        finally:
            _restore_tqdm(orig)
        _loaded_model_id = model_id


def unload_model() -> None:
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