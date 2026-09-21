import gc
import os
import threading
import torch
import tqdm.auto
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS_DIR = os.environ.get("ABLIT_MODELS_DIR", "/workspace/models")


def get_model(model_path: str | None = None):
    """Load the resident model, or a fresh one from model_path if given."""
    global _model
    if model_path is None:
        if _model is None:
            raise RuntimeError("No model loaded")
        return _model

    # Load directly onto GPU — no CPU intermediate. device_map="cuda:0" makes
    # accelerate stream each safetensors shard straight to the card.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    return model




def get_tokenizer(model_path: str | None = None):
    """Return the resident tokenizer, or load one from model_path if given."""
    global _tokenizer
    if model_path is None:
        if _tokenizer is None:
            raise RuntimeError("No tokenizer loaded")
        return _tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
BAKE_DIR = os.environ.get("ABLIT_BAKE_DIR", MODELS_DIR)


def _resolve_model_path(model_id: str) -> str:
    """Resolve the on-disk path for model_id.

    If it's already a directory (or lives loose in MODELS_DIR), use it;
    otherwise treat it as an HF repo id and download into MODELS_DIR."""
    if os.path.isdir(model_id):
        return model_id
    # Model files may live loose directly in MODELS_DIR (no subdirectory).
    if os.path.isfile(os.path.join(MODELS_DIR, "config.json")):
        return MODELS_DIR
    candidate = os.path.join(MODELS_DIR, model_id)
    if os.path.isfile(os.path.join(candidate, "config.json")):
        return candidate
    print(f"[model_loader] downloading {model_id} from HF", flush=True)
    return snapshot_download(model_id, local_dir=os.path.join(MODELS_DIR, model_id))


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


def load_model(model_id: str) -> None:
    """Load model into the resident slot.

    Reuses the existing resident model unless it's a different id or was
    marked dirty by an in-place ablation."""
    global _model, _tokenizer, _loaded_model_id, _load_progress
    with _load_lock:
        if _loaded_model_id == model_id and not _model_dirty:
            return
        unload_model()
        _load_progress = 0.0
        model_path = _resolve_model_path(model_id)
        print(f"[model_loader] loading {model_path} (device_map=auto)", flush=True)
        orig = _patch_tqdm()
        try:
            _model = get_model(model_path)
            _model.eval()
            _tokenizer = get_tokenizer(model_path)
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
