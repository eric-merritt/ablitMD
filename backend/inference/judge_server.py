"""Locate and start the Qwen3.5-9B GGUF judge/classifier."""

import os
import subprocess
import time
from pathlib import Path

# Roots to search for the Qwen3.5-9B GGUF, in order. The app's models dir comes
# first (that's where it lives on vast.ai); ~/models is this dev box's home for it.
_SEARCH_ROOTS = [
    os.environ.get("ABLIT_MODELS_DIR", "/workspace/models"),
    os.path.expanduser("~/models"),
]

CLASSIFIER_PORT = int(os.environ.get("CLASSIFIER_PORT", "8239"))


def find_judge_model() -> str:
    """Locate the Qwen3.5-9B GGUF on disk. Returns the first match found.

    Searches each root recursively for a .gguf whose path mentions '9B' (case
    insensitive) and isn't an imatrix/mmproj sidecar. Raises if none is found."""
    seen: set[str] = set()
    for root in _SEARCH_ROOTS:
        root_path = Path(root).expanduser()
        if not root_path.is_dir():
            continue
        for gguf in sorted(root_path.rglob("*.gguf")):
            name = gguf.name.lower()
            # Skip imatrix/mmproj sidecars and any other non-9B model.
            if "imatrix" in name or "mmproj" in name or "9b" not in name:
                continue
            resolved = str(gguf.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            return resolved
    raise FileNotFoundError(
        f"no Qwen3.5-9B GGUF found under: {', '.join(_SEARCH_ROOTS)}"
    )


def start_judge_server(model_path: str | None = None) -> subprocess.Popen:
    """Start llama-server for the judge on CLASSIFIER_PORT."""
    model = model_path or find_judge_model()
    log_file = open("/tmp/ablitmd-classifier.log", "a")
    print(f"[judge] spawning llama-server: {model} (port {CLASSIFIER_PORT})", flush=True)
    proc = subprocess.Popen(
        [
            "llama-server",
            "-m", model,
            "--port", str(CLASSIFIER_PORT),
            "-c", "4096",
            "-ngl", "999",
            "--parallel", "1",
            "--threads", "8",
            "--temp", "0",
            "--reasoning", "off",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    # Give llama-server time to load the model into VRAM before we proceed.
    # The main model just loaded; don't fight it for GPU memory.
    print("[judge] waiting 30s for model to load...", flush=True)
    time.sleep(30)
    print("[judge] llama-server started", flush=True)
    return proc
