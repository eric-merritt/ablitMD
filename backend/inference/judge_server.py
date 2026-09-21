"""Own the 9B judge's llama-server process.

The audit's refusal classifier (see :mod:`classifier_llm`) talks to a llama-server
running Qwen3.5-9B on CLASSIFIER_PORT. Rather than assume it's already up, we spawn
and own it here: if the port isn't answering when an audit starts, we locate the
Qwen3.5-9B GGUF on disk, launch a lean llama-server on it (no cache quant, no
reasoning, no tools — a classifier only needs to read one prompt and reply with a
word), and wait until it's ready.

The model file is searched for, not hardcoded: we look under the app's models dir
(ABLIT_MODELS_DIR, default /workspace/models on vast.ai) and ~/models, so the same
code works on this box and on the instance.

IMPORTANT: The classifier MUST NOT start until AFTER the first generation completes
and VRAM stabilizes. We poll the inference service's /status endpoint for
first_generation_done=true before spawning llama-server.
"""

import os
import socket
import subprocess
import time
from pathlib import Path

import requests

CLASSIFIER_PORT = int(os.environ.get("CLASSIFIER_PORT", "8239"))
INFERENCE_STATUS_URL = os.environ.get(
    "INFERENCE_STATUS_URL", "http://localhost:8238/status"
)

# Roots to search for the Qwen3.5-9B GGUF, in order. The app's models dir comes
# first (that's where it lives on vast.ai); ~/models is this dev box's home for it.
_SEARCH_ROOTS = [
    os.environ.get("ABLIT_MODELS_DIR", "/workspace/models"),
    os.path.expanduser("~/models"),
]

_proc: subprocess.Popen | None = None


def _wait_for_first_generation(timeout: float = 600.0) -> None:
    """Block until the inference service reports first_generation_done=true."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(INFERENCE_STATUS_URL, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("first_generation_done"):
                    print("[judge] first generation complete, proceeding...", flush=True)
                    return
        except (requests.ConnectionError, requests.Timeout, ValueError):
            pass
        time.sleep(3)
    raise TimeoutError(
        "Inference service did not report first_generation_done=true "
        f"within {timeout}s"
    )


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


def _port_open(port: int = CLASSIFIER_PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_ready(timeout: float = 900.0) -> None:
    """Block until the server answers on its port, or raise after `timeout` seconds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open():
            return
        time.sleep(1.0)
    raise TimeoutError(f"llama-server did not become ready on port {CLASSIFIER_PORT}")


def ensure_judge_server(model_path: str | None = None) -> None:
    """Make sure the judge is reachable on CLASSIFIER_PORT, spawning it if needed.

    Idempotent: if something already answers on the port we do nothing. Otherwise
    wait for first generation to complete (so VRAM has stabilized), locate the 9B
    GGUF, launch llama-server on it, and wait for readiness."""
    global _proc
    if _port_open():
        return

    # Wait for VRAM to stabilize after first generation
    _wait_for_first_generation()

    model = model_path or find_judge_model()
    log_path = os.path.expanduser("~/llama_server_judge.log")
    log_file = open(log_path, "a")
    print(f"[judge] spawning llama-server: {model} (port {CLASSIFIER_PORT})", flush=True)
    _proc = subprocess.Popen(
        [
            "llama-server",
            "-m", model,
            "--port", str(CLASSIFIER_PORT),
            # A classifier reads one prompt and answers a word; no big context needed.
            # GPU offload (-ngl 999) — the 27B uses ~53GB of 96GB, leaving headroom for the 9B.
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
    try:
        _wait_ready()
    except BaseException:
        # Dump the log so the user sees WHY it failed (OOM, file not found, etc.)
        log_file.seek(0)
        log_snippet = log_file.read(-1)[-2000:]  # last 2KB
        print(f"[judge] FAILED to start — log tail:\n{log_snippet}", flush=True)
        _proc.terminate()
        raise
    print(f"[judge] ready on port {CLASSIFIER_PORT}", flush=True)
