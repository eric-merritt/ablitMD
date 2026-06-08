# ablitMD

Try my **ablitMD** method or the classic (*heretic*-style) abliteration — all in one app.

## Quickstart

```bash
# Clone the repo
git clone git@github.com:eric-merritt/ablitMD.git

# Sync deps
cd ablitMD
uv sync && npm i

# Download & install the flash-attention wheel for 2.8.3
curl -O https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
uv pip install flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# Start the FastAPI inference service & Express proxy
npm run dev
```
