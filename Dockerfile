FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y git curl

RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - && apt-get install -y nodejs

RUN curl -sfS https://dotenvx.sh | sh

COPY pkgs/ /workspace/pkgs/
RUN pip install /workspace/pkgs/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

COPY .env .env.keys /workspace/env-staging/
COPY frontend/.env frontend/.env.keys /workspace/env-staging/frontend/
COPY backend/.env backend/.env.keys /workspace/env-staging/backend/

EXPOSE 5400 8237 8238

CMD ["bash", "/workspace/ablitMD/startup.sh"]
