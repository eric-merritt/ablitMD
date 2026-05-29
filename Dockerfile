FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y git curl

RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - && apt-get install -y nodejs

RUN curl -sfS https://dotenvx.sh | sh

COPY pkgs/ /workspace/pkgs/

COPY .env /workspace/env-staging/
COPY frontend/.env /workspace/env-staging/frontend/
COPY backend/.env /workspace/env-staging/backend/

EXPOSE 5400 8237 8238

CMD ["bash", "/workspace/ablitMD/startup.sh"]
