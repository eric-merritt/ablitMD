FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y git curl

RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - && apt-get install -y nodejs

EXPOSE 5400 8237 8238

CMD ["bash", "/workspace/ablitMD/startup.sh"]
