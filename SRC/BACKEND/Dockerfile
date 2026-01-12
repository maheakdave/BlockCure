FROM python:3.11

WORKDIR /app

COPY utils.py ./

RUN pip install websockets fastapi --no-cache-dir fastapi uvicorn transformers accelerate


CMD ["python","server.py"]