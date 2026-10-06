# syntax=docker/dockerfile:1

# ---- Stage 1: export the model to ONNX (needs PyTorch, thrown away afterwards) ----
FROM python:3.12-slim AS builder

ARG MODEL_ID=distilbert-base-uncased-finetuned-sst-2-english
# Set to 0 to skip the PyTorch-vs-ONNX benchmark and build faster.
ARG RUN_BENCHMARK=1

WORKDIR /build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY app app
COPY scripts scripts

RUN python -m scripts.export_onnx --model "$MODEL_ID" --out artifacts/model
RUN if [ "$RUN_BENCHMARK" = "1" ]; then \
      python -m scripts.benchmark --requests 200 --batch-sizes 1,8; \
    fi
# The serving stage only needs the quantized graph, tokenizer and config.
RUN rm -f artifacts/model/model.safetensors artifacts/model/model.onnx

# ---- Stage 2: serving image (ONNX Runtime only, no PyTorch) ----
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user app app
COPY --chown=user --from=builder /build/artifacts artifacts

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '7860') + '/readyz', timeout=3)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
