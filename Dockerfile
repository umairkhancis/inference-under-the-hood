# vLLM lab — CPU-only Linux environment.
# The vLLM CPU wheel is x86_64-only, so build and run with an explicit
# platform (emulated via Rosetta on Apple silicon):
#   docker build --platform linux/amd64 -t vllm-lab .
#   docker run --rm -it --platform linux/amd64 vllm-lab
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libnuma1 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home app

# ARM64 binaries that cannot run on Linux. It lives outside /app so a bind-mounted project directory can never shadow it.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV" && chown -R app:app "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

USER app

WORKDIR /app

COPY --chown=app:app requirements.txt .

RUN pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY --chown=app:app . .

# Pre-create the HF cache dir so a fresh named volume mounted here
# inherits app's ownership instead of defaulting to root
RUN mkdir -p /home/app/.cache/huggingface

CMD ["bash"]
