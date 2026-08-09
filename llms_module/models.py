"""Model identifiers and paths, shared by every stage of the pipeline."""

# the base model on the Hub. from_pretrained and oneshot both resolve this
# through the local HF cache, downloading only on a cache miss
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"

# where the cached bf16 weights land on disk. only used to measure the base
# model's size — it holds a revision subdirectory, so it is NOT loadable by
# from_pretrained; load BASE_MODEL_ID instead
MODEL_DIR = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"

# where quantization writes the W4A16 model, and where inference loads it from
OUTPUT_DIR = "models/Qwen3-0.6B-W4A16"
