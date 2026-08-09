"""Model identifiers and paths, shared by every stage of the pipeline."""

# HF model's id for the base model to be quantized.
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"

# Original model cache directory.
MODEL_DIR = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots" 

# Quantized model output directory.
OUTPUT_DIR = "models/Qwen3-0.6B-W4A16" 
