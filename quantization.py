import warnings
warnings.filterwarnings("ignore")

import os, gc, math, pathlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import mps_compat
mps_compat.install()

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

from utils import folder_size, format_size

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

## Step: 1: Define the model and output directories

MODEL_DIR = "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
OUTPUT_DIR = "models/Qwen3-0.6B-W4A16"

print(f"\n{'=' * 20} Quantization {'=' * 20}\n")
print(f"Base model:      {MODEL_DIR}")
print(f"Quantized model: {OUTPUT_DIR}")

## Step 2: Define the quantization recipe

recipe = GPTQModifier(
    scheme="W4A16",
    targets="Linear",
    ignore=["lm_head"],
)

# print(f"Recipe: {recipe}")

## Step 3: Run the quantization process

if not os.path.isdir(OUTPUT_DIR):
    oneshot(
        model="Qwen/Qwen3-0.6B",
        dataset="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
        recipe=recipe,
        output_dir=OUTPUT_DIR,
        max_seq_length=4096,
        num_calibration_samples=256,
        # torch 2.12's pin_memory() is broken when MPS is the accelerator, and the
        # activation cache pins every CPU-offloaded tensor. Offload to MPS instead;
        # on unified memory this costs nothing.
        sequential_offload_device=(
            "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    )
    print(f"Quantization complete. Model saved to: {OUTPUT_DIR}")

size_orig = folder_size(MODEL_DIR)
size_q = folder_size(OUTPUT_DIR)
reduction = (1 - size_q / size_orig) * 100 if size_orig > 0 else 0

print(f"\n{'=' * 20} Quantization Impact {'=' * 20}\n")
print(f"Original (BF16):    {format_size(size_orig)}")
print(f"Quantized (W4A16):  {format_size(size_q)}")
print(f"Reduction:          {reduction:.0f}%\n")
