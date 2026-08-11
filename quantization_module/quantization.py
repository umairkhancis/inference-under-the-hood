
from llms_module import BASE_MODEL_ID, OUTPUT_DIR, load_model
from inference_module import infer
from benchmark_module import benchmark

import warnings
warnings.filterwarnings("ignore")

import os, gc, math, pathlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from . import mps_compat
mps_compat.install()

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

from llms_module import BASE_MODEL_ID, MODEL_DIR, OUTPUT_DIR
from utils import folder_size, format_size

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def quantize():
    if not is_required():
        print(f"Quantized model already exists at {OUTPUT_DIR}.")
        print(f"Skipping quantization step.\n")
        return
    
    """Quantize the base model to W4A16 and report the size reduction."""

    ## Step 1: Report the model and output directories

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

    if is_required():
        oneshot(
            model=BASE_MODEL_ID,
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


def is_required():
    """Return True if the quantized model is missing and needs to be generated."""

    return not os.path.isdir(OUTPUT_DIR)

def demo_quantization():
    """ Step 1: Quantize the original model. """
    print(f"\n{'=' * 20} Quantization {'=' * 20}\n")
    quantize()

    """ Step 2: Load the original model. """
    base_model, base_tokenizer = load_model(BASE_MODEL_ID)
    
    """ Step 3: Load the quantized model. """
    quant_model, quant_tokenizer = load_model(OUTPUT_DIR)
    
    """ Step 4: Run inference on the original model. """
    print(f"\n{'=' * 20} Original Model Inference {'=' * 20}\n")
    infer(base_model, base_tokenizer)

    """ Step 5: Run inference on the quantized model. """
    print(f"\n{'=' * 20} Quantized Model Inference {'=' * 20}\n")
    infer(quant_model, quant_tokenizer)
    
    print(f"\n{'=' * 20} Benchmarking Original vs Quantized {'=' * 20}\n")
    print(f"Calculating perplexity scores...\n")

    """ Step 6: Benchmark the original model. """
    base_perplexity = benchmark(base_model, base_tokenizer, label="Base (BF16)")

    """ Step 7: Benchmark the quantized model. """
    quant_perplexity = benchmark(quant_model, quant_tokenizer, label="Quantized (W4A16)")

    """ Step 8: Compare the results and print a summary. """
    print(f"\nBase (BF16):        {base_perplexity:.2f}")
    print(f"Quantized (W4A16):  {quant_perplexity:.2f}")
    print(f"Difference:         {quant_perplexity - base_perplexity:+.2f} ({(quant_perplexity / base_perplexity - 1)*100:+.1f}%)")
    print(f"\nA small increase in perplexity is expected and does not degrade the model's accuracy significantly\n")