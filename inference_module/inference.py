import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

## Step 1: Define the prompt to generate from

PROMPT = "Machine learning is a branch of"


## Step 2: Load the tokenizer and model

def load_model(model_id):
    """Load the weights and tokenizer for `model_id`.

    `model_id` is anything from_pretrained accepts — a Hub id for the base
    model, or a local directory for the quantized one. Both ship their own
    tokenizer, so it loads from the same place as the weights.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", dtype=torch.bfloat16,
    )
    return model, tokenizer


## Step 3: Generate

def infer(model, tokenizer):
    """Generate a completion from an already-loaded `model` and print it."""

    inputs = tokenizer(PROMPT, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[-1]:]

    print(f"\nPrompt: {PROMPT}")
    print(f"\nResponse: {tokenizer.decode(generated, skip_special_tokens=True)}\n")
