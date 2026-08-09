import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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
