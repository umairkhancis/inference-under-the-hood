import math

import torch
from datasets import load_dataset

## Step 1: Define the evaluation dataset

# the fully-qualified repo id — huggingface_hub 1.x rejects a bare "wikitext",
# which only works through oneshot because llm-compressor canonicalizes it
DATASET = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"

## Step 2: Score a loaded model
def calculate_perplexity(model, tokenizer, dataset, max_tokens=5000, stride=512):
    encodings = tokenizer(
        "\n\n".join(dataset["text"]),
        return_tensors="pt", truncation=True, max_length=max_tokens,
    )
    input_ids = encodings.input_ids
    nlls, prev_end = [], 0

    for begin_loc in range(0, input_ids.size(1), stride):
        end_loc = min(begin_loc + stride, input_ids.size(1))
        trg_len = end_loc - prev_end
        input_slice = input_ids[:, begin_loc:end_loc]
        target_slice = input_slice.clone()
        target_slice[:, :-trg_len] = -100
        with torch.no_grad():
            loss = model(input_slice, labels=target_slice).loss
            nlls.append(loss * trg_len)
        prev_end = end_loc

    return math.exp(torch.stack(nlls).sum() / prev_end)


## Step 3: Report

def benchmark(model, tokenizer):
    """Measure perplexity of an already-loaded `model` and print it."""
    
    test_data = load_dataset(DATASET, DATASET_CONFIG, split="test")
    perplexity = calculate_perplexity(model, tokenizer, test_data)
    
    return perplexity
