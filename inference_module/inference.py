import warnings
warnings.filterwarnings("ignore")

## Step 1: Define the prompt to generate from

PROMPT = "Machine learning is a branch of"


## Step 2: Generate

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
