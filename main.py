from llms_module import BASE_MODEL_ID, OUTPUT_DIR, load_model
from quantization_module import quantize
from inference_module import infer
from benchmark_module import benchmark

if __name__ == "__main__":

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
    print(f"Calculating perplexity scores...")

    """ Step 6: Benchmark the original model. """
    base_perplexity = benchmark(base_model, base_tokenizer)

    """ Step 7: Benchmark the quantized model. """
    quant_perplexity = benchmark(quant_model, quant_tokenizer)
    
    """ Step 8: Compare the results and print a summary. """
    print(f"Base (BF16):        {base_perplexity:.2f}")
    print(f"Quantized (W4A16):  {quant_perplexity:.2f}")
    print(f"Difference:         {quant_perplexity - base_perplexity:+.2f} ({(quant_perplexity / base_perplexity - 1)*100:+.1f}%)")
    print(f"\nA small increase in perplexity is expected and does not degrade the model's accuracy significantly\n")
