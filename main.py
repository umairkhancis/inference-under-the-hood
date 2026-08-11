import json
import math
import time

from inference_module.inference import continuous_batching_demo, prefix_caching_demo
from llms_module import BASE_MODEL_ID, OUTPUT_DIR, load_model
from quantization_module import quantize
from inference_module import infer, init_vllm, vllm_infer, get_vllm_metrics
from benchmark_module import benchmark

def main():
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


if __name__ == "__main__":
    """ Step 0: Initialize the vLLM server and get the loaded model ID. """
    model, inference_server_url = init_vllm()
    print(f"Connected to {inference_server_url} — model: {model}")


    """ Step 1: Run inference on the vLLM server. """
    prompt = "What is PagedAttention?"
    
    start = time.time()
    resp = vllm_infer(model, inference_server_url, prompt)
    elapsed = time.time() - start

    print(f"Response ({elapsed:.2f}s, {resp.usage.completion_tokens} tokens):")
    print(resp.choices[0].message.content)
    print(f"\nUsage: {resp.usage.prompt_tokens} prompt + "
            f"{resp.usage.completion_tokens} completion = {resp.usage.total_tokens} total")
    

    """ Step 2: Run inference on the vLLM server and print token-by-token probabilities. """
    prompt = "The capital of UAE is"
    
    start = time.time()
    resp = vllm_infer(model, inference_server_url, prompt, temperature=0.7, max_tokens=15, top_p=0.8, top_logprobs=5)
    elapsed = time.time() - start
    
    print(f"Response ({elapsed:.2f}s, {resp.usage.completion_tokens} tokens):")
    
    print("Token-by-token probabilities:\n")
    for tok in resp.choices[0].logprobs.content[:8]:
        print(f"  Chosen: '{tok.token}'  (logprob {tok.logprob:.2f})")
        if tok.top_logprobs:
            for alt in tok.top_logprobs[:5]:
                pct = math.exp(alt.logprob) * 100
                bar = "\u2588" * min(20, max(1, int(pct / 5)))
                print(f"    {pct:5.1f}%  {bar}  '{alt.token}'")
        print()
    
    """ Step 3: Get vLLM metrics and save a snapshot to outputs/metrics_snapshot.json. """
    metrics = get_vllm_metrics(inference_server_url)
    print("Current vLLM Metrics:")
    for key in ["vllm:num_requests_running", "vllm:num_requests_waiting",
                "vllm:gpu_cache_usage_perc", "vllm:cpu_cache_usage_perc",
                "vllm:prompt_tokens_total", "vllm:generation_tokens_total"]:
        if key in metrics:
            print(f"  {key.replace('vllm:', '')}: {metrics[key]:g}")

    with open("outputs/metrics_snapshot.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nFull metrics saved to outputs/metrics_snapshot.json")
    
    
    """ Step 4: Continuous Batching Demo: Run concurrent inference requests. """
    continuous_batching_demo(model, inference_server_url)

    """ Step 5: Prefix Caching Demo: Run inference requests with a shared system prompt. """
    prefix_caching_demo(model, inference_server_url)