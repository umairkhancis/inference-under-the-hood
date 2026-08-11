
import time
from inference_module import init_vllm, vllm_infer, demo_inference_performance, continuous_batching_demo

if __name__ == "__main__":
    
    """ Step 0: Initialize the vLLM server and get the loaded model ID. """
    print(f"\n{'=' * 20} Inference Server {'=' * 20}\n")
    model, inference_server_url = init_vllm()
    print(f"Connected to {inference_server_url} — model: {model}")

    """ Step 1: Run inference on the vLLM server. """
    prompt = "Explain why one should use vLLM?"
    start = time.time()
    resp = vllm_infer(model, inference_server_url, prompt, max_tokens=50)
    elapsed = time.time() - start
    
    # """ Step 2: Calculate Inference Service Level Indicators (SLIs). """
    # demo_inference_performance(inference_server_url, prompt, resp)
    
    """ Step 4: Continuous Batching Demo: Run concurrent inference requests. """
    continuous_batching_demo(model, inference_server_url)

    """ Step 5: Prefix Caching Demo: Run inference requests with a shared system prompt. """
    # prefix_caching_demo(model, inference_server_url)