import warnings
import concurrent
import time, requests, json, os, math, sys
from openai import OpenAI, base_url
from functools import partial
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


def init_vllm():
    VLLM_URL = "http://localhost:8000"
    os.makedirs("outputs", exist_ok=True)

    print("Waiting for vLLM server...")
    for attempt in range(60):
        try:
            r = requests.get(f"{VLLM_URL}/v1/models", timeout=5)
            if r.status_code == 200:
                MODEL = r.json()["data"][0]["id"]
                break
        except requests.ConnectionError:
            pass
        time.sleep(5)
        if attempt % 6 == 5:
            print(f"  Still waiting... ({(attempt + 1) * 5}s elapsed)")
    else:
        raise RuntimeError(
            "vLLM server not reachable after 5 minutes."
        )

    return MODEL, VLLM_URL

def vllm_infer(model, inference_server_url, prompt, temperature=0.7, max_tokens=80, top_p=0.8, top_logprobs=5):
    client = OpenAI(base_url=f"{inference_server_url}/v1", api_key="unused")
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_logprobs=top_logprobs,
        logprobs=True,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    )
    
    return resp

def get_vllm_metrics(inference_server_url):
    """Scrape vLLM Prometheus /metrics and return {name: value}."""
    r = requests.get(f"{inference_server_url}/metrics")
    metrics = {}
    for line in r.text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split()[0]
        try:
            metrics[name] = float(line.split()[-1])
        except (ValueError, IndexError):
            continue

    return metrics

def debug_metrics(inference_server_url):
    metrics_of_interest = [
        "vllm:time_to_first_token_seconds_sum", 
        "vllm:time_to_first_token_seconds_count", 
        "vllm:inter_token_latency_seconds_sum", 
        "vllm:inter_token_latency_seconds_count",
        "vllm:e2e_request_latency_seconds_sum", 
        "vllm:e2e_request_latency_seconds_count",
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_total"
    ]
    
    metrics = get_vllm_metrics(inference_server_url)
    
    print("vLLM Metrics:\n")
    for key in metrics_of_interest:
        if key in metrics:
            print(f"  {key.replace('vllm:', '')}: {metrics[key]:g}")

def demo_inference_performance(inference_server_url, prompt, resp):
    metrics = get_vllm_metrics(inference_server_url)
    ttft = metrics.get("vllm:time_to_first_token_seconds_sum", 0) / max(1, metrics.get("vllm:time_to_first_token_seconds_count", 1))
    itl = metrics.get("vllm:inter_token_latency_seconds_sum", 0) / max(1, metrics.get("vllm:inter_token_latency_seconds_count", 1))
    tl = metrics.get("vllm:e2e_request_latency_seconds_sum", 0) / max(1, metrics.get("vllm:e2e_request_latency_seconds_count", 1))
    decode_throughput = metrics.get("vllm:generation_tokens_total", 0) / max(1, metrics.get("vllm:e2e_request_latency_seconds_sum", 1))
    prefill_throughput = metrics.get("vllm:prompt_tokens_total", 0) / max(1, metrics.get("vllm:e2e_request_latency_seconds_sum", 1))
    sys_throughput = (metrics.get("vllm:prompt_tokens_total", 0) + metrics.get("vllm:generation_tokens_total", 0)) / max(1, metrics.get("vllm:e2e_request_latency_seconds_sum", 1))
    
    print(f"\nPrompt: {prompt}\n")
    print(f"Response: {resp.choices[0].message.content}\n")
    print(f"Usage: {resp.usage.prompt_tokens} prompt + "
          f"{resp.usage.completion_tokens} completion = {resp.usage.total_tokens} total\n")
    print(f"Inference SLIs:\n")
    print(f"- Average Time To First Token (TTFT): {ttft:.2f}s")
    print(f"- Average Inter Token Latency (ITL): {itl:.2f}s")
    print(f"- Average Total Latency (TL): {tl:.2f}s")
    print(f"- Generation Throughput (Decode Phase): {decode_throughput:.2f} tokens/s")
    print(f"- Prompt Ingestion Throughput (Prefill Phase): {prefill_throughput:.2f} tokens/s")
    print(f"- System Throughput (Prefill + Decode Phases): {sys_throughput:.2f} tokens/s\n")


_ask = partial(vllm_infer, max_tokens=50, temperature=0.7, top_p=0.8, top_logprobs=5)

MAX_CONCURRENT_REQUESTS = 50

def continuous_batching_demo(model, inference_server_url):
    concurrent_batches = [
        ["What is Quantization?"] * MAX_CONCURRENT_REQUESTS,
        ["How is KV Cache?"] * MAX_CONCURRENT_REQUESTS,
        ["What is Continuous Batching?"] * MAX_CONCURRENT_REQUESTS,
    ]
    
    print(f"\n{'=' * 20} Continuous Batching Demonstration {'=' * 20}\n")

    for batch_num, batch in enumerate(concurrent_batches, start=1):
        prompts = batch
        
        before = get_vllm_metrics(inference_server_url)
        print(f"\nBatch #{batch_num}: Issuing burst of {len(prompts)} concurrent requests...\n")
        start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = { pool.submit(_ask, prompt=p, model=model, inference_server_url=inference_server_url): p for p in prompts }
            
            iter = 0
            while not all(f.done() for f in futures):
                iter += 1
                during_metrics = get_vllm_metrics(inference_server_url)
                
                running = during_metrics.get("vllm:num_requests_running", "--")
                waiting = during_metrics.get("vllm:num_requests_waiting", "--")
                
                status = f"{iter}.  ⏳ [vLLM STATE] Running: {running} | Waiting: {waiting}"
                print(status)

                time.sleep(2)
            
        elapsed = time.time() - start
        after = get_vllm_metrics(inference_server_url)
        tokens = after.get("vllm:generation_tokens_total", 0) - before.get("vllm:generation_tokens_total", 0)

        print(f"\n✅ Completed {len(prompts)} requests | {elapsed:.2f}s | ~{tokens / elapsed:.1f} tokens/s")
        print("")


def prefix_caching_demo(model, inference_server_url):
    client = OpenAI(base_url=f"{inference_server_url}/v1", api_key="unused")
    
    SYSTEM_PROMPT = (
        "You are a helpful assistant that answers questions about vLLM, a high-performance inference engine for LLMs. "
    )

    questions = [
        "What is Quantization?",
        "How is KV Cache?",
        "What is Continuous Batching?",
        "Why use Prefix Caching?",
        "What is PagedAttention?",
    ]

    before = get_vllm_metrics(inference_server_url)
    timings = []
    tok_counts = []

    print("Sending 5 requests with the SAME system prompt...\n")
    for i, q in enumerate(questions):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            max_tokens=60, temperature=0.7,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        dt = time.time() - t0
        timings.append(dt)
        tok_counts.append(resp.usage.completion_tokens)
        tokens = resp.usage.completion_tokens
        ms_per_tok = (dt / tokens * 1000) if tokens > 0 else 0.0
        print(f"  [{i+1}] {q:<35} {dt:.2f}s  ({tokens} tok, {ms_per_tok:.0f} ms/tok)")
        
    after = get_vllm_metrics(inference_server_url)

    prefix_before = before.get("vllm:prefix_cache_queries_total", 0)
    prefix_after = after.get("vllm:prefix_cache_queries_total", 0)

    print(f"\nPrefix cache queries: {prefix_before:g} -> {prefix_after:g}  (+{prefix_after - prefix_before:g})")

    cache_keys = [k for k in after if "prefix" in k.lower() or "cache_hit" in k.lower()]
    
    for k in sorted(cache_keys):
        b, a = before.get(k, 0), after.get(k, 0)
        if a != b and k != "vllm:prefix_cache_queries_total":
            print(f"  {k}: {b:g} -> {a:g}")
    
        
    hits_before = before.get("vllm:prefix_cache_hits_total", 0)
    hits_after = after.get("vllm:prefix_cache_hits_total", 0)
    hit_rate = (hits_after / prefix_after * 100) if prefix_after > 0 else 0.0
    
    print(f"\nPrefix cache hit rate: {hit_rate:.1f}%")

    print("\n The increasing prefix_cache_queries count confirms vLLM is ")
    print("checking and reusing cached KV blocks for the shared system prompt.")