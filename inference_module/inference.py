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



_ask = partial(vllm_infer, max_tokens=50, temperature=0.7, top_p=0.8, top_logprobs=5)

def continuous_batching_demo(model, inference_server_url):
    prompts = [
        "What is Quantization?",
        "How is KV Cache?",
        "What is Continuous Batching?",
        "Why use Prefix Caching?",
        "What is PagedAttention?",
    ]
    
    before = get_vllm_metrics(inference_server_url)
    print(f"Sending {len(prompts)} concurrent requests...\n")
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(prompts)) as pool:
        futures = {pool.submit(_ask, prompt=p, model=model, inference_server_url=inference_server_url): p for p in prompts}
        time.sleep(0.5)
        during = get_vllm_metrics(inference_server_url)
        running = during.get("vllm:num_requests_running", "--")
        waiting = during.get("vllm:num_requests_waiting", "--")
        print(f"  [mid-flight]  running: {running}  |  waiting: {waiting}")

        for f in concurrent.futures.as_completed(futures):
            resp = f.result()
            print(f"  done: \"{futures[f][:40]}\" -> {resp.usage.completion_tokens} tokens")

    elapsed = time.time() - start
    after = get_vllm_metrics(inference_server_url)
    tokens = after.get("vllm:generation_tokens_total", 0) - before.get("vllm:generation_tokens_total", 0)

    print(f"\nAll {len(prompts)} completed in {elapsed:.2f}s")
    if tokens > 0:
        print(f"Tokens generated: {tokens:g}  |  ~{tokens / elapsed:.1f} tokens/s")


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