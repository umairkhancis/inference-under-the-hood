# inference-under-the-hood

This repo is about understanding the inference layer of the AI stack from the ground up — not by reading about it, but by running each piece and watching where it breaks. 

**Quantization** is the first step. 

`quantization.py` compresses Qwen3-0.6B from `BF16` format to `W4A16` format using GPTQ via `llm-compressor`, taking the weights from 1.41 GB to 524 MB (about 64% smaller).

Stay tuned!