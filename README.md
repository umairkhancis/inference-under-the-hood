# inference-under-the-hood

This repo is about understanding the inference layer of the AI stack from the ground up — not by reading about it, but by running each piece and watching where it breaks. 

## Step 1: Quantization

A technique to compress the size of LLMs by reducing the numerical precision of their parameters (weights/biases).

Quantization is based on the insight: 

“Not all weights are equally important. Some weights changed slightly cause large changes in the model’s output while other weights can be rounded quite aggressively without much effect on model’s output.”

It is about conversion of model's parameters from its released format BF16 (16 bits) to lower precision formats like FP8 (8 bits), INT8 (8 bits), INT4 (4 bits).

Fewer bits means lesser GPU ($$$) memory required.

Quantization done naively (rounding every number down) does degrade the model quality.

Quantization “done correctly”, maintains the same accuracy across the benchmarks.

Here, “done correctly” means calibrated techniques like GPTQ, AWQ, SmoothQuant.

And use small representative dataset during quantization to calibrate which weights matter the most.

And protect those sensitive weights during quantization process while aggressively round-off.

`quantization.py` compresses Qwen3-0.6B from `BF16` format to `W4A16` format using GPTQ via `llm-compressor`, taking the weights from 1.41 GB to 524 MB (about 64% smaller).

<img width="1165" height="323" alt="Screenshot 2026-08-09 at 8 13 28 AM" src="https://github.com/user-attachments/assets/3baac170-7dad-4876-a0a2-a0f3c58a3a0f" /> 
<img width="1166" height="420" alt="Screenshot 2026-08-09 at 9 09 53 AM" src="https://github.com/user-attachments/assets/ef00e045-bf38-4d59-b9e0-ade3d8faadef" />


Stay tuned!
