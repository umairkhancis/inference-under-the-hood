from llms_module import BASE_MODEL_ID, OUTPUT_DIR
from quantization_module import quantize
from inference_module import infer, load_model

if __name__ == "__main__":
    
    quantize()

    base_model, base_tokenizer = load_model(BASE_MODEL_ID)
    
    print(f"\n{'=' * 20} Original Model Inference {'=' * 20}\n")
    infer(base_model, base_tokenizer)

    print(f"\n{'=' * 20} Quantized Model Inference {'=' * 20}\n")
    quant_model, quant_tokenizer = load_model(OUTPUT_DIR)
    infer(quant_model, quant_tokenizer)
