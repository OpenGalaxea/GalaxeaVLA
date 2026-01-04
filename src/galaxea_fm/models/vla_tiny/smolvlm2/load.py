import glob
import json
import os

from safetensors import safe_open
from transformers import AutoTokenizer

from .config import SmolVLMConfig
from .smolvlm2_model import SmolVLMForConditionalGeneration


def load_hf_model(
    model_path: str,
    device: str,
    quantize: bool = False,
    check_param: bool = False,
):
    if quantize:
        print("Running qunatized model")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = SmolVLMConfig(**model_config_file)

    # Create the model using the configuration
    model = SmolVLMForConditionalGeneration(config)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=True)
    
    # Verify that all parameters are loaded correctly
    print("\n" + "="*80)
    print("Verifying model parameter loading...")
    print("="*80)
    
    if check_param:
        import torch
        all_loaded = True
        max_diff = 0.0
        param_count = 0
        missing_params = []
        mismatched_params = [] 

        for name, param in model.named_parameters():
            param_count += 1
            if name not in tensors:
                all_loaded = False
                missing_params.append(name)
                print(f"❌ Parameter NOT found in state_dict: {name}")
            else:
                # Compare the loaded parameter with the original tensor
                diff = torch.abs(param.data - tensors[name]).max().item()
                max_diff = max(max_diff, diff)
                
                if diff > 1e-6:
                    mismatched_params.append((name, diff))
                    print(f"⚠️  Parameter {name}: max diff = {diff:.2e} (exceeds 1e-6)")
        
        print("\n" + "-"*80)
        print(f"Total parameters checked: {param_count}")
        print(f"All parameters loaded: {'✅ YES' if all_loaded else '❌ NO'}")
        print(f"Maximum difference: {max_diff:.2e}")
        print(f"Difference within 1e-6: {'✅ YES' if max_diff <= 1e-6 else '❌ NO'}")
        
        if missing_params:
            print(f"\n❌ Missing parameters ({len(missing_params)}):")
            for name in missing_params[:10]:  # Show first 10
                print(f"   - {name}")
            if len(missing_params) > 10:
                print(f"   ... and {len(missing_params) - 10} more")
        
        if mismatched_params:
            print(f"\n⚠️  Parameters with diff > 1e-6 ({len(mismatched_params)}):")
            for name, diff in mismatched_params[:10]:  # Show first 10
                print(f"   - {name}: {diff:.2e}")
            if len(mismatched_params) > 10:
                print(f"   ... and {len(mismatched_params) - 10} more")
        
        if all_loaded and max_diff <= 1e-6:
            print("\n✅ All parameters loaded successfully with acceptable precision!")
        
        print("="*80 + "\n")

    # Move the model to the device --- quantization happens if the model is quantized
    model = model.to(device)

    # Note: Do NOT tie weights here! The config has tie_word_embeddings=False
    # which means lm_head and embed_tokens should have separate weights

    return (model, tokenizer)
