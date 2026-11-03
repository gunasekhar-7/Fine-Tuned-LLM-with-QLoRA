import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import json

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Test: what does the base model generate for a troubleshooting prompt?
messages = [{"role": "user", "content": "Wi-Fi won't connect to network"}]
encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
input_ids = encoded["input_ids"].to("cpu")
attention_mask = encoded["attention_mask"].to("cpu")

# Base model generation
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
with torch.no_grad():
    base_output = base_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64)
base_gen = tokenizer.decode(base_output[0][input_ids.shape[1]:], skip_special_tokens=False)
print("BASE MODEL GENERATION:")
print(base_gen)
print()

# Now with LoRA adapter
model_with_adapter = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
model_with_adapter = PeftModel.from_pretrained(model_with_adapter, r"C:\Projects\Fine-Tuned LLM with QLoRA\artifacts\trained_adapter")
model_with_adapter.eval()
with torch.no_grad():
    adapter_output = model_with_adapter.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64)
adapter_gen = tokenizer.decode(adapter_output[0][input_ids.shape[1]:], skip_special_tokens=False)
print("FT ADAPTER GENERATION:")
print(adapter_gen)
print()

# Check if they're different
print(f"Base: '{base_gen[:50]}'")
print(f"Adapter: '{adapter_gen[:50]}'")
print(f"Different: {base_gen != adapter_gen}")