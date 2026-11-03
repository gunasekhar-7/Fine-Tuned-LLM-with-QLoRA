import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct', torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True
)

# Load adapter config manually
adapter_path = r'C:\Projects\Fine-Tuned LLM with QLoRA\artifacts\trained_adapter'
with open(os.path.join(adapter_path, 'adapter_config.json')) as f:
    adapter_config = json.load(f)
print('Adapter config:', adapter_config)

# Update config to match our training
adapter_config['r'] = 16
adapter_config['lora_alpha'] = 64
adapter_config['target_modules'] = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

# Create PeftConfig and load adapter
peft_config = PeftConfig(**adapter_config)
adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
adapter_model.eval()

# Test generation
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
messages = [{'role': 'user', 'content': 'Wi-Fi won\\'t connect'}]
encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt')
input_ids = encoded['input_ids'].to('cpu')
attention_mask = encoded['attention_mask'].to('cpu')

with torch.no_grad():
    output = adapter_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64)

generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
print('Generated:', generated[:80])
"