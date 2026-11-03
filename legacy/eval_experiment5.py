import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
test_file = r"C:\Projects\Fine-Tuned LLM with QLoRA\domainadapt\data\processed\test.jsonl"
SCHEMA_REQUIRED_KEYS = {"category", "action", "parameters", "estimated_time_minutes"}

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load base model
print("Loading base model (no adapter)...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
)

# Load best fine-tuned model (r16_alpha64)
print("Loading base model + LoRA adapter (r16_alpha64)...")
model_with_adapter = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
)
model_with_adapter = PeftModel.from_pretrained(model_with_adapter, r"C:\Projects\Fine-Tuned LLM with QLoRA\artifacts\experiments\r16_alpha64")
model_with_adapter.eval()

# Load test examples (10 for speed)
print("Loading test examples...")
with open(test_file) as f:
    all_test_examples = [json.loads(line) for line in f]
test_examples = all_test_examples[:10]

# Evaluate base model
print("\n" + "="*60)
print("EVALUATION: Base model (no adapter)")
print("="*60)

base_em = 0
base_schema_valid = 0
base_invalid = 0

for i, example in enumerate(test_examples):
    instruction = example["instruction"]
    expected_json = json.loads(example["expected_output"])
    
    messages = [{"role": "user", "content": instruction}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    with torch.no_grad():
        output = base_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=128)
    
    generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
    
    json_start = generated_text.find("{")
    json_end = generated_text.rfind("}") + 1
    parsed_output = None
    if json_start >= 0 and json_end > json_start:
        try:
            parsed_output = json.loads(generated_text[json_start:json_end])
        except json.JSONDecodeError:
            parsed_output = None
    
    if parsed_output == expected_json:
        base_em += 1
    if parsed_output is not None and SCHEMA_REQUIRED_KEYS.issubset(parsed_output.keys()):
        base_schema_valid += 1
    if parsed_output is None:
        base_invalid += 1

base_total = len(test_examples)
print(f"Exact Match: {base_em}/{base_total} ({base_em/base_total*100:.1f}%)")
print(f"Schema Validity: {base_schema_valid}/{base_total} ({base_schema_valid/base_total*100:.1f}%)")
print(f"Invalid Output Rate: {base_invalid}/{base_total} ({base_invalid/base_total*100:.1f}%)")

# Evaluate fine-tuned model
print("\n" + "="*60)
print("EVALUATION: Fine-tuned model (r16_alpha64 adapter)")
print("="*60)

ft_em = 0
ft_schema_valid = 0
ft_invalid = 0

for i, example in enumerate(test_examples):
    instruction = example["instruction"]
    expected_json = json.loads(example["expected_output"])
    
    messages = [{"role": "user", "content": instruction}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    with torch.no_grad():
        output = model_with_adapter.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=128)
    
    generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
    
    json_start = generated_text.find("{")
    json_end = generated_text.rfind("}") + 1
    parsed_output = None
    if json_start >= 0 and json_end > json_start:
        try:
            parsed_output = json.loads(generated_text[json_start:json_end])
        except json.JSONDecodeError:
            parsed_output = None
    
    if parsed_output == expected_json:
        ft_em += 1
    if parsed_output is not None and SCHEMA_REQUIRED_KEYS.issubset(parsed_output.keys()):
        ft_schema_valid += 1
    if parsed_output is None:
        ft_invalid += 1

ft_total = len(test_examples)
print(f"Exact Match: {ft_em}/{ft_total} ({ft_em/ft_total*100:.1f}%)")
print(f"Schema Validity: {ft_schema_valid}/{ft_total} ({ft_schema_valid/ft_total*100:.1f}%)")
print(f"Invalid Output Rate: {ft_invalid}/{ft_total} ({ft_invalid/ft_total*100:.1f}%)")

# Compare
print("\n" + "="*60)
print("COMPARISON: Base Model vs Fine-Tuned (r16_alpha64)")
print("-"*60)
print(f"{'Metric':<25} {'Base':>8} {'FT r16':>8} {'Change':>8}")
print("-"*35)
print(f"{'Exact Match':<25} {base_em/base_total*100:>7.1f}% {ft_em/ft_total*100:>7.1f}% {(ft_em/ft_total*100-base_em/base_total*100):>+7.1f}%")
print(f"{'Schema Validity':<25} {base_schema_valid/base_total*100:>7.1f}% {ft_schema_valid/ft_total*100:>7.1f}% {(ft_schema_valid/ft_total*100-base_schema_valid/base_total*100):>+7.1f}%")
print(f"{'Invalid Output Rate':<25} {base_invalid/base_total*100:>7.1f}% {ft_invalid/ft_total*100:>7.1f}% {(ft_invalid/ft_total*100-base_invalid/base_total*100):>+7.1f}%")

# Save results
results = {
    "base_exact_match_pct": base_em/base_total*100,
    "base_schema_validity_pct": base_schema_valid/base_total*100,
    "base_invalid_rate_pct": base_invalid/base_total*100,
    "ft_exact_match_pct": ft_em/ft_total*100,
    "ft_schema_validity_pct": ft_schema_valid/ft_total*100,
    "ft_invalid_rate_pct": ft_invalid/ft_total*100,
    "num_examples": ft_total,
    "best_config": {"rank": 16, "alpha": 64, "lr": 0.0003},
    "final_val_loss": 1.3524
}

output_file = r"C:\Projects\Fine-Tuned LLM with QLoRA\reports\experiment5_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_file}")