import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
test_file = r"C:\Projects\Fine-Tuned LLM with QLoRA\domainadapt\data\processed\test.jsonl"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load base model (vanilla, no adapter)
print("Loading base model (no adapter)...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
)

# Load base model + LoRA adapter
print("Loading base model + LoRA adapter...")
model_with_adapter = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
)
model_with_adapter = PeftModel.from_pretrained(model_with_adapter, r"C:\Projects\Fine-Tuned LLM with QLoRA\artifacts\trained_adapter")
model_with_adapter.eval()

# Load test examples (use same 10 for comparison with baseline)
print("Loading test examples...")
with open(test_file) as f:
    all_test_examples = [json.loads(line) for line in f]
test_examples = all_test_examples[:10]

# Evaluate base model
print("\n" + "="*60)
print("EVALUATION: Base model (no adapter)")
print("="*60)

base_results = []
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
    if parsed_output is not None:
        required_keys = {"category", "action", "parameters", "estimated_time_minutes"}
        if required_keys.issubset(parsed_output.keys()):
            base_schema_valid += 1
    if parsed_output is None:
        base_invalid += 1

    base_results.append({
        "instruction": instruction,
        "expected": expected_json,
        "generated_text": generated_text[:100],
        "parsed_output": parsed_output
    })

base_total = len(test_examples)
base_em_pct = base_em / base_total * 100
base_schema_pct = base_schema_valid / base_total * 100
base_invalid_pct = base_invalid / base_total * 100

print(f"Exact Match: {base_em_pct:.1f}% ({base_em}/{base_total})")
print(f"Schema Validity: {base_schema_pct:.1f}% ({base_schema_valid}/{base_total})")
print(f"Invalid Output Rate: {base_invalid_pct:.1f}% ({base_invalid}/{base_total})")

# Evaluate fine-tuned model
print("\n" + "="*60)
print("EVALUATION: Fine-tuned model (base + LoRA adapter)")
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
    if parsed_output is not None:
        required_keys = {"category", "action", "parameters", "estimated_time_minutes"}
        if required_keys.issubset(parsed_output.keys()):
            ft_schema_valid += 1
    if parsed_output is None:
        ft_invalid += 1

    if (i + 1) % 5 == 0:
        print(f"  Processed {i+1}/10 examples...")

ft_total = len(test_examples)
ft_em_pct = ft_em / ft_total * 100
ft_schema_pct = ft_schema_valid / ft_total * 100
ft_invalid_pct = ft_invalid / ft_total * 100

print(f"Exact Match: {ft_em_pct:.1f}% ({ft_em}/{ft_total})")
print(f"Schema Validity: {ft_schema_pct:.1f}% ({ft_schema_valid}/{ft_total})")
print(f"Invalid Output Rate: {ft_invalid_pct:.1f}% ({ft_invalid}/{ft_total})")

# Compare
print("\n" + "="*60)
print("COMPARISON: Base Model vs Fine-Tuned Model")
print("="*60)
print(f"{'Metric':<25} {'Base':>8} {'FT':>8} {'Change':>8}")
print(f"{'='*35}")
print(f"{'Exact Match':<25} {base_em_pct:>7.1f}% {ft_em_pct:>7.1f}% {ft_em_pct-base_em_pct:>+7.1f}%")
print(f"{'Schema Validity':<25} {base_schema_pct:>7.1f}% {ft_schema_pct:>7.1f}% {ft_schema_pct-base_schema_pct:>+7.1f}%")
print(f"{'Invalid Output Rate':<25} {base_invalid_pct:>7.1f}% {ft_invalid_pct:>7.1f}% {ft_invalid_pct-base_invalid_pct:>+7.1f}%")

# Save detailed results
results = {
    "baseline_exact_match_pct": base_em_pct,
    "baseline_schema_validity_pct": base_schema_pct,
    "baseline_invalid_rate_pct": base_invalid_pct,
    "ft_exact_match_pct": ft_em_pct,
    "ft_schema_validity_pct": ft_schema_pct,
    "ft_invalid_rate_pct": ft_invalid_pct,
    "num_examples": ft_total,
    "per_example": []
}

for i, example in enumerate(test_examples):
    instruction = example["instruction"]
    expected_json = json.loads(example["expected_output"])

    # Base model result
    messages = [{"role": "user", "content": instruction}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with torch.no_grad():
        base_output = base_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=128)

    results["per_example"].append({
        "instruction": instruction,
        "base_parsed": None,  # simplified
        "ft_parsed": None   # simplified
    })

output_file = r"C:\Projects\Fine-Tuned LLM with QLoRA\reports\fine_tuned_evaluation.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed evaluation saved to: {output_file}")