import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
test_file = r"C:\Projects\Fine-Tuned LLM with QLoRA\domainadapt\data\processed\test.jsonl"

# Load tokenizer
print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("Tokenizer loaded successfully.")

# Load base model on CPU
print(f"Loading base model: {model_name} (float32, CPU)")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)
print(f"Base model loaded: {base_model.__class__.__name__}, total params: {sum(p.numel() for p in base_model.parameters()):,}")

# Load test examples
print(f"Loading test examples from: {test_file}")
with open(test_file) as f:
    all_test_examples = [json.loads(line) for line in f]

# Use a small subset for baseline evaluation (first 10 examples for speed)
num_examples = 10
test_examples = all_test_examples[:num_examples]
print(f"Using {num_examples} examples for baseline evaluation")

# Evaluate base model
print("\n" + "="*60)
print("BASELINE EVALUATION: Base model on test set")
print("="*60)

results = []
em_count = 0
schema_valid_count = 0
invalid_count = 0
category_stats = {}

for i, example in enumerate(test_examples):
    instruction = example["instruction"]
    expected_json = json.loads(example["expected_output"])
    expected_category = expected_json["category"]
    
    # Initialize category stats if not seen yet
    if expected_category not in category_stats:
        category_stats[expected_category] = {"total": 0, "em": 0, "schema_valid": 0}
    
    category_stats[expected_category]["total"] += 1
    
    # Tokenize instruction using Qwen chat template
    messages = [
        {"role": "user", "content": instruction}
    ]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    # Generate output
    with torch.no_grad():
        start_time = time.time()
        outputs = base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
        )
        generation_time = time.time() - start_time
    
    # Decode generated text
    # Remove input tokens from output to get just the generated part
    generated_tokens = outputs[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    
    # Try to extract JSON from the generated text
    # Look for JSON object in the text
    json_start = generated_text.find("{")
    json_end = generated_text.rfind("}") + 1
    
    parsed_output = None
    if json_start >= 0 and json_end > json_start:
        json_candidate = generated_text[json_start:json_end]
        try:
            parsed_output = json.loads(json_candidate)
        except json.JSONDecodeError:
            parsed_output = None
    
    # Compute metrics
    em = False
    if parsed_output is not None:
        # Exact match: all fields including nested parameters must match
        if parsed_output == expected_json:
            em = True
            em_count += 1
        
        # Schema validity: check that all required keys exist
        required_keys = {"category", "action", "parameters", "estimated_time_minutes"}
        if required_keys.issubset(parsed_output.keys()):
            schema_valid_count += 1
    
    if parsed_output is None:
        invalid_count += 1
    
    # Category tracking
    if parsed_output is not None and required_keys.issubset(parsed_output.keys()):
        category_stats[expected_category]["schema_valid"] += 1
    
    results.append({
        "instruction": instruction,
        "expected_output": expected_json,
        "generated_text": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text,
        "parsed_output": parsed_output,
        "exact_match": em,
        "generation_time": generation_time
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{num_examples} examples...")

print(f"Done processing all {num_examples} examples.")

# Compute overall metrics
overall_em = em_count / num_examples * 100
overall_schema_valid = schema_valid_count / num_examples * 100
overall_invalid = invalid_count / num_examples * 100

print(f"\n{'Metric':<25} {'Value':>10}")
print(f"{'='*35}")
print(f"{'Exact Match':<25} {overall_em:>9.1f}%")
print(f"{'Schema Validity':<25} {overall_schema_valid:>9.1f}%")
print(f"{'Invalid Output Rate':<25} {overall_invalid:>9.1f}%")

print(f"\nPer-category analysis:")
for cat in sorted(category_stats.keys()):
    cs = category_stats[cat]
    em_pct = cs["em"] / cs["total"] * 100 if cs["total"] > 0 else 0
    sv_pct = cs["schema_valid"] / cs["total"] * 100 if cs["total"] > 0 else 0
    print(f"  {cat:<15}: EM={em_pct:5.1f}% ({cs['em']}/{cs['total']}), Schema={sv_pct:5.1f}% ({cs['schema_valid']}/{cs['total']})")

# Save predictions
output_file = r"C:\Projects\Fine-Tuned LLM with QLoRA\reports\baseline_predictions.json"
with open(output_file, "w") as f:
    json.dump({
        "model": model_name,
        "evaluation": "baseline_base_model",
        "num_examples": num_examples,
        "exact_match": overall_em,
        "schema_validity": overall_schema_valid,
        "invalid_rate": overall_invalid,
        "per_category": {cat: {"exact_match_pct": cs["em"] / cs["total"] * 100 if cs["total"] > 0 else 0, "schema_validity_pct": cs["schema_valid"] / cs["total"] * 100 if cs["total"] > 0 else 0, "count": cs["total"]} for cat, cs in category_stats.items()},
        "results": results
    }, f, indent=2)

print(f"\nBaseline predictions saved to: {output_file}")

# Also save a summary report
summary_file = r"C:\Projects\Fine-Tuned LLM with QLoRA\reports\\baseline_report.txt"
with open(summary_file, "w") as f:
    f.write("="*60 + "\n")
    f.write("BASELINE EVALUATION REPORT: Base Model (Qwen2.5-0.5B-Instruct)\n")
    f.write("="*60 + "\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Examples evaluated: {num_examples}\n")
    f.write(f"Exact Match: {overall_em:.1f}% ({em_count}/{num_examples})\n")
    f.write(f"Schema Validity: {overall_schema_valid:.1f}% ({schema_valid_count}/{num_examples})\n")
    f.write(f"Invalid Output Rate: {overall_invalid:.1f}% ({invalid_count}/{num_examples})\n")
    f.write("\nPer-category analysis:\n")
    for cat in sorted(category_stats.keys()):
        cs = category_stats[cat]
        em_pct = cs["em"] / cs["total"] * 100 if cs["total"] > 0 else 0
        sv_pct = cs["schema_valid"] / cs["total"] * 100 if cs["total"] > 0 else 0
        f.write(f"  {cat}: EM={em_pct:.1f}% ({cs['em']}/{cs['total']}), Schema={sv_pct:.1f}% ({cs['schema_valid']}/{cs['total']})\n")
    f.write("\n")

print(f"Baseline summary saved to: {summary_file}")