import json
import torch
import os
import random
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Configuration
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
train_file = r"C:\Projects\Fine-Tuned LLM with QLoRA\domainadapt\data\processed\train.jsonl"
valid_file = r"C:\Projects\Fine-Tuned LLM with QLoRA\domainadapt\data\processed\val.jsonl"
output_dir_base = r"C:\Projects\Fine-Tuned LLM with QLoRA\artifacts\experiments"

# Fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Experiment configurations (3 LoRA rank variants)
experiments = [
    {
        "name": "r4_alpha16",
        "lora_config": LoraConfig(
            r=4, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
    },
    {
        "name": "r8_alpha32",  # baseline
        "lora_config": LoraConfig(
            r=8, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
    },
    {
        "name": "r16_alpha64",
        "lora_config": LoraConfig(
            r=16, lora_alpha=64, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
    },
]

# Training configuration (fixed for all experiments)
training_config = {
    "epochs": 3,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": None,  # Will be set per experiment
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "max_seq_length": 512,
    "logging_steps": 5,
    "save_steps": 20,
    "seed": SEED,
    "use_gradient_checkpointing": True,
    "fp16": False,
    "bf16": False,
}

# Override learning rate per experiment
experiment_lrs = {
    "r4_alpha16": 1e-4,
    "r8_alpha32": 2e-4,
    "r16_alpha64": 3e-4,
}


def run_experiment(exp):
    """Run a single LoRA experiment and return training results."""
    name = exp["name"]
    lora_config = exp["lora_config"]
    lr = experiment_lrs[name]
    
    print("="*60)
    print(f"EXPERIMENT: {name}")
    print(f"  LoRA r={lora_config.r}, alpha={lora_config.lora_alpha}, lr={lr}")
    print("="*60)
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    model = get_peft_model(base_model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = trainable_params / total_params * 100
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    
    # Load training data
    print("Loading training data...")
    with open(train_file) as f:
        train_examples = [json.loads(line) for line in f]
    with open(valid_file) as f:
        valid_examples = [json.loads(line) for line in f]
    print(f"  {len(train_examples)} training examples, {len(valid_examples)} validation examples")
    
    # Tokenize data (limited subset for CPU feasibility)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    def tokenize_example(example, max_seq_length=512):
        messages = [{"role": "user", "content": example["instruction"]}]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, max_length=max_seq_length, truncation=True, return_tensors="pt"
        )
        return {"input_ids": encoded["input_ids"][0], "attention_mask": encoded["attention_mask"][0]}
    
    train_inputs = [tokenize_example(ex) for ex in train_examples[:50]]  # 50 for speed
    valid_inputs = [tokenize_example(ex) for ex in valid_examples[:20]]  # 20 for speed
    
    # Training setup
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=training_config["weight_decay"])
    
    # Training loop
    print("Starting training...")
    losses = []
    val_losses = []
    
    for epoch in range(training_config["epochs"]):
        # Shuffle
        random.seed(SEED + epoch)
        random.shuffle(train_inputs)
        
        # Train
        accumulation_steps = training_config["gradient_accumulation_steps"]
        epoch_loss = 0
        epoch_steps = 0
        
        for step in range(0, len(train_inputs), accumulation_steps):
            micro_batch = train_inputs[step:step + accumulation_steps]
            if len(micro_batch) == 0:
                continue
            while len(micro_batch) < accumulation_steps:
                micro_batch.append(train_inputs[0])
            
            batch_input_ids = [item["input_ids"] for item in micro_batch]
            batch_attention_mask = [item["attention_mask"] for item in micro_batch]
            
            max_len = max(ids.shape[0] for ids in batch_input_ids)
            padded_input_ids = []
            padded_attention_mask = []
            for ids, mask in zip(batch_input_ids, batch_attention_mask):
                pad_len = max_len - ids.shape[0]
                padded_id = torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype)])
                padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
                padded_input_ids.append(padded_id)
                padded_attention_mask.append(padded_mask)
            
            input_ids_batch = torch.stack(padded_input_ids)
            attention_mask_batch = torch.stack(padded_attention_mask)
            
            outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, labels=input_ids_batch)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * accumulation_steps
            epoch_steps += 1
            
            if (epoch_steps + 1) % accumulation_steps == 0 or step + accumulation_steps >= len(train_inputs):
                optimizer.step()
                optimizer.zero_grad()
        
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        losses.append(avg_train_loss)
        print(f"  Epoch {epoch+1}/3: train loss = {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss_sum = 0
        val_steps = 0
        with torch.no_grad():
            for step in range(0, len(valid_inputs), accumulation_steps):
                micro_batch = valid_inputs[step:step + accumulation_steps]
                if len(micro_batch) == 0:
                    continue
                while len(micro_batch) < accumulation_steps:
                    micro_batch.append(valid_inputs[0])
                
                batch_input_ids = [item["input_ids"] for item in micro_batch]
                batch_attention_mask = [item["attention_mask"] for item in micro_batch]
                
                max_len = max(ids.shape[0] for ids in batch_input_ids)
                padded_input_ids = []
                padded_attention_mask = []
                for ids, mask in zip(batch_input_ids, batch_attention_mask):
                    pad_len = max_len - ids.shape[0]
                    padded_id = torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype)])
                    padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
                    padded_input_ids.append(padded_id)
                    padded_attention_mask.append(padded_mask)
                
                input_ids_batch = torch.stack(padded_input_ids)
                attention_mask_batch = torch.stack(padded_attention_mask)
                
                outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, labels=input_ids_batch)
                val_loss_sum += outputs.loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss_sum / max(val_steps, 1)
        val_losses.append(avg_val_loss)
        print(f"  Epoch {epoch+1}/3: val loss = {avg_val_loss:.4f}")
    
    # Compute trainable parameters
    final_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    final_total = sum(p.numel() for p in model.parameters())
    
    # Save adapter
    adapter_dir = os.path.join(train_output_dir, name)
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    print(f"  Adapter saved to: {adapter_dir}")
    
    # Return results
    return {
        "name": name,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "learning_rate": lr,
        "trainable_parameters": final_trainable,
        "total_parameters": final_total,
        "trainable_percentage": final_trainable / final_total * 100,
        "train_losses": losses,
        "val_losses": val_losses,
        "final_train_loss": losses[-1] if losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "adapter_dir": adapter_dir,
    }


# Run all experiments
train_output_dir = r"C:\Projects\Fine-Tuned LLM with QLoRA\artifacts\experiments"
os.makedirs(train_output_dir, exist_ok=True)

print("="*60)
print("LORA HYPERPARAMETER EXPERIMENTS")
print("="*60)
print(f"Running {len(experiments)} experiments with 3 epochs each on CPU")
print("Each experiment may take several minutes.")
print()

results = []
for i, exp in enumerate(experiments):
    print(f"\\n[{i+1}/{len(experiments)}] Starting experiment...")
    result = run_experiment(exp)
    results.append(result)
    
    # Save experiment result
    result_path = os.path.join(train_output_dir, f"exp_{exp['name']}_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to: {result_path}")

# Compare results
print("="*60)
print("EXPERIMENT COMPARISON")
print("="*60)
print(f"{'Experiment':<15} {'Train Loss':<15} {'Val Loss':<15} {'Trainable %':<15}")
print("-"*60)
for r in results:
    print(f"{r['name']:<15} {str(r['final_train_loss']):<15} {str(r['final_val_loss']):<15} {r['trainable_percentage']:<15.2f}%")

# Select best based on validation loss
best = min(results, key=lambda x: x["final_val_loss"])
print(f"\\nBest experiment: {best['name']} (val loss = {best['final_val_loss']:.4f})")

# Save comparison summary
comparison = {
    "experiments": results,
    "best_experiment": best["name"],
    "best_val_loss": best["final_val_loss"],
    "total_experiments": len(results),
}
comparison_path = os.path.join(train_output_dir, "experiment_comparison.json")
with open(comparison_path, "w") as f:
    json.dump(comparison, f, indent=2)
print(f"\nComparison summary saved to: {comparison_path}")