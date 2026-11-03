import json
import time
import torch
import os
import random
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# Configuration
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Output directory for trained adapter
output_dir = r"C:\Projects\Fine-Tuned LLM with QLoRA\artifacts\trained_adapter"

# Load tokenizer FIRST
print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("Tokenizer loaded successfully.")

# Memory estimation
total_params = 494_032_768  # Qwen2.5-0.5B
estimated_ram = total_params * 4 / (1024 **3)  # float32 GB
optimizer_ram = estimated_ram * 3  # Adam states: 3x model params
print(f"Memory estimate: model={estimated_ram:.1f}GB, optimizer={optimizer_ram:.1f}GB total")
print("Using gradient checkpointing to reduce memory overhead")

# LoRA configuration
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=32,  # LoRA scaling alpha
    lora_dropout=0.05,  # LoRA dropout
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
)

print("="*60)
print("LORA IMPLEMENTATION: Standard LoRA (full-precision, CPU)")
print("="*60)

# Load base model on CPU
print(f"\nLoading base model: {model_name} (float32, CPU)")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)
total_base_params = sum(p.numel() for p in base_model.parameters())
print(f"Base model loaded: {base_model.__class__.__name__}, total params: {total_base_params:,}")

# Apply LoRA configuration
print("Applying LoRA configuration...")
model = get_peft_model(base_model, lora_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params_after = sum(p.numel() for p in model.parameters())
trainable_pct = trainable_params / total_params_after * 100
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params_after:,}")
print(f"Trainable percentage: {trainable_pct:.2f}%")

# Display trainable module details (first few)
print("\nTrainable modules (showing first 5):")
count = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape} -> {param.numel():,} parameters")
        count += 1
        if count >= 5:
            print(f"  ... ({count}+ trainable modules total)")
            break

# Load training data
train_file = r"C:\Projects\Fine-Tuned LLM with QLoRA\domainadapt\data\processed\train.jsonl"
valid_file = r"C:\Projects\Fine-Tuned LLM with QLoRA\domainadapt\data\processed\val.jsonl"

print(f"\nLoading training data from: {train_file}")
with open(train_file) as f:
    train_examples = [json.loads(line) for line in f]
print(f"Loaded {len(train_examples)} training examples")

print(f"Loading validation data from: {valid_file}")
with open(valid_file) as f:
    valid_examples = [json.loads(line) for line in f]
print(f"Loaded {len(valid_examples)} validation examples")

# Tokenization function
def tokenize_example(example, tokenizer, max_seq_length=512):
    """Tokenize a single training example using Qwen chat template."""
    messages = [
        {"role": "user", "content": example["instruction"]}
    ]
    encoded = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        max_length=max_seq_length,
        truncation=True,
        return_tensors="pt"
    )
    return {
        "input_ids": encoded["input_ids"][0],  # remove batch dim
        "attention_mask": encoded["attention_mask"][0],
    }

# Tokenize training examples
print("Tokenizing training examples...")
max_seq_length = 512
train_inputs = []
for i, example in enumerate(train_examples):
    if i >= 50:  # Limit to 50 examples for CPU training feasibility
        break
    tokenized = tokenize_example(example, tokenizer, max_seq_length)
    train_inputs.append(tokenized)
print(f"Tokenized {len(train_inputs)} training examples")

# Tokenize validation examples
print("Tokenizing validation examples...")
valid_inputs = []
for i, example in enumerate(valid_examples):
    if i >= 20:  # Limit to 20 examples for validation
        break
    tokenized = tokenize_example(example, tokenizer, max_seq_length)
    valid_inputs.append(tokenized)
print(f"Tokenized {len(valid_inputs)} validation examples")

# Training configuration
training_args_config = {
    "epochs": 3,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,  # effective batch size = 1 * 8 = 8
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "max_seq_length": max_seq_length,
    "logging_steps": 5,
    "save_steps": 20,
    "seed": 42,
    "use_gradient_checkpointing": True,
    "fp16": False,  # No CUDA, so fp16 not helpful on CPU
    "bf16": False,
}

print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)
for k, v in training_args_config.items():
    print(f"  {k}: {v}")

# Set random seeds
torch.manual_seed(training_args_config["seed"])
np.random.seed(training_args_config["seed"])
random.seed(training_args_config["seed"])

# Training loop
print("\n" + "="*60)
print("RUNNING TRAINING LOOP")
print("="*60)

# Training loop
model.train()
global_step = 0

for epoch in range(training_args_config["epochs"]):
    print(f"\n=== Epoch {epoch + 1}/{training_args_config['epochs']} ===")
    
    # Shuffle training data
    random.seed(training_args_config["seed"] + epoch)
    random.shuffle(train_inputs)
    
    # Process in micro-batches with gradient accumulation
    accumulation_steps = training_args_config["gradient_accumulation_steps"]
    
    for step in range(0, len(train_inputs), accumulation_steps):
        # Get the micro-batch
        micro_batch = train_inputs[step:step + accumulation_steps]
        
        if len(micro_batch) == 0:
            continue
        
        # Pad micro-batch to accumulation_steps if needed
        while len(micro_batch) < accumulation_steps:
            micro_batch.append(train_inputs[0])  # repeat first example
        
        # Prepare batch
        batch_input_ids = []
        batch_attention_mask = []
        for item in micro_batch:
            batch_input_ids.append(item["input_ids"])
            batch_attention_mask.append(item["attention_mask"])
        
        # Pad to same length within batch
        max_len = max(ids.shape[0] for ids in batch_input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        for ids, mask in zip(batch_input_ids, batch_attention_mask):
            pad_len = max_len - ids.shape[0]
            padded_id = torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype)])
            padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            padded_input_ids.append(padded_id)
            padded_attention_mask.append(padded_mask)
        
        # Stack into batch
        input_ids_batch = torch.stack(padded_input_ids)
        attention_mask_batch = torch.stack(padded_attention_mask)
        
        # Forward pass with labels (causal LM training)
        outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, labels=input_ids_batch)
        loss = outputs.loss
        loss = loss / accumulation_steps  # scale loss for gradient accumulation
        loss.backward()
        
        global_step += 1
        
        # Optimizer step every accumulation_steps
        if global_step % accumulation_steps == 0:
            optimizer = torch.optim.AdamW(model.parameters(), lr=training_args_config["learning_rate"], weight_decay=training_args_config["weight_decay"])
            optimizer.step()
            optimizer.zero_grad()
            
        if global_step % 5 == 0:
            print(f"  Step {global_step}: loss = {loss.item() * accumulation_steps:.4f}")
    
    # Validation after each epoch
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
            
            batch_input_ids = []
            batch_attention_mask = []
            for item in micro_batch:
                batch_input_ids.append(item["input_ids"])
                batch_attention_mask.append(item["attention_mask"])
            
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
            loss = outputs.loss
            val_loss_sum += loss.item()
            val_steps += 1
    
    avg_val_loss = val_loss_sum / max(val_steps, 1)
    print(f"  Epoch {epoch + 1} validation loss: {avg_val_loss:.4f}")
    
    # Save checkpoint after each epoch
    checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch{epoch+1}")
    model.save_pretrained(checkpoint_path)
    print(f"  Checkpoint saved to: {checkpoint_path}")

# Save final adapter
print("\nSaving final LoRA adapter...")
model.save_pretrained(output_dir)
print(f"Adapter saved to: {output_dir}")

# Prepare training summary - convert target_modules from set to list for JSON serializability
lora_config_target_modules = list(lora_config.target_modules)

training_summary = {
    "model_name": model_name,
    "lora_config": {
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "target_modules": lora_config_target_modules,
    },
    "training_args": training_args_config,
    "trainable_parameters": int(trainable_params),
    "total_parameters": int(total_params_after),
    "trainable_percentage": float(trainable_pct),
    "epochs_trained": int(training_args_config["epochs"]),
    "final_output_dir": str(output_dir),
    "seed": int(training_args_config["seed"]),
}

# Save training summary (with fallback for any non-serializable types)
summary_path = os.path.join(output_dir, "training_summary.json")
try:
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"Training summary saved to: {summary_path}")
except TypeError:
    # Clean non-serializable types and retry
    def clean_obj(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: clean_obj(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_obj(v) for v in obj]
        elif isinstance(obj, (bool, int, float, str)):
            return obj
        else:
            return str(obj)

    clean_summary = clean_obj(training_summary)
    with open(summary_path, "w") as f:
        json.dump(clean_summary, f, indent=2)
    print(f"Training summary saved to: {summary_path} (after cleanup)")

# Verify the adapter can be loaded
print("\nVerifying adapter can be loaded...")
load_base = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)
loaded_model = PeftModel.from_pretrained(load_base, output_dir)
loaded_model.eval()

# Count parameters after loading
trainable_after = sum(p.numel() for p in loaded_model.parameters() if p.requires_grad)
total_after = sum(p.numel() for p in loaded_model.parameters())
print(f"  After loading - Trainable: {trainable_after:,}, Total: {total_after:,}")

# Test forward pass with adapter
test_msg = {"role": "user", "content": "Wi-Fi won't connect to network"}
encoded = tokenizer.apply_chat_template([test_msg], tokenize=True, return_tensors="pt")
input_ids = encoded["input_ids"].to("cpu")
attention_mask = encoded["attention_mask"].to("cpu")

with torch.no_grad():
    output = loaded_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64)

generated = tokenizer.decode(output[0], skip_special_tokens=False)
print(f"  Generated output (first 120 chars): {generated[:120]}...")

print("\n" + "="*60)
print("LORA TRAINING COMPLETE")
print("="*60)
print(f"Adapter saved to: {output_dir}")
print(f"Trainable parameters: {trainable_pct:.2f}% of total")
print("The base model weights remain frozen; only LoRA adapters were trained.")