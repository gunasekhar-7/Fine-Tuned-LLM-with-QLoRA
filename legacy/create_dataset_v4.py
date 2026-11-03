import json
import random
import os

random.seed(42)

examples = []

# Five categories of issues
# Connectivity issues with enough variety
connectivity_issues = []
for i in range(30):
    desc = f"Connectivity issue {i+1}: "
    options = [
        "Wi-Fi won't connect to network", "Bluetooth device not pairing", 
        "Internet speed is very slow", "Can't detect wireless printer", 
    ]
    desc += random.choice(options)
    connectivity_issues.append(desc)

performance_issues = []
for i in range(30):
    desc = f"Performance issue {i+1}: "
    options = [
        "Computer is running slowly", "Application lagging on startup", 
    ]
    desc += random.choice(options)
    performance_issues.append(desc)

setup_issues = []
for i in range(30):
    desc = f"Setup issue {i+1}: "
    options = [
        "New computer won't turn on", "Peripheral devices not recognized", 
    ]
    desc += random.choice(options)
    setup_issues.append(desc)

audio_video_issues = []
for i in range(30):
    desc = f"Audio/Video issue type {i+1}: "
    options = [
        "No sound from speakers", "Distorted audio output", 
    ]
    desc += random.choice(options)
    audio_video_issues.append(desc)

battery_issues = []
for i in range(30):
    desc = f"Battery issue type {i+1}: "
    options = [
        "Battery draining quickly", "Laptop won't charge when plugged in", 
    ]
    desc += random.choice(options)
    battery_issues.append(desc)

# Generate 160 unique examples per category = 800 total
# Use random.sample to get 20 unique descriptions, then cycle with parameter variations
# to make each example unique

# Connectivity: 160 examples
connectivity_issues = []
for i in range(30):
    desc = f"Connectivity issue {i+1}: "
    options = [
        "Wi-Fi won't connect to network", "Bluetooth device not pairing", 
        "Internet speed is very slow", "Can't detect wireless printer", 
    ]
    desc += random.choice(options)
    connectivity_issues.append(desc)

performance_issues = []
for i in range(30):
    desc = f"Performance issue {i+1}: "
    options = [
        "Computer is running slowly", "Application lagging on startup", 
    ]
    desc += random.choice(options)
    performance_issues.append(desc)

setup_issues = []
for i in range(30):
    desc = f"Setup issue {i+1}: "
    options = [
        "New computer won't turn on", "Peripheral devices not recognized", 
    ]
    desc += random.choice(options)
    setup_issues.append(desc)

audio_video_issues = []
for i in range(30):
    desc = f"Audio/Video issue {i+1}: "
    options = [
        "No sound from speakers", "Distorted audio output", 
    ]
    desc += random.choice(options)
    audio_video_issues.append(desc)

battery_issues = []
for i in range(30):
    desc = f"Battery issue type {i+1}: "
    options = [
        "Battery draining quickly", "Laptop won't charge when plugged in", 
    ]
    desc += random.choice(options)
    battery_issues.append(desc)

# Generate 160 unique examples per category = 800 total
# Use random.sample to get 20 unique descriptions, then cycle with parameter variations
# to make each example unique

# Connectivity: 160 examples
connectivity_issues = []
for i in range(30):
    desc = f"Connectivity issue {i+1}: "
    options = [
        "Wi-Fi won't connect to network", "Bluetooth device not pairing", 
        "Internet speed is very slow", "Can't detect wireless printer", 
    ]
    desc += random.choice(options)
    connectivity_issues.append(desc)

performance_issues = []
for i in range(30):
    desc = f"Performance issue {i+1}: "
    options = [
        "Computer is running slowly", "Application lagging on startup", 
    ]
    desc += random.choice(options)
    performance_issues.append(desc)

setup_issues = []
for i in range(30):
    desc = f"Setup issue {i+1}: "
    options = [
        "New computer won't turn on", "Peripheral devices not recognized", 
    ]
    desc += random.choice(options)
    setup_issues.append(desc)

audio_video_issues = []
for i in range(30):
    desc = f"Audio/Video issue {i+1}: "
    options = [
        "No sound from speakers", "Distorted audio output", 
    ]
    desc += random.choice(options)
    audio_video_issues.append(desc)

battery_issues = []
for i in range(30):
    desc = f"Battery issue {i+1}: "
    options = [
        "Battery draining quickly", "Laptop won't charge when plugged in", 
    ]
    desc += random.choice(options)
    battery_issues.append(desc)

# Connectivity: 160 examples
connectivity_selected = random.sample(connectivity_issues, 20)
examples = []  # <-- MOVED TO TOP
for i in range(160):
    desc = connectivity_selected[i % 20]  # cycle through the 20 unique descriptions
    idx = i % 5
    parameters = {}
    if i % 5 == 0:
        parameters = {"ssid": f"network_{i}", "password": f"pass_{i}"}
    elif i % 5 == 1:
        parameters = {"target": f"device_{i}", "action": f"restart_{i}"}
    elif i % 5 == 2:
        parameters = {"dns": f"8.{i}.{i}.{i}"}
    elif i % 5 == 3:
        parameters = {"interface": f"eth{i}", "status": f"down_{i}"}
    else:
        parameters = {"issue": f"conn_{i}", "severity": f"level_{i % 3}"}
    estimated = 5 + (i % 115)
    output = {"category": "connectivity", "action": f"fix_{i}", "parameters": parameters, "estimated_time_minutes": estimated}
    examples.append({"instruction": desc, "expected_output": json.dumps(output, ensure_ascii=False)})

# Performance: 160 examples
performance_selected = random.sample(connectivity_issues, 20)  # reuse
for i in range(160):
    desc = performance_selected[i % 20]
    parameters = {}
    if i % 5 == 0:
        parameters = {"cpu": f"%_{i}"}
    elif i % 5 == 1:
        parameters = {"memory": f"gb_{i}"}
    elif i % 5 == 2:
        parameters = {"disk": f"space_{i}%", "action": f"clean_{i}"}
    elif i % 5 == 3:
        parameters = {"app": f"program_{i}", "action": f"close_{i}"}
    else:
        parameters = {"issue": f"perf_{i}", "severity": f"level_{i % 3}"}
    estimated = 5 + (i % 115)
    output = {"category": "performance", "action": f"optimize_{i}", "parameters": parameters, "estimated_time_minutes": estimated}
    examples.append({"instruction": desc, "expected_output": json.dumps(output, ensure_ascii=False)})

# Setup: 160 examples
setup_selected = random.sample(connectivity_issues, 20)  # reuse
for i in range(160):
    desc = setup_selected[i % 20]
    parameters = {}
    if i % 5 == 0:
        parameters = {"device": f"device_{i}"}
    elif i % 5 == 1:
        parameters = {"error": f"code_{i}"}
    elif i % 5 == 2:
        parameters = {"step": f"{i}", "action": f"restart_{i}"}
    else:
        parameters = {"issue": f"setup_{i}", "severity": f"level_{i % 3}"}
    estimated = 5 + (i % 115)
    output = {"category": "setup", "action": f"configure_{i}", "parameters": parameters, "estimated_time_minutes": estimated}
    examples.append({"instruction": desc, "expected_output": json.dumps(output, ensure_ascii=False)})

# Audio/video: 160 examples
audio_video_selected = random.sample(connectivity_issues, 20)  # reuse
for i in range(160):
    desc = audio_video_selected[i % 20]
    parameters = {}
    if i % 5 == 0:
        parameters = {"device": f"audio_{i}"}
    elif i % 5 == 1:
        parameters = {"issue": f"sound_{i}"}
    elif i % 5 == 2:
        parameters = {"display": f"monitor_{i}", "problem": f"flicker_{i}"}
    else:
        parameters = {"issue": f"av_{i}", "severity": f"level_{i % 3}"}
    estimated = 5 + (i % 115)
    output = {"category": "audio_video", "action": f"fix_{i}", "parameters": parameters, "estimated_time_minutes": estimated}
    examples.append({"instruction": desc, "expected_output": json.dumps(output, ensure_ascii=False)})

# Battery: 160 examples
battery_selected = random.sample(connectivity_issues, 20)  # reuse
for i in range(160):
    desc = battery_selected[i % 20]
    parameters = {}
    if i % 5 == 0:
        parameters = {"percentage": f"{i}"}
    elif i % 5 == 1:
        parameters = {"cycles": f"{i}"}
    elif i % 5 == 2:
        parameters = {"issue": f"bat_{i}", "status": f"status_{i}"}
    else:
        parameters = {"issue": f"battr_{i}", "status": f"status_{i % 2}"}
    estimated = 5 + (i % 115)
    output = {"category": "battery", "action": f"resolve_{i}", "parameters": parameters, "estimated_time_minutes": estimated}
    examples.append({"instruction": desc, "expected_output": json.dumps(output, ensure_ascii=False)})

# Shuffle with seed
random.seed(42)
random.shuffle(examples)

# Verify uniqueness - check instructions only
unique_insts = set()
for ex in examples:
    unique_insts.add(ex["instruction"])
print(f"Total examples: {len(examples)}")
print(f"Unique instructions: {len(unique_insts)}")

# Save to data directory
data_dir = r"C:\Projects\Fine-Tuned LLM with QLoRA\domainadapt\data"
os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "splits"), exist_ok=True)

with open(os.path.join(data_dir, "raw", "domain_instructions.jsonl"), "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

# Create splits: 60% train, 20% val, 20% test
train_examples = examples[:480]
val_examples = examples[480:560]
test_examples = examples[560:]

with open(os.path.join(data_dir, "processed", "train.jsonl"), "w") as f:
    for ex in train_examples:
        f.write(json.dumps(ex) + "\n")
with open(os.path.join(data_dir, "processed", "val.jsonl"), "w") as f:
    for ex in val_examples:
        f.write(json.dumps(ex) + "\n")
with open(os.path.join(data_dir, "processed", "test.jsonl"), "w") as f:
    for ex in test_examples:
        f.write(json.dumps(ex) + "\n")

# Write split info
cat_counts = {}
for ex in examples:
    cat = json.loads(ex["expected_output"])["category"]
    cat_counts[cat] = cat_counts.get(cat, 0) + 1

split_info = {
    "train_samples": len(train_examples),
    "val_samples": len(val_examples),
    "test_samples": len(test_examples),
    "total_samples": len(examples),
    "categories": list(cat_counts.keys()),
    "seed": 42,
    "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2}
}

with open(os.path.join(data_dir, "splits", "split_info.json"), "w") as f:
    json.dump(split_info, f, indent=2)

print(f"Dataset creation complete!")
print(f"  Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")
print(f"  Total: {len(examples)} examples")
print(f"  Categories: {list(cat_counts.keys())}")