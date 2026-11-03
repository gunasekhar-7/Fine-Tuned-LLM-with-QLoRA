import json
import random
import os

random.seed(42)

# Five categories of issues
connectivity_issues = [
    "Wi-Fi won't connect to network", "Bluetooth device not pairing", 
    "Internet speed is very slow", "Can't detect wireless printer", 
    "Ethernet connection dropped", "Mobile hotspot not working", 
    "DNS resolution failing", "IP address conflict", 
    "Website not loading", "VPN connection blocked", 
]
performance_issues = [
    "Computer is running slowly", "Application lagging on startup", 
    "System feels unresponsive", "High CPU usage identified", 
    "Memory usage is critically high", "Disk is nearly full", 
    "Boot time has increased", "Applications freezing frequently", 
]
setup_issues = [
    "New computer won't turn on", "Peripheral devices not recognized", 
    "Operating system installation failed", "Driver installation failed", 
    "Updating BIOS failed", "Initial account setup stuck", 
]
audio_video_issues = [
    "No sound from speakers", "Distorted audio output", 
    "Screen flickering or lines", "Monitor not detected", 
    "Headphones mic not working", "Volume randomly changing", 
]
battery_issues = [
    "Battery draining quickly", "Laptop won't charge when plugged in", 
    "Battery percentage stuck at 1%", "System shutting down unexpectedly", 
]

# Generate exactly 20 unique examples per category
# Use random.sample to pick unique descriptions
connectivity_selected = random.sample(connectivity_issues, 20)
performance_selected = random.sample(performance_issues, 20)
setup_selected = random.sample(setup_issues, 20)
audio_video_selected = random.sample(audio_video_issues, 20)
battery_selected = random.sample(battery_issues, 20)

examples = []

# Connectivity: 20 examples
for i, desc in enumerate(connectivity_selected):
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

# Performance: 20 examples
for i, desc in enumerate(performance_selected):
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

# Setup: 20 examples
for i, desc in enumerate(setup_selected):
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

# Audio/video: 20 examples
for i, desc in enumerate(audio_video_selected):
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

# Battery: 20 examples
for i, desc in enumerate(battery_selected):
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

# Verify uniqueness
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
train_examples = examples[:60]
val_examples = examples[60:80]
test_examples = examples[80:]

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
print(f"  Total: {len(examples)} unique instructions")
print(f"  Categories: {list(cat_counts.keys())}")