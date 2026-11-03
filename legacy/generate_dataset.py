import json
import random
import os

random.seed(42)

categories = {
    "connectivity": {
        "descriptions": [
            "Wi-Fi won't connect to network",
            "Bluetooth device not pairing",
            "Internet speed is very slow",
            "Can't detect wireless printer",
            "Ethernet connection dropped",
            "Mobile hotspot not working",
            "DNS resolution failing",
            "IP address conflict",
        ],
        "actions": ["restart_router", "reset_network", "update_drivers", "reconfigure_dns", "check_cables"],
        "parameter_templates": [
            {"ssid": "STR", "password": "STR"},
            {"device_name": "STR", "pair_mode": "STR"},
            {"dns_server": "STR"},
            {"interface": "STR"},
        ]
    },
    "performance": {
        "descriptions": [
            "Computer is running slowly",
            "Application lagging on startup",
            "System feels unresponsive",
            "High CPU usage identified",
            "Memory usage is critically high",
            "Disk is nearly full",
            "Boot time has increased",
            "Applications freezing frequently",
        ],
        "actions": ["cleanup_disk", "disable_startup_apps", "add_ram", "run_disk_check", "update_os"],
        "parameter_templates": [
            {"target": "disk", "action": "cleanup"},
            {"memory_gb": "NUM"},
            {"days_since_clean": "NUM"},
            {"app_name": "STR"},
        ]
    },
    "setup": {
        "descriptions": [
            "New computer won't turn on",
            "Peripheral devices not recognized",
            "Operating system installation failed",
            "Driver installation failed",
            "Updating BIOS failed",
            "Initial account setup stuck",
            "Network configuration wizard failing",
            "Software license activation failed",
        ],
        "actions": ["check_power", "reinstall_drivers", "system_restore", "retry_update", "contact_support"],
        "parameter_templates": [
            {"device": "STR"},
            {"error_code": "STR"},
            {"step": "NUM"},
        ]
    },
    "audio_video": {
        "descriptions": [
            "No sound from speakers",
            "Distorted audio output",
            "Screen flickering or lines",
            "Monitor not detected",
            "Headphones mic not working",
            "Volume randomly changing",
            "Video playback is choppy",
            "Audio out of sync with video",
        ],
        "actions": "update_audio_drivers, check_cables, adjust_settings, run_troubleshooter, replace_hardware",
        "parameter_templates": [
            {"device": "STR", "issue": "STR"},
            {"driver_version": "STR"},
        ]
    },
    "battery": {
        "descriptions": [
            "Battery draining quickly",
            "Laptop won't charge when plugged in",
            "Battery percentage stuck at 1%",
            "System shutting down unexpectedly",
            "Charging light not turning on",
            "Battery health has degraded",
            "Power plan not switching",
            "USB-C port not charging",
        ],
        "actions": "calibrate_battery, update_bios, check_charger, replace_battery, adjust_power_settings",
        "parameter_templates": [
            {"battery_percentage": "NUM"},
            {"charge_cycles": "NUM"},
            {"power_source": "STR"},
        ]
    }
}

# Generate 800 examples (about 160 per category, balanced)
examples = []
for cat_name, cat_data in categories.items():
    for i in range(160):
        idx = i % len(cat_data["descriptions"])
        desc = cat_data["descriptions"][idx]

        # Select action and parameters
        action = cat_data["actions"][i % len(cat_data["actions"])]

        # Generate parameters based on template
        param_template = random.choice(cat_data["parameter_templates"]) if "parameter_templates" in cat_data else {}
        parameters = {}
        for key, val_type in param_template.items():
            if val_type == "STR":
                parameters[key] = f"{cat_name}_{key}_{i}"
            elif val_type == "NUM":
                parameters[key] = random.randint(1, 100)

        # Add estimated time
        estimated = random.randint(5, 120)

        # Construct the expected JSON output
        output = {"category": cat_name, "action": action, "parameters": parameters, "estimated_time_minutes": estimated}
        output_json = json.dumps(output, ensure_ascii=False)

        example = {
            "instruction": desc,
            "expected_output": output_json
        }
        examples.append(example)

# Shuffle with seed
random.shuffle(examples)

# Verify balance
cat_counts = {}
for ex in examples:
    cat = json.loads(ex["expected_output"])["category"]
    cat_counts[cat] = cat_counts.get(cat, 0) + 1
print(f"Total examples: {len(examples)}")
print(f"Category distribution: {cat_counts}")

# Create splits: 600 train, 100 validation, 100 test
train_examples = examples[:600]
val_examples = examples[600:700]
test_examples = examples[700:800]

# Write to JSONL files
data_dir = r"C:\Projects\Fine-Tuned LLM with QLoRA\domainadapt\data"
os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "splits"), exist_ok=True)

with open(os.path.join(data_dir, "raw", "domain_instructions.jsonl"), "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

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
split_info = {
    "train_samples": len(train_examples),
    "val_samples": len(val_examples),
    "test_samples": len(test_examples),
    "total_samples": len(examples),
    "categories": list(categories.keys()),
    "seed": 42,
    "split_ratios": {"train": 0.75, "val": 0.125, "test": 0.125}
}

with open(os.path.join(data_dir, "splits", "split_info.json"), "w") as f:
    json.dump(split_info, f, indent=2)

# Verify a sample
print("\nSample training example:")
print(json.dumps(train_examples[0], indent=2))

print("\nSample test example:")
print(json.dumps(test_examples[0], indent=2))

print("\nDataset creation completed successfully!")