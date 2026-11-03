import sys
sys.path.insert(0, r'C:\Projects\Fine-Tuned LLM with QLoRA\domainadapt\api')
from main import app
import json

# Test using FastAPI's test client
from fastapi.testclient import TestClient

client = TestClient(app)

# Test /health
print("=== Test /health ===")
resp = client.get("/health")
print(f"Status: {resp.status_code}")
print(f"Body: {resp.json()}")

# Test /model/info
print("\n=== Test /model/info ===")
resp = client.get("/model/info")
print(f"Status: {resp.status_code}")
print(f"Body: {resp.json()}")

# Test /predict
print("\n=== Test /predict ===")
resp = client.post("/predict", json={"instruction": "Wi-Fi won't connect to network", "max_new_tokens": 64})
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Instruction: {data['instruction']}")
    print(f"Generated text: {data['generated_text'][:100]}...")
    print(f"Exact match: {data['exact_match']}")
    print(f"Schema valid: {data['schema_valid']}")
    print(f"Generation time: {data['generation_time']:.3f}s")
    print(f"Adapter used: {data['adapter_used']}")
else:
    print(f"Error: {resp.json()}")

# Test /predict/batch
print("\n=== Test /predict/batch ===")
resp = client.post("/predict/batch", json={"instructions": ["Wi-Fi won't connect to network", "Slow computer performance"], "max_new_tokens": 64})
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Number of results: {len(data)}")
    for i, r in enumerate(data):
        print(f"  {i}: {r['generated_text'][:80]}...")