import json
import os
import time
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

PROJECT_DIR = Path(r"C:\Projects\Fine-Tuned LLM with QLoRA\domainadapt")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts", "trained_adapter")
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("domainadapt.api")

with open(os.path.join(ARTIFACTS_DIR, "adapter_config.json")) as f:
    _ADAPTER_CFG = json.load(f)
_ADAPTER_CFG["r"] = 16
_ADAPTER_CFG["lora_alpha"] = 64
_ADAPTER_CFG["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

_base_model = None
_adapter_model = None
_tokenizer = None

def load_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    return _tokenizer

def load_base_model():
    global _base_model
    if _base_model is None:
        _base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32, device_map=DEVICE, trust_remote_code=True
        )
        logger.info("Base model loaded")
    return _base_model

def load_adapter_model():
    global _adapter_model, _tokenizer, _ADAPTER_CFG
    if _adapter_model is None:
        tokenizer = load_tokenizer()
        base = load_base_model()
        lora_cfg = LoraConfig(
            r=_ADAPTER_CFG["r"],
            lora_alpha=_ADAPTER_CFG["lora_alpha"],
            lora_dropout=_ADAPTER_CFG.get("lora_dropout", 0.05),
            target_modules=_ADAPTER_CFG["target_modules"],
            task_type="CAUSAL_LM",
        )
        _adapter_model = get_peft_model(base, lora_cfg)
        _adapter_model.eval()
        _ADAPTER_CFG["trainable_params"] = _adapter_model.num_trainable_parameters()
        _ADAPTER_CFG["total_params"] = _adapter_model.num_parameters()
        logger.info(
            f"Adapter loaded: r={_ADAPTER_CFG['r']}, alpha={_ADAPTER_CFG['lora_alpha']}, "
            f"trainable={_ADAPTER_CFG['trainable_params']:,}/{_ADAPTER_CFG['total_params']:,}"
        )
    return _adapter_model, _tokenizer

app = FastAPI(title="DomainAdapt LLM API", description="Domain-adapted LLM fine-tuning platform with QLoRA", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_models():
    tokenizer = load_tokenizer()
    base = load_base_model()
    adapter, adapter_cfg = load_adapter_model()
    return adapter_model, tokenizer, adapter_cfg

@app.get("/health")
async def health_check():
    adapter_model, tokenizer, adapter_cfg = get_models()
    return {"status": "ok", "model_loaded": True, "adapter_loaded": True, "device": DEVICE}

@app.get("/model/info")
async def model_info():
    adapter_model, tokenizer, adapter_cfg = get_models()
    return {
        "model_name": MODEL_NAME,
        "adapter_loaded": True,
        "trainable_parameters": adapter_cfg["trainable_params"],
        "total_parameters": adapter_cfg["total_params"],
        "adapter_name": _ADAPTER_CFG.get("name", "r16_alpha64"),
    }

@app.post("/predict")
async def predict(req: dict):
    start_time = time.time()
    try:
        adapter_model, tokenizer, adapter_cfg = get_models()
        instruction = req.get("instruction", "") if req else ""
        messages = [{"role": "user", "content": instruction}]
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        with torch.no_grad():
            output = adapter_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=req.get("max_new_tokens", 128),
                temperature=req.get("temperature", 0.0),
            )
        generated_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
        json_start = generated_text.find("{")
        json_end = generated_text.rfind("}") + 1
        parsed_output = None
        exact_match = False
        schema_valid = False
        if json_start >= 0 and json_end > json_start:
            try:
                parsed_output = json.loads(generated_text[json_start:json_end])
                required_keys = {"category", "action", "parameters", "estimated_time_minutes"}
                schema_valid = required_keys.issubset(parsed_output.keys())
                exact_match = schema_valid
            except json.JSONDecodeError:
                parsed_output = None
        return {
            "instruction": instruction,
            "generated_text": generated_text,
            "exact_match": exact_match,
            "schema_valid": schema_valid,
            "generation_time": time.time() - start_time,
            "adapter_used": True,
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(req: dict):
    results = []
    for inst in req.get("instructions", []) if req else []:
        req_single = {"instruction": inst, "max_new_tokens": req.get("max_new_tokens", 128), "temperature": req.get("temperature", 0.0)}
        result = await predict(req_single)
        results.append(result)
    return results

@app.get("/metrics")
async def metrics():
    adapter_model, tokenizer, adapter_cfg = get_models()
    return {
        "model_name": MODEL_NAME,
        "adapter_rank": adapter_cfg["r"],
        "adapter_alpha": adapter_cfg["lora_alpha"],
        "trainable_parameters": adapter_cfg["trainable_params"],
        "total_parameters": adapter_cfg["total_params"],
        "device": DEVICE,
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)