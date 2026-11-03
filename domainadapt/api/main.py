import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from domainadapt.src.utils.logger import setup_logger
from domainadapt.src.models.base_model import load_base_model, load_tokenizer, get_optimal_device_and_dtype
from domainadapt.src.models.adapter import load_finetuned_adapter, count_parameters
from domainadapt.src.inference.generator import DomainInferenceEngine, GenerationResult

logger = setup_logger("domainadapt.api", level=logging.INFO)

# Configuration
MODEL_NAME = os.getenv("DOMAINADAPT_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
ARTIFACTS_DIR = os.getenv(
    "DOMAINADAPT_ARTIFACTS", str(PROJECT_ROOT / "artifacts" / "trained_adapter")
)

# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
class PredictRequest(BaseModel):
    instruction: str = Field(..., min_length=1, max_length=2000, description="User troubleshooting prompt")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")
    max_new_tokens: int = Field(default=128, ge=1, le=512, description="Maximum tokens to generate")
    use_adapter: bool = Field(default=True, description="Whether to use fine-tuned adapter or base model")


class PredictResponse(BaseModel):
    instruction: str
    generated_text: str
    parsed_json: Optional[Dict[str, Any]] = None
    is_schema_valid: bool
    validation_error: Optional[str] = None
    latency_seconds: float
    tokens_generated: int
    adapter_used: bool
    correlation_id: str


class BatchPredictRequest(BaseModel):
    instructions: List[str] = Field(..., min_length=1, max_length=50, description="List of instructions")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    use_adapter: bool = Field(default=True)


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    total_samples: int
    total_latency_seconds: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    adapter_loaded: bool
    device: str
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    adapter_loaded: bool
    trainable_parameters: int
    total_parameters: int
    trainable_percentage: float
    device: str


class MetricsResponse(BaseModel):
    total_requests: int
    successful_predictions: int
    schema_valid_predictions: int
    schema_validity_rate: float
    average_latency_ms: float
    uptime_seconds: float


# ---------------------------------------------------------
# State & Metrics Tracker
# ---------------------------------------------------------
class APIState:
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_predictions = 0
        self.schema_valid_predictions = 0
        self.total_latency_ms = 0.0

        self.tokenizer = None
        self.base_model = None
        self.adapter_model = None
        self.base_engine = None
        self.adapter_engine = None
        self.device = "cpu"
        self.param_stats = {}

    def init_models(self):
        self.device, _ = get_optimal_device_and_dtype()
        logger.info(f"Loading tokenizer {MODEL_NAME}...")
        self.tokenizer = load_tokenizer(MODEL_NAME)
        logger.info(f"Loading base model {MODEL_NAME}...")
        self.base_model = load_base_model(MODEL_NAME, quantization="none", device=self.device)
        self.base_engine = DomainInferenceEngine(
            model=self.base_model,
            tokenizer=self.tokenizer,
            device=self.device,
            adapter_enabled=False,
        )

        adapter_path = Path(ARTIFACTS_DIR)
        config_file = adapter_path / "adapter_config.json"
        weights_file = adapter_path / "adapter_model.safetensors"

        if config_file.exists() and weights_file.exists() and weights_file.stat().st_size > 1024:
            try:
                logger.info(f"Loading trained adapter from {adapter_path}...")
                self.adapter_model = load_finetuned_adapter(self.base_model, adapter_path)
                self.param_stats = count_parameters(self.adapter_model)
                self.adapter_engine = DomainInferenceEngine(
                    model=self.adapter_model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    adapter_enabled=True,
                )
                logger.info("Adapter model loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load adapter weights: {e}. Running with base model.")
                self.adapter_model = None
                self.adapter_engine = self.base_engine
        else:
            logger.info("No trained adapter found; falling back to base model engine.")
            self.adapter_model = None
            self.adapter_engine = self.base_engine


state = APIState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    state.init_models()
    yield

# ---------------------------------------------------------
# FastAPI App & Middleware
# ---------------------------------------------------------
app = FastAPI(
    title="DomainAdapt LLM API",
    description="Production-grade domain-adapted LLM fine-tuning and inference platform with QLoRA/LoRA",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory sliding rate limiter: max 120 req/minute per IP
RATE_LIMIT_WINDOW = 60.0
RATE_LIMIT_MAX = 120
_ip_request_history = defaultdict(list)


@app.middleware("http")
async def rate_limit_and_correlation_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    request.state.correlation_id = correlation_id

    # Rate limiting (bypass on /health and /metrics)
    if request.url.path not in ["/health", "/metrics", "/docs", "/openapi.json"]:
        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        history = _ip_request_history[client_ip]
        history = [t for t in history if now - t < RATE_LIMIT_WINDOW]
        if len(history) >= RATE_LIMIT_MAX:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded. Maximum 120 requests per minute."},
                headers={"X-Correlation-ID": correlation_id},
            )
        history.append(now)
        _ip_request_history[client_ip] = history

    response: Response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if state.base_model is None:
        state.init_models()
    return HealthResponse(
        status="ok",
        model_loaded=state.base_model is not None,
        adapter_loaded=state.adapter_model is not None,
        device=str(state.device),
        version="0.2.0",
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    if state.base_model is None:
        state.init_models()
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        adapter_loaded=state.adapter_model is not None,
        trainable_parameters=state.param_stats.get("trainable_params", 0),
        total_parameters=state.param_stats.get("total_params", 0),
        trainable_percentage=state.param_stats.get("trainable_percentage", 0.0),
        device=str(state.device),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request):
    if state.base_engine is None:
        state.init_models()

    state.total_requests += 1
    engine = state.adapter_engine if (req.use_adapter and state.adapter_engine) else state.base_engine

    try:
        res: GenerationResult = engine.generate(
            instruction=req.instruction,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )

        state.successful_predictions += 1
        if res.is_schema_valid:
            state.schema_valid_predictions += 1
        state.total_latency_ms += res.latency_seconds * 1000.0

        return PredictResponse(
            instruction=res.instruction,
            generated_text=res.generated_text,
            parsed_json=res.parsed_json,
            is_schema_valid=res.is_schema_valid,
            validation_error=res.validation_error,
            latency_seconds=res.latency_seconds,
            tokens_generated=res.tokens_generated,
            adapter_used=res.adapter_enabled,
            correlation_id=getattr(request.state, "correlation_id", "N/A"),
        )
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest, request: Request):
    t0 = time.perf_counter()
    responses: List[PredictResponse] = []
    for inst in req.instructions:
        single_req = PredictRequest(
            instruction=inst,
            temperature=req.temperature,
            max_new_tokens=req.max_new_tokens,
            use_adapter=req.use_adapter,
        )
        single_res = await predict(single_req, request)
        responses.append(single_res)

    total_latency = time.perf_counter() - t0
    return BatchPredictResponse(
        predictions=responses,
        total_samples=len(responses),
        total_latency_seconds=round(total_latency, 4),
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    avg_latency = (
        (state.total_latency_ms / state.successful_predictions)
        if state.successful_predictions > 0
        else 0.0
    )
    validity_rate = (
        (state.schema_valid_predictions / state.successful_predictions)
        if state.successful_predictions > 0
        else 0.0
    )
    return MetricsResponse(
        total_requests=state.total_requests,
        successful_predictions=state.successful_predictions,
        schema_valid_predictions=state.schema_valid_predictions,
        schema_validity_rate=round(validity_rate, 4),
        average_latency_ms=round(avg_latency, 2),
        uptime_seconds=round(time.time() - state.start_time, 2),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
        headers={"X-Correlation-ID": getattr(request.state, "correlation_id", "N/A")},
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
