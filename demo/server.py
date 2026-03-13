"""
demo/server.py

FastAPI inference server for MicroLM.

Endpoints:
    POST /generate         → full completion as JSON
    POST /stream           → token-by-token via Server-Sent Events (SSE)
    GET  /health           → health check

Start:
    uvicorn demo.server:app --host 0.0.0.0 --port 8000

Example:
    curl -X POST http://localhost:8000/generate \\
        -H "Content-Type: application/json" \\
        -d '{"prompt": "def fibonacci(n):", "max_new_tokens": 100}'
"""

from __future__ import annotations

import sys
import os
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import MicroLM, MicroLMConfig
from tokenizer.bpe import BPETokenizer
from inference.generate import generate


# ──────────────────────────────────────────────
# App configuration
# ──────────────────────────────────────────────

CHECKPOINT_PATH = os.environ.get("MICROLM_CHECKPOINT", "checkpoints/microlm-125m-final.pt")
TOKENIZER_PATH = os.environ.get("MICROLM_TOKENIZER", "tokenizer/tokenizer.json")

app = FastAPI(
    title="MicroLM API",
    description="FastAPI inference server for a custom 125M-parameter LLM built from scratch.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Model state (loaded once at startup)
# ──────────────────────────────────────────────

class ModelState:
    model: Optional[MicroLM] = None
    tokenizer: Optional[BPETokenizer] = None
    device: Optional[torch.device] = None


state = ModelState()


@app.on_event("startup")
async def load_model():
    """Load model and tokenizer into memory at server startup."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    state.device = device
    print(f"[MicroLM Server] Device: {device}")

    if not Path(TOKENIZER_PATH).exists():
        print(f"[WARNING] Tokenizer not found at {TOKENIZER_PATH}. /generate will fail.")
        return

    if not Path(CHECKPOINT_PATH).exists():
        print(f"[WARNING] Checkpoint not found at {CHECKPOINT_PATH}. /generate will fail.")
        return

    state.tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    print(f"[MicroLM Server] Tokenizer loaded: {state.tokenizer.vocab_size} tokens")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("model_config", MicroLMConfig())
    state.model = MicroLM(model_cfg).to(device)
    state.model.load_state_dict(ckpt["model_state_dict"])
    state.model.eval()
    print(f"[MicroLM Server] Model loaded: {state.model.param_count():,} parameters")


# ──────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to complete.")
    max_new_tokens: int = Field(200, ge=1, le=1024)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=1, le=200)
    top_p: float = Field(0.95, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    tokens_generated: int


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "device": str(state.device),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_completion(req: GenerateRequest):
    """Generate a text completion (blocking, full response)."""
    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    prompt_ids = state.tokenizer.encode(req.prompt)

    completion_ids = generate(
        model=state.model,
        prompt_ids=prompt_ids,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        device=state.device,
    )
    completion = state.tokenizer.decode(completion_ids)

    return GenerateResponse(
        prompt=req.prompt,
        completion=completion,
        tokens_generated=len(completion_ids),
    )


@app.post("/stream")
async def stream_completion(req: GenerateRequest):
    """Stream completion tokens via Server-Sent Events (SSE)."""
    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    async def token_generator() -> AsyncGenerator[str, None]:
        prompt_ids = state.tokenizer.encode(req.prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(state.device)

        # We stream one token at a time using the KV-cache in inference mode
        past_ids = prompt_ids[:]
        for _ in range(req.max_new_tokens):
            with torch.no_grad():
                with torch.amp.autocast(device_type=state.device.type, dtype=torch.bfloat16):
                    ctx = torch.tensor([past_ids[-1024:]], dtype=torch.long).to(state.device)
                    logits, _ = state.model(ctx)
                    next_logits = logits[0, -1, :]

            # Temperature + top-k sampling
            if req.temperature > 0:
                next_logits = next_logits / req.temperature
            if req.top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, req.top_k)
                next_logits[next_logits < top_k_vals[-1]] = float("-inf")
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            past_ids.append(next_id)
            token_text = state.tokenizer.decode([next_id])
            yield f"data: {token_text}\n\n"

            # Simple EOS check (token 0 or newline-heavy output)
            if next_id == 0:
                break
            await asyncio.sleep(0)  # yield control to event loop

        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")
