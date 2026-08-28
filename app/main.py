from __future__ import annotations

import logging
import os
import time
import uuid

import torch
from fastapi import FastAPI, HTTPException, status
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    HealthResponse,
    ModelCard,
    ModelList,
    ResponseMessage,
    Usage,
)

# Set up logging for device selection visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serving-stack")

MODEL_ID = os.environ.get("MODEL_ID", r"C:\Models\Qwen2.5-1.5B")

app = FastAPI(title="serving-stack", version="wk4-gpu")

# Step 2: Dynamic Device Detection (CUDA if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f" Selected compute device: {DEVICE.upper()}")

print(f"Loading {MODEL_ID} on device: {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if DEVICE == "cuda":
    # 4-bit quantization config for VRAM constrained environments
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="cuda",
    )
else:
    # CPU fallback path using float32 / auto map
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

model.eval()
print(f"Model ready on {DEVICE.upper()}")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Included active device in health check output
    return HealthResponse(
        status="ok",
        model=MODEL_ID,
        device=DEVICE,
        cuda_available=torch.cuda.is_available(),
    )


@app.get("/v1/models", response_model=ModelList)
def list_models() -> ModelList:
    card = ModelCard(
        id=MODEL_ID,
        object="model",
        created=int(time.time()),
        owned_by="serving-stack",
    )
    return ModelList(object="list", data=[card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
    if req.model != MODEL_ID:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{req.model}' not found. Served model is '{MODEL_ID}'.",
        )

    messages_dict = [m.model_dump() for m in req.messages]

    input_ids = tokenizer.apply_chat_template(
        messages_dict,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    prompt_tokens = input_ids.shape[1]

    do_sample = req.temperature > 0.0
    generate_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": req.max_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = req.temperature

    with torch.no_grad():
        out = model.generate(**generate_kwargs)

    new_tokens = out[0][prompt_tokens:] 
    completion_tokens = len(new_tokens)
    total_tokens = prompt_tokens + completion_tokens

    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    finish_reason = "length" if completion_tokens >= req.max_tokens else "stop"

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )