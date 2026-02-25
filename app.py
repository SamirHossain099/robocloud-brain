# Cloud Brain Gateway
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import hmac
import hashlib
import base64
import time
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VISION_URL = os.getenv("VISION_URL")
TOKEN_SECRET = os.getenv("TOKEN_SECRET", "supersecret")

# -------------------------
# Models
# -------------------------

class ASRInput(BaseModel):
    text: str

class TokenRequest(BaseModel):
    robot_id: str
    scene: dict
    intent: str

# -------------------------
# Vision Inference
# -------------------------

@app.post("/ingest/frame")
async def ingest_frame(file: UploadFile = File(...)):
    image_bytes = await file.read()

    runpod_url = "https://v1yuzqqacbzvbi.api.runpod.ai/infer"

    headers = {
        "Authorization": f"Bearer {os.getenv('RUNPOD_API_KEY')}"
    }

    files = {
        "file": ("frame.jpg", image_bytes, "image/jpeg")
    }

    response = requests.post(runpod_url, headers=headers, files=files)

    return response.json()

# -------------------------
# LLM Planner
# -------------------------

def call_llm(prompt):

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a robotics planner. Respond ONLY with valid JSON. Do not explain."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )

    print("Groq status:", response.status_code)
    print("Groq response:", response.text)

    result = response.json()

    if "choices" not in result:
        raise ValueError(f"Groq error: {result}")

    content = result["choices"][0]["message"]["content"]

    # Remove markdown code fences if present
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("json", "").strip()

    return content

# -------------------------
# Token Generator
# -------------------------

def sign_token(payload: str):
    signature = hmac.new(
        TOKEN_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode()

@app.post("/token/next")
def next_token(request: TokenRequest):

    prompt = f"""
    Scene: {request.scene}
    Intent: {request.intent}
    Generate next action in JSON format:
    {{
      "cmd": "move_arc",
      "v": float,
      "r": float,
      "duration_ms": int
    }}
    """

    llm_output = call_llm(prompt)

    signature = sign_token(llm_output)

    return {
        "token": llm_output,
        "signature": signature
    }
