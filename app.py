# Cloud Brain Gateway
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import requests
import os
import hmac
import hashlib
import base64
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

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

    response = requests.post(
        VISION_URL,
        files={"file": image_bytes}
    )

    return response.json()

# -------------------------
# LLM Planner
# -------------------------

def call_llm(prompt):

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a robotics planner. Output JSON only."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )

    return response.json()["choices"][0]["message"]["content"]

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
