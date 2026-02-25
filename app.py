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
    image_b64 = base64.b64encode(image_bytes).decode()

    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
    api_key = os.getenv("RUNPOD_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Submit job
    run_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"

    response = requests.post(run_url, headers=headers, json={
        "input": {
            "image": image_b64
        }
    })

    job = response.json()
    job_id = job["id"]

    # Poll for result
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"

    for _ in range(20):  # poll up to 20 times
        status_response = requests.get(status_url, headers=headers)
        status_data = status_response.json()

        if status_data["status"] == "COMPLETED":
            return status_data["output"]

        if status_data["status"] == "FAILED":
            return {"error": "Vision job failed"}

        time.sleep(0.5)

    return {"error": "Vision job timeout"}

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
