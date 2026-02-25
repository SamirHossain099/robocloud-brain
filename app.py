# Cloud Brain Gateway (robocloud-brain)
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import requests
import json
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

# -------------------------
# Request / Response Schemas
# -------------------------

class PlanRequest(BaseModel):
    global_goal: str
    current_subgoal: str
    world_state: Dict[str, Any]
    progress_metrics: Dict[str, Any]
    device_profile: Dict[str, Any]

class Action(BaseModel):
    cmd: str
    params: Dict[str, Any]

class TerminationCondition(BaseModel):
    type: str
    params: Dict[str, Any]

class PlanResponse(BaseModel):
    subgoal: str
    actions: List[Action]
    termination_condition: TerminationCondition
    replan_after: str
    confidence: Optional[float] = None

class ASRInput(BaseModel):
    text: str

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

def call_llm(messages: list):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-oss-20b",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 800,
        "response_format": {"type": "json_object"}
    }

    response = requests.post(url, headers=headers, json=payload)

    data = response.json()

    if "choices" not in data:
        raise RuntimeError(f"Invalid Groq response: {data}")

    return data["choices"][0]["message"]["content"]


def build_messages(req: PlanRequest):

    system_prompt = """
You are a robotics planner.

Return ONLY valid JSON with this exact schema:

{
  "subgoal": string,
  "actions": [
    {
      "cmd": string,
      "params": object
    }
  ],
  "termination_condition": {
    "type": string,
    "params": object
  },
  "replan_after": "actions_complete" | "on_deviation",
  "confidence": number
}

No explanation.
No markdown.
No extra text.
JSON only.
"""

    user_content = json.dumps({
        "global_goal": req.global_goal,
        "current_subgoal": req.current_subgoal,
        "world_state": req.world_state,
        "progress_metrics": req.progress_metrics,
        "device_profile": req.device_profile
    })

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


@app.post("/plan", response_model=PlanResponse)
async def plan(req: PlanRequest):

    messages = build_messages(req)

    llm_output = call_llm(messages)

    try:
        structured = json.loads(llm_output)
    except Exception:
        raise RuntimeError(f"LLM did not return valid JSON: {llm_output}")

    return structured
