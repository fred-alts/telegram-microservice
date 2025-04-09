from fastapi import FastAPI, Request, HTTPException
from pyrogram import Client
from pydantic import BaseModel
import os
import openai
import json

app = FastAPI()

# --- ENV config ---
API_KEY = os.environ.get("TELEGRAM_SERVICE_API_KEY")
API_ID = int(os.environ.get("TELEGRAM_API_ID"))
API_HASH = os.environ.get("TELEGRAM_API_HASH")
SESSION_STRING = os.environ.get("TELEGRAM_SESSION_STRING")
openai.api_key = os.environ.get("OPENAI_API_KEY")

# --- AUTH ---
def auth_check(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer ") or auth.split(" ")[1] != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# --- MODELS ---
class ChannelRequest(BaseModel):
    chat_id: str

class CollectTipsRequest(BaseModel):
    chat_ids: list[str]
    limit: int = 10

# --- OPENAI LOGIC ---
def analyze_message_with_openai(text: str) -> dict:
    if not text:
        return { "is_tip": False }

    system_prompt = """
You are a tip detection assistant. Analyze the message and determine if it contains a betting tip.

If it does, return JSON like:
{
  "is_tip": true,
  "match": "Barcelona vs Real Madrid",
  "teams": ["Barcelona", "Real Madrid"],
  "tournament": "La Liga",
  "datetime": "2025-04-11T20:00:00",
  "type": "single",
  "bets": [
    {
      "market": "Over 2.5 goals",
      "outcome": "Yes",
      "odd": 1.85,
      "value": 50,
      "expected_value": "High"
    }
  ]
}

If it is not a tip, return:
{ "is_tip": false }
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": text }
            ],
            temperature=0.3
        )
        result = response.choices[0].message.content.strip()
        return json.loads(result)
    except Exception as e:
        return { "is_tip": False, "error": str(e) }

# --- ENDPOINTS ---

@app.post("/test-connection")
async def test_connection(request: Request):
    auth_check(request)
    try:
        app = Client("session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING)
        await app.connect()
       
