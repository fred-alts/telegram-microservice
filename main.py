from fastapi import FastAPI, Request, Header, Body, HTTPException
from fastapi.responses import JSONResponse
from pyrogram import Client
from pydantic import BaseModel
import os
import json
import requests
import base64
from datetime import datetime
from supabase import create_client
from openai import OpenAI
from PIL import Image
from pyrogram.enums import MessageMediaType
import asyncio
import traceback

app = FastAPI()

# --- ENV config ---
API_KEY = os.environ.get("TELEGRAM_SERVICE_API_KEY")
API_ID = int(os.environ.get("TELEGRAM_API_ID"))
API_HASH = os.environ.get("TELEGRAM_API_HASH")
SESSION_STRING = os.environ.get("TELEGRAM_SESSION_STRING")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "telegram-tips")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- LOG ---
def log_request(request: Request, payload: dict):
    print(f"[Request] {request.method} {request.url}")
    print(f"[Payload] {payload}")

# --- AUTH ---
def is_authorized(auth_header: str) -> bool:
    if not auth_header or not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split(" ")[1]
    return token == API_KEY

def auth_check(request: Request):
    auth_header = request.headers.get("Authorization")
    if not is_authorized(auth_header):
        raise HTTPException(status_code=401, detail="Unauthorized")

# --- MODELS ---
class ChannelRequest(BaseModel):
    chat_id: str

class CollectTipsRequest(BaseModel):
    chat_ids: list[str]
    limit: int = 10

class StrategyRequest(BaseModel):
    tips: list[dict]

# --- Supabase Upload ---
def upload_image_to_supabase(file_path: str, telegram_message_id: int) -> str:
    if not file_path:
        return None
    try:
        converted_path = f"/tmp/{telegram_message_id}.jpeg"
        with Image.open(file_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(converted_path, "JPEG")

        file_name = f"{telegram_message_id}_{datetime.utcnow().isoformat()}.jpeg"
        with open(converted_path, "rb") as f:
            supabase.storage.from_(SUPABASE_BUCKET).upload(file_name, f, {"content-type": "image/jpeg"})

        return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{file_name}"
    except Exception as e:
        print("Upload failed:", e)
        return None

# --- OpenAI Prompts ---
def get_tip_prompt():
    return """
Você é um extrator de dicas de apostas (tips). A partir do conteúdo fornecido (imagem ou texto), identifique se é uma tip válida.

Se não for uma tip, retorne:
```json
{ "is_tip": false }
```

Se for uma tip, retorne exatamente neste formato:
```json
{
  "is_tip": true,
  "type": "single" ou "multiple",
  "odd": float,
  "bets": [
    {
      "match": "Time A vs Time B",
      "tournament": "Nome do torneio (se visível)",
      "datetime": "Data e hora do jogo (formato ISO 8601, se visível)",
      "market": "Tipo de mercado",
      "outcome": "Seleção feita",
      "individual_odd": float
    }
  ]
}
"""

def get_strategy_prompt():
    return """
Você é um analista de apostas. Dada uma lista de tips, responda com a estratégia geral do tipster e uma lista de tags que descrevam seu estilo.
Responda nesse formato:
```json
{
  "strategy": "Descrição da estratégia do tipster",
  "tags": ["mercado preferido", "ligas em foco", "pré-jogo ou ao vivo"]
}
``` 
"""

# --- OpenAI Analysis ---
def analyze_message_with_openai_text(text: str) -> dict:
    if not text:
        return { "is_tip": False }
    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                { "role": "system", "content": get_tip_prompt() },
                { "role": "user", "content": text }
            ],
            temperature=0.0
        )
        content = result.choices[0].message.content.strip()
        cleaned = content.strip("`json ")
        return json.loads(cleaned)
    except Exception as e:
        print("OpenAI text analysis failed:", str(e))
        return { "is_tip": False, "error": str(e) }

def analyze_message_with_openai_image(image_url: str) -> dict:
    try:
        response = requests.get(image_url)
        base64_img = base64.b64encode(response.content).decode("utf-8")
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                { "role": "system", "content": get_tip_prompt() },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.0
        )
        content = result.choices[0].message.content.strip()
        cleaned = content.strip("`json ")
        return json.loads(cleaned)
    except Exception as e:
        print("OpenAI image analysis failed:", str(e))
        return { "is_tip": False, "error": str(e) }

def analyze_tipster_strategy(tips: list[dict]) -> dict:
    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": get_strategy_prompt()},
                {"role": "user", "content": json.dumps(tips)}
            ],
            temperature=0.0
        )
        content = result.choices[0].message.content.strip()
        cleaned = content.strip("`json ")
        return json.loads(cleaned)
    except Exception as e:
        print("OpenAI strategy analysis failed:", str(e))
        return {"strategy": "", "tags": [], "error": str(e)}

# --- ROUTES ---
@app.post("/test-connection")
async def test_connection(request: Request):
    auth_check(request)
    return { "success": True }

@app.post("/analyze-strategy")
def analyze_strategy(request: Request, payload: StrategyRequest):
    auth_check(request)
    print("[Strategy] 🎯 Analyzing tipster strategy...")
    response = analyze_tipster_strategy(payload.tips)
    print("[Strategy] ✅ Result:", response)
    return { "success": True, "analysis": response }

@app.post("/get-channel-info")
def get_channel_info(request: Request, payload: dict = Body(...), authorization: str = Header(None)):
    log_request(request, payload)
    if not is_authorized(authorization):
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    chat_id = payload.get("chat_id")
    if not chat_id:
        return JSONResponse(status_code=400, content={"error": "Missing chat_id"})

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with Client("session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING, no_updates=True) as app:
            chat = app.get_chat(chat_id)
            info = {
                "chat_id": chat_id,
                "title": chat.title,
                "username": chat.username,
                "type": chat.type,
                "members": chat.members_count,
                "description": getattr(chat, "bio", None) or getattr(chat, "description", None),
                "invite_link": chat.invite_link
            }
            return {"success": True, "info": info}

    except Exception as e:
        print("[Info] Error:", e)
        return {"success": False, "error": str(e)}

@app.post("/collect-tips")
def collect_tips(request: Request, payload: dict = Body(...), authorization: str = Header(None)):
    log_request(request, payload)
    if not is_authorized(authorization):
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    channels = payload.get("channels")
    collected_tips = []

    with Client("session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING, no_updates=True) as app:
        for channel in channels:
            chat_id = channel.get("chat_id")
            since_str = channel.get("since")
            try:
                since = datetime.fromisoformat(since_str) if since_str else None
            except:
                continue

            for msg in app.get_chat_history(chat_id, reverse=True):
                if since and msg.date < since:
                    break

                parsed = None
                if msg.media == MessageMediaType.PHOTO:
                    path = app.download_media(msg)
                    image_url = upload_image_to_supabase(path, msg.id)
                    parsed = analyze_message_with_openai_image(image_url)
                else:
                    parsed = analyze_message_with_openai_text(msg.text or msg.caption)

                if parsed.get("is_tip"):
                    collected_tips.append({
                        "chat_id": chat_id,
                        "message_id": msg.id,
                        "text": msg.text or msg.caption,
                        "date": msg.date.isoformat(),
                        "parsed": parsed,
                        "image_url": image_url if msg.media == MessageMediaType.PHOTO else None
                    })

    return {"success": True, "tips": collected_tips}
