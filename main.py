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

class AnalyzeStrategyRequest(BaseModel):
    tips: list[dict]  # Expected structure from Supabase

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

# --- OpenAI Prompt ---
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
      "market": "Tipo de mercado (ex: Resultado, Total de Gols, etc)",
      "outcome": "Seleção feita na aposta",
      "individual_odd": float
    }
  ]
}
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
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned.removesuffix("```").strip()
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
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned.removesuffix("```").strip()
        return json.loads(cleaned)
    except Exception as e:
        print("OpenAI image analysis failed:", str(e))
        return { "is_tip": False, "error": str(e) }

# --- ROUTES ---
@app.post("/test-connection")
async def test_connection(request: Request):
    auth_check(request)
    return { "success": True }

@app.post("/get-channel-info")
def get_channel_info(request: Request, payload: dict = Body(...), authorization: str = Header(None)):
    log_request(request, payload)

    if not is_authorized(authorization):
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    chat_id = payload.get("chat_id")
    if not chat_id:
        return JSONResponse(status_code=400, content={"error": "Missing chat_id"})

    try:
        with Client("session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING, no_updates=True) as app:
            chat = app.get_chat(chat_id)

            photo_url = None
            if chat.photo:
                file_path = app.download_media(chat.photo, file_name=f"{chat.id}_profile.jpg")
                photo_url = upload_image_to_supabase(file_path, f"avatars/{chat.id}.jpg")

            info = {
                "chat_id": chat_id,
                "title": chat.title,
                "username": chat.username,
                "type": chat.type,
                "members": chat.members_count,
                "description": getattr(chat, "bio", None) or getattr(chat, "description", None),
                "photo_url": photo_url,
                "invite_link": chat.invite_link
            }

            print(f"[Info] 📡 Channel Info for {chat_id}: {info}")
            return {"success": True, "info": info}

    except Exception as e:
        print(f"[Info] 💥 Error getting channel info: {e}")
        return {"success": False, "error": str(e)}

@app.post("/test-channel-message")
async def test_channel_message(data: ChannelRequest, request: Request):
    auth_check(request)
    async with Client("session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING, workdir="/tmp", no_updates=True) as app_client:
        async for msg in app_client.get_chat_history(data.chat_id, limit=1):
            return {
                "success": True,
                "chat_id": data.chat_id,
                "message_id": msg.id,
                "text": msg.text or msg.caption,
                "media_type": msg.media,
                "date": msg.date.isoformat()
            }

@app.post("/collect-tips")
def collect_tips(request: Request, payload: dict = Body(...), authorization: str = Header(None)):
    log_request(request, payload)

    if not is_authorized(authorization):
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    channels = payload.get("channels")
    collected_tips = []

    if not channels:
        return JSONResponse(status_code=400, content={"error": "Missing channels"})

    with Client("session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING, no_updates=True) as app:
        for channel in channels:
            chat_id = channel.get("chat_id")
            since_str = channel.get("since")

            try:
                since = datetime.fromisoformat(since_str) if since_str else None
            except:
                print(f"[Collect] ⚠️ Invalid date for {chat_id}, skipping")
                continue

            print(f"[Collect] ▶️ {chat_id} since {since or 'beginning'}")

            try:
                for msg in app.get_chat_history(chat_id, reverse=True):
                    if since and msg.date < since:
                        break

                    tip_data = process_message(msg, chat_id)
                    if tip_data:
                        collected_tips.append(tip_data)

            except Exception as e:
                print(f"[Collect] ❌ Error with {chat_id}: {e}")

    print(f"[Collect] ✅ Total tips collected: {len(collected_tips)}")
    return {"success": True, "tips": collected_tips}

@app.post("/analyze-tipster-strategy")
async def analyze_tipster_strategy(request: Request, payload: AnalyzeStrategyRequest):
    auth_check(request)

    tips = payload.tips

    # Send tips to OpenAI for analysis
    try:
        strategy_prompt = """
Analisando as dicas de apostas de um tipster, forneça uma análise da sua estratégia e quais são as melhores tags que podem ser usadas para identificar seus padrões de aposta, como mercados preferidos, torneios mais focados, entre outros.
Analise as seguintes dicas de apostas:
"""
        result = client.chat.completions.create(
            model="gpt-4",
            messages=[
                { "role": "system", "content": strategy_prompt },
                { "role": "user", "content": json.dumps(tips) }
            ],
            temperature=0.7
        )

        content = result.choices[0].message.content.strip()

        return {"success": True, "strategy": content}

    except Exception as e:
        print(f"[Analyze Strategy] 💥 Error: {str(e)}")
        return {"success": False, "error": str(e)}

# --- Process Tip ---
def process_message(msg, chat_id):
    tip_data = None
    if msg.text:  # Analyzing text messages
        tip_data = analyze_message_with_openai_text(msg.text)

    if msg.media and isinstance(msg.media, MessageMediaType.PHOTO):  # Analyzing image messages
        tip_data = analyze_message_with_openai_image(msg.media.file_id)

    if tip_data.get("is_tip"):
        tip_data["chat_id"] = chat_id
        tip_data["message_id"] = msg.id
        tip_data["text"] = msg.text
        tip_data["date"] = msg.date.isoformat()  # Add the tip's timestamp
        return tip_data

    return None
