from fastapi import FastAPI, Request, HTTPException
from pyrogram import Client
from pydantic import BaseModel
import os
import json
import requests
import base64
from datetime import datetime
from supabase import create_client
import traceback
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

@app.post("/test-channel-message")
async def test_channel_message(data: ChannelRequest, request: Request):
    auth_check(request)
    async with Client("session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING, workdir="/tmp", workers=0) as app_client:
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
async def collect_tips(data: CollectTipsRequest, request: Request):
    auth_check(request)
    tips = []

    async with Client("session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING, workdir="/tmp", workers=0) as app_client:
        for chat_id in data.chat_ids:
            async for msg in app_client.get_chat_history(chat_id, limit=data.limit):
                tip_data = {
                    "chat_id": chat_id,
                    "message_id": msg.id,
                    "text": msg.text or msg.caption,
                    "date": msg.date.isoformat(),
                    "parsed": None
                }
                try:
                    if msg.media == MessageMediaType.PHOTO:
                        print(f"[Media] 🔍 Message {msg.id} has media: {msg.media}")
                        path = await app_client.download_media(msg)
                        image_url = upload_image_to_supabase(path, msg.id)
                        if image_url:
                            tip_data["image_url"] = image_url
                            print("[Media] 🧠 Sending to OpenAI Vision...")
                            tip_data["parsed"] = analyze_message_with_openai_image(image_url)
                    else:
                        print(f"[Text] ✍️ Analyzing text message {msg.id}")
                        tip_data["parsed"] = analyze_message_with_openai_text(tip_data["text"])
                except Exception as e:
                    tip_data["parsed"] = { "is_tip": False, "error": str(e) }

                tips.append(tip_data)

    return { "success": True, "tips": tips }
