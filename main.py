from fastapi import FastAPI, Request, Header, Body, HTTPException
from fastapi.responses import JSONResponse
from pyrogram import Client
from pydantic import BaseModel
from pyrogram.enums import MessageMediaType
from PIL import Image
from openai import OpenAI
from supabase import create_client
from datetime import datetime
import os
import json
import requests
import base64

app = FastAPI()

# --- ENV config ---
API_KEY = os.environ.get("TELEGRAM_SERVICE_API_KEY")
API_ID = int(os.environ.get("TELEGRAM_API_ID"))
API_HASH = os.environ.get("TELEGRAM_API_HASH")
SESSION_STRING = os.environ.get("TELEGRAM_SESSION_STRING")
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "telegram-tips")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# --- External Clients ---
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Helpers ---
def log_request(request: Request, payload: dict):
    print(f"[Request] {request.method} {request.url}")
    print(f"[Payload] {payload}")

def auth_check(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer ") or auth.split(" ")[1] != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def upload_to_supabase(file_path: str, key: str) -> str:
    try:
        with open(file_path, "rb") as f:
            supabase.storage.from_(SUPABASE_BUCKET).upload(key, f)
        return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{key}"
    except Exception as e:
        print(f"[Supabase] ❌ Upload error: {e}")
        return None

def upload_image_to_supabase(file_path: str, telegram_message_id: int) -> str:
    if not file_path:
        return None
    try:
        converted_path = f"/tmp/{telegram_message_id}.jpeg"
        with Image.open(file_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(converted_path, "JPEG")
        file_name = f"{telegram_message_id}_{datetime.utcnow().isoformat()}.jpeg"
        return upload_to_supabase(converted_path, file_name)
    except Exception as e:
        print("Upload failed:", e)
        return None

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

def analyze_message_with_openai_text(text: str) -> dict:
    if not text:
        return {"is_tip": False}
    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": get_tip_prompt()},
                {"role": "user", "content": text}
            ],
            temperature=0.0
        )
        content = result.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.removeprefix("```json").strip()
        if content.endswith("```"):
            content = content.removesuffix("```").strip()
        return json.loads(content)
    except Exception as e:
        print("OpenAI text analysis failed:", str(e))
        return {"is_tip": False, "error": str(e)}

def analyze_message_with_openai_image(image_url: str) -> dict:
    try:
        response = requests.get(image_url)
        base64_img = base64.b64encode(response.content).decode("utf-8")
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": get_tip_prompt()},
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
        if content.startswith("```json"):
            content = content.removeprefix("```json").strip()
        if content.endswith("```"):
            content = content.removesuffix("```").strip()
        return json.loads(content)
    except Exception as e:
        print("OpenAI image analysis failed:", str(e))
        return {"is_tip": False, "error": str(e)}

# --- Models ---
class ChannelRequest(BaseModel):
    chat_id: str

# --- Endpoints ---
@app.post("/test-connection")
async def test_connection(request: Request):
    auth_check(request)
    return {"success": True}

@app.post("/get-channel-info")
def get_channel_info(request: Request, payload: dict = Body(...)):
    log_request(request, payload)
    auth_check(request)

    chat_id = payload.get("chat_id")
    if not chat_id:
        return JSONResponse(status_code=400, content={"error": "Missing chat_id"})

    try:
        with Client("session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING, no_updates=True) as app:
            chat = app.get_chat(chat_id)
            photo_url = None
            if chat.photo:
                file_path = app.download_media(chat.photo, file_name=f"{chat.id}_profile.jpg")
                photo_url = upload_to_supabase(file_path, f"avatars/{chat.id}.jpg")

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
