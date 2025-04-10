from fastapi import FastAPI, Request, HTTPException
from pyrogram import Client
from pydantic import BaseModel
import os
import json
import requests
import base64
import re
from datetime import datetime
from supabase import create_client
import traceback
from openai import OpenAI
from PIL import Image
from pyrogram.enums import MessageMediaType

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
        print("[Upload] ❌ Received None as file_path, skipping upload")
        return None

    try:
        # Converte imagem para JPEG válido
        converted_path = f"/tmp/{telegram_message_id}.jpeg"
        with Image.open(file_path) as img:
            rgb_img = img.convert("RGB")  # força imagem RGB
            rgb_img.save(converted_path, "JPEG")

        file_name = f"{telegram_message_id}_{datetime.utcnow().isoformat()}.jpeg"

        with open(converted_path, "rb") as f:
            supabase.storage.from_(SUPABASE_BUCKET).upload(file_name, f, {"content-type": "image/jpeg"})

        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{file_name}"
        return public_url

    except Exception as e:
        print("[Upload] 💥 Upload failed:", e)
        return None

# --- OpenAI ANALYSIS ---
def analyze_message_with_openai_text(text: str) -> dict:
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
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": text }
            ],
            temperature=0.0
        )
        return json.loads(result.choices[0].message.content.strip())
    except Exception as e:
        return { "is_tip": False, "error": str(e) }

def analyze_message_with_openai_image(image_url: str) -> dict:
    try:
        print(f"Analyzing image: {image_url}")
        response = requests.get(image_url)
        base64_img = base64.b64encode(response.content).decode("utf-8")

        system_prompt = """
You are a professional betting tip extractor. Analyze the image and extract all betting tip information, even if it's partially visible.

Always return a strict JSON with the structure below. If you can't find some data, return null for that field.

{ 
  "is_tip": true or false, 
  ...
}
"""

        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                { "role": "system", "content": system_prompt },
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
        print(f"[GPT RAW] 🔍 Raw GPT content:\n{content}")

        # Remove blocos de markdown tipo ```json
        cleaned = content.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned.removesuffix("```").strip()

        try:
            parsed = json.loads(cleaned)
            return parsed
        except Exception as e:
            print(f"[JSON Parse] 💥 Failed to parse GPT response: {e}")
            return { "is_tip": False, "error": f"json.loads failed: {str(e)}", "raw": cleaned }

    except Exception as e:
        print("OpenAI image analysis failed:", str(e))
        return { "is_tip": False, "error": str(e) }

# --- ENDPOINTS ---
@app.post("/collect-tips")
async def collect_tips(request: Request, body: CollectTipsRequest):
    auth_check(request)
    app = Client("session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING)

    try:
        await app.connect()
        all_tips = []

        for chat_id in body.chat_ids:
            messages = [m async for m in app.get_chat_history(chat_id, limit=body.limit)]
            for msg in messages:
                parsed = None
                text = msg.text or msg.caption

                if msg.media == MessageMediaType.ANIMATION:
                    print(f"[Media] ⚠️ Skipping animation message {msg.id}")
                    continue

                if msg.media:
                    try:
                        print(f"[Media] 🔍 Message {msg.id} has media: {msg.media}")
                        print("[Media] ⌛ Attempting to download...")
                        file_path = await app.download_media(msg)

                        if not file_path:
                            print(f"[Media] ❌ Failed to download media from message {msg.id}")
                            continue

                        print(f"[Media] ✅ Downloaded media to: {file_path}")

                        image_url = upload_image_to_supabase(file_path, msg.id)

                        if not image_url:
                            print(f"[Media] ❌ Failed to upload media from message {msg.id}")
                            continue

                        print(f"[Media] 📤 Uploaded to Supabase: {image_url}")
                        print(f"[Media] 🧠 Sending to OpenAI Vision...")
                        parsed = analyze_message_with_openai_image(image_url)
                        print(f"[Media] 🧠 GPT-4 Vision response: {parsed}")

                    except Exception as media_error:
                        print(f"[Media] 💥 Exception during media processing: {str(media_error)}")
                        continue

                if parsed is None and text:
                    print(f"[Text] ✍️ Analyzing text message {msg.id}")
                    parsed = analyze_message_with_openai_text(text)

                if parsed and parsed.get("is_tip"):
                    print(f"[TIP] ✅ Valid tip detected in message {msg.id}")
                    all_tips.append({
                        "chat_id": chat_id,
                        "message_id": msg.id,
                        "text": text,
                        "parsed": parsed,
                        "date": msg.date.isoformat()
                    })

        return { "success": True, "tips": all_tips }

    except Exception as e:
        print("[collect-tips] 💥 Exception:", str(e))
        traceback.print_exc()
        return { "success": False, "error": str(e) }

    finally:
        await app.disconnect()

@app.post("/test-connection")
async def test_connection(request: Request):
    auth_check(request)
    try:
        app = Client("session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING)
        await app.connect()
        me = await app.get_me()
        await app.disconnect()
        return { "success": True, "username": me.username, "user_id": me.id }
    except Exception as e:
        return { "success": False, "error": str(e) }

@app.post("/test-channel-message")
async def test_channel_message(request: Request, body: ChannelRequest):
    auth_check(request)
    try:
        app = Client("session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING)
        await app.connect()
        messages = [m async for m in app.get_chat_history(body.chat_id, limit=1)]
        await app.disconnect()

        if messages:
            msg = messages[0]
            return {
                "success": True,
                "chat_id": body.chat_id,
                "message_id": msg.id,
                "text": msg.text or msg.caption,
                "media_type": str(msg.media) if msg.media else None,
                "date": msg.date.isoformat()
            }
        else:
            return { "success": False, "error": "No messages found." }
    except Exception as e:
        return { "success": False, "error": str(e) }
