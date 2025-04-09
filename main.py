from fastapi import FastAPI, Request, HTTPException
from pyrogram import Client
from pydantic import BaseModel
import os

app = FastAPI()

# --- ENV config ---
API_KEY = os.environ.get("TELEGRAM_SERVICE_API_KEY")
API_ID = int(os.environ.get("TELEGRAM_API_ID"))
API_HASH = os.environ.get("TELEGRAM_API_HASH")
SESSION_STRING = os.environ.get("TELEGRAM_SESSION_STRING")

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
    limit: int = 10  # messages per chat

# --- ENDPOINTS ---

@app.post("/test-connection")
async def test_connection(request: Request):
    auth_check(request)
    try:
        app = Client(name="session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING)
        await app.connect()
        me = await app.get_me()
        await app.disconnect()
        return {
            "success": True,
            "username": me.username,
            "user_id": me.id
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/test-channel-message")
async def test_channel_message(request: Request, body: ChannelRequest):
    auth_check(request)
    try:
        app = Client(name="session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING)
        await app.connect()

        # Fetch latest message
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
                "date": msg.date.isoformat(),
            }
        else:
            return {
                "success": False,
                "error": "No messages found."
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/collect-tips")
async def collect_tips(request: Request, body: CollectTipsRequest):
    auth_check(request)
    try:
        app = Client(name="session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING)
        await app.connect()

        all_tips = []

        for chat_id in body.chat_ids:
            messages = [m async for m in app.get_chat_history(chat_id, limit=body.limit)]
            for msg in messages:
                all_tips.append({
                    "chat_id": chat_id,
                    "message_id": msg.id,
                    "text": msg.text or msg.caption,
                    "media_type": str(msg.media) if msg.media else None,
                    "date": msg.date.isoformat()
                })

        await app.disconnect()

        return {
            "success": True,
            "tips": all_tips
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
