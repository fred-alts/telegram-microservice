from fastapi import FastAPI, Request, HTTPException
from pyrogram import Client
import os

app = FastAPI()

API_KEY = os.environ.get("TELEGRAM_SERVICE_API_KEY")
API_ID = int(os.environ.get("TELEGRAM_API_ID"))
API_HASH = os.environ.get("TELEGRAM_API_HASH")
SESSION_STRING = os.environ.get("TELEGRAM_SESSION_STRING")

def auth_check(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer ") or auth.split(" ")[1] != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/test-connection")
async def test_connection(request: Request):
    auth_check(request)
    try:
        app = Client(name="session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING)
        await app.connect()
        me = await app.get_me()
        await app.disconnect()
        return { "success": True, "username": me.username, "user_id": me.id }
    except Exception as e:
        return { "success": False, "error": str(e) }
