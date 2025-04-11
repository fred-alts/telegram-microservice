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
import traceback
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
