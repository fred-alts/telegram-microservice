from fastapi import FastAPI, Request, Header, Body, HTTPException
from fastapi.responses import JSONResponse
from pyrogram import Client
from pydantic import BaseModel
from dateutil import parser
import os
import json
import requests
import base64
from datetime import datetime, timedelta, timezone
from supabase import create_client
from openai import OpenAI
from PIL import Image
from pyrogram.enums import MessageMediaType
import asyncio
import uuid
import time
from pyrogram.errors import FloodWait
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
    tips: list[dict]

# --- FloodWait-safe wrappers ---
async def safe_get_chat_history(app, chat_id, limit=100):
    try:
        messages = []
        async for msg in app.get_chat_history(chat_id, limit=limit):
            messages.append(msg)
        return messages
    except FloodWait as e:
        print(f"[FloodWait] ⏳ Esperando {e.value} segundos (get_chat_history)...")
        await asyncio.sleep(e.value)
        messages = []
        async for msg in app.get_chat_history(chat_id, limit=limit):
            messages.append(msg)
        return messages
        
# --- Supabase Upload ---
def upload_image_to_supabase(file_path: str, identifier: str) -> str:
    if not file_path:
        return None
    try:
        converted_path = f"/tmp/{uuid.uuid4().hex}.jpeg"
        with Image.open(file_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(converted_path, "JPEG")

        file_name = f"avatars/{identifier}_{datetime.utcnow().isoformat()}.jpeg"
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
  "tip_entries": [
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

def get_strategy_prompt():
    return """
A tua tarefa é analisar uma lista de apostas (tips) e identificar a estratégia do tipster.

Cada tip contém:
- `date`: data e hora em que foi publicada a tip
- `tip_entries[]`: apostas feitas, com data e hora do jogo (`datetime`)

Podes aferir que uma tip foi colocada em live quando a data da tip é dentro do horário do jogo

Devolve o seguinte JSON:
{
  "strategy_description": "Descreve a análise estratégica de como o tipster aposta, em relação a esportes, ligas e mercados, gestão de risco, gestão de banca, tendências de apostas ou outras informações relevantes",
  "tags": {
    "Mercados Preferidos": ["..."],
    "Ligas em Foco": ["..."],
    "Momento das Apostas": [
      "Live", 
      "Mesmo dia", 
      "1 dia antes", 
      "2+ dias antes"
    ],
    "Outras": ["qualquer outro padrão que observes"]
  }
}
Só devolve os momentos que se aplicam (não precisa todos). Usa sempre JSON válido.
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
        # Log inicial
        print(f"[Image Analysis] 🖼️ Downloading image from URL: {image_url}")
        response = requests.get(image_url)
        if not response.ok:
            print(f"[Image Analysis] ❌ Failed to fetch image ({response.status_code}) from {image_url}")
            return { "is_tip": False, "error": "Failed to download image" }
        image_bytes = response.content
        if not image_bytes:
            print(f"[Image Analysis] ❌ Image content is empty.")
            return { "is_tip": False, "error": "Empty image content" }
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{image_base64}"
        print(f"[Image Analysis] ✅ Image downloaded and encoded (size: {len(image_base64)} chars)")
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                { "role": "system", "content": get_tip_prompt() },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": { "url": data_url }
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
        print(f"[Image Analysis] ✅ OpenAI response received.")
        return json.loads(cleaned)
    except Exception as e:
        print(f"[Image Analysis] ❌ Exception: {str(e)}")
        return { "is_tip": False, "error": str(e) }

async def safe_download_media(app, media, file_name=None):
    try:
        return await app.download_media(media, file_name=file_name)
    except FloodWait as e:
        print(f"[FloodWait] ⏳ Esperando {e.value} segundos (download_media)...")
        await asyncio.sleep(e.value)
        return await app.download_media(media, file_name=file_name)
    except Exception as e:
        print(f"[safe_download_media] ❌ Erro inesperado no download: {e}")
        return None
        
# --- Processar mensagem com validações robustas ---
async def process_message(msg, chat_id, pyro):
    print(f"\n[Process] 📩 Message ID: {msg.id} | Date: {msg.date.isoformat()} | Has text: {bool(msg.text)} | Has photo: {bool(msg.photo)}")
    tip_data = None

    if msg.text:
        print(f"[Process] 🧠 Analisando texto da mensagem {msg.id}")
        tip_data = analyze_message_with_openai_text(msg.text)
        print(f"[Process] ✅ Resultado texto: {tip_data}")

    elif msg.photo:
        print(f"[Process] 🧠 A mensagem {msg.id} tem imagem. Verificando detalhes...")
        print(f"[DEBUG] msg.photo.file_id={getattr(msg.photo, 'file_id', '❌ sem file_id')}")

        try:
            # Recupera a mensagem completa por ID antes de baixar a media
            msg_full = await pyro.get_messages(chat_id, msg.id)

            if not msg_full.photo:
                print(f"[DEBUG] msg_full.photo está vazio mesmo após get_messages! ID: {msg.id}")
                return None
                
            file_path = await pyro.download_media(msg_full)

            if file_path is None:
                print(f"[Process] ❌ Falha no download da imagem da mensagem {msg.id} — file_path é None")
                print(f"[DEBUG] type(file_path): {type(file_path)}")
                print(f"[DEBUG] repr(file_path): {repr(file_path)}")
                print(f"[DEBUG] msg.photo: {msg.photo}")
                return None
            
            if not os.path.exists(file_path):
                print(f"[Process] ❌ Caminho {file_path} não existe após download.")
                return None
            
            print(f"[Process] ✅ Imagem da mensagem {msg.id} salva em {file_path}")
            image_url = upload_image_to_supabase(file_path, f"{chat_id}_{msg.id}")
            if not image_url:
                print(f"[Process] ❌ Upload falhou para imagem da mensagem {msg.id}")
                return None

            print(f"[Process] ✅ Imagem da mensagem {msg.id} disponível em {image_url}")
            tip_data = analyze_message_with_openai_image(image_url)
            print(f"[Process] ✅ Resultado imagem: {tip_data}")
        except Exception as e:
            print(f"[Process] ❌ Erro ao processar imagem da mensagem {msg.id}: {e}")
            return None

    else:
        print(f"[Process] ℹ️ Mensagem {msg.id} não tem texto nem imagem suportada")

    if tip_data and tip_data.get("is_tip"):
        tip_data["chat_id"] = chat_id
        tip_data["message_id"] = msg.id
        tip_data["date"] = msg.date.isoformat()
        print(f"[Process] ✅ Tip válida detectada na mensagem {msg.id}")
        return tip_data

    print(f"[Process] ⛔️ Mensagem {msg.id} não é uma tip")
    return None

def analyze_tipster_strategy_with_openai(tips: list[dict]) -> dict:
    try:
        messages = [
            { "role": "system", "content": get_strategy_prompt() },
            { "role": "user", "content": json.dumps(tips) }
        ]

        result = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3
        )

        content = result.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.removeprefix("```json").strip()
        if content.endswith("```"):
            content = content.removesuffix("```").strip()

        return json.loads(content)

    except Exception as e:
        print("[OpenAI] ❌ Strategy Analysis Failed:", e)
        return {
            "strategy_description": "Erro na análise",
            "tags": {
                "Mercados Preferidos": [],
                "Ligas em Foco": [],
                "Momento das Apostas": [],
                "Outras": [str(e)]
            }
        }
        
# --- Atualizado: coleta com limite e FloodWait safe ---
async def collect_tips_until_date(chat_id, until_date, batch_size=5, max_messages=10):
    collected_tips = []
    collected_messages = 0
    last_message_date = datetime.now(timezone.utc)
    last_message_id = None
    async with Client("session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING, no_updates=True) as pyro:
        while collected_messages < max_messages:
            print(f"[Collect] 🔄 Fetching {batch_size} messages from {chat_id}")
            try:
                messages = []
                if last_message_id:
                    history = pyro.get_chat_history(chat_id, limit=batch_size, offset_id=last_message_id)
                else:
                    history = pyro.get_chat_history(chat_id, limit=batch_size)
                async for msg in history:
                    messages.append(msg)
                if not messages:
                    break
                for msg in messages:
                    msg_date_utc = msg.date.replace(tzinfo=timezone.utc)
                    if msg_date_utc < until_date:
                        return collected_tips
                    last_message_id = msg.id
                    print(f"[Process] 📩 Message ID: {msg.id} | Date: {msg.date.isoformat()} | Has text: {bool(msg.text)} | Has photo: {bool(msg.photo)}")
                    tip_data = await process_message(msg, chat_id, pyro)
                    collected_messages += 1
                    if tip_data:
                        print(f"[Process] ✅ Tip detected in message {msg.id}")
                        collected_tips.append(tip_data)
                    else:
                        print(f"[Process] ⛔️ Message {msg.id} is not a tip")
                    if collected_messages >= max_messages:
                        break
            except FloodWait as e:
                print(f"[FloodWait] ⏳ Esperando {e.value} segundos...")
                await asyncio.sleep(e.value)
        print(f"[Collect] ✅ Collected {len(collected_tips)} tips from {collected_messages} messages")
    return collected_tips

@app.post("/test-connection")
async def test_connection(request: Request):
    auth_check(request)
    return { "success": True }

@app.post("/test-channel-message", summary="Testar se o canal pode ser acedido e devolver a última mensagem", tags=["Telegram"])
async def test_channel_message(request: Request, body: dict = Body(...), authorization: str = Header(None, description="Bearer token da API")):
    log_request(request, body)
    auth_check(request)
    chat_id = body.get("chat_id")
    if not chat_id:
        return JSONResponse(status_code=400, content={"error": "Missing chat_id"})
    try:
        async with Client("session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING, no_updates=True) as app:
            try:
                messages = [msg async for msg in app.get_chat_history(chat_id, limit=1)]
            except Exception as fetch_error:
                return {
                    "success": False,
                    "error": f"Não foi possível aceder ao canal: {str(fetch_error)}",
                    "reason": "Canal pode ser privado, não acessível ou com permissões limitadas"
                }
            if messages:
                msg = messages[0]
                last_message = {
                    "id": msg.id,
                    "date": msg.date.isoformat()
                }
                if msg.text:
                    last_message["type"] = "text"
                    last_message["content"] = msg.text
                elif msg.photo:
                    try:
                        file_path = await app.download_media(msg.photo)
                        photo_url = upload_image_to_supabase(file_path, f"lastmsg_{msg.id}")
                        last_message["type"] = "photo"
                        last_message["content"] = photo_url
                    except Exception as e:
                        last_message["type"] = "photo"
                        last_message["content"] = f"Erro ao obter imagem: {str(e)}"
                elif msg.video:
                    last_message["type"] = "video"
                    last_message["content"] = "Mensagem contém um vídeo"
                elif msg.sticker:
                    last_message["type"] = "sticker"
                    last_message["content"] = f"Sticker: {msg.sticker.emoji or 'sem emoji'}"
                else:
                    last_message["type"] = "none"
                    last_message["content"] = "Última mensagem sem texto e sem imagem."
                return {"success": True, "last_message": last_message}
            else:
                return {"success": True, "last_message": None, "info": "Canal acessível, mas sem mensagens"}
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "reason": "Erro inesperado ao testar o canal"
        }

@app.post("/get-channel-info")
async def get_channel_info(request: Request, payload: dict = Body(...), authorization: str = Header(None)):
    log_request(request, payload)
    if not is_authorized(authorization):
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    chat_id = payload.get("chat_id")
    if not chat_id:
        return JSONResponse(status_code=400, content={"error": "Missing chat_id"})
    try:
        async with Client("session", api_id=API_ID, api_hash=API_HASH, session_string=SESSION_STRING, no_updates=True) as app:
            chat = await app.get_chat(chat_id)
            photo_url = None
            if chat.photo:
                file_id = chat.photo.big_file_id
                file_path = await app.download_media(file_id, file_name=f"{chat.id}_profile.jpg")
                photo_url = upload_image_to_supabase(file_path, chat.id)
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
            return {"success": True, "info": info}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/collect-tips")
async def collect_tips(request: Request, payload: dict = Body(...), authorization: str = Header(None)):
    log_request(request, payload)
    try:
        if not is_authorized(authorization):
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        channels = payload.get("channels")
        collected_tips = []
        if not channels:
            print("[Collect] ❌ Nenhum canal recebido")
            return JSONResponse(status_code=400, content={"error": "Missing channels"})
        for channel in channels:
            chat_id = channel.get("chat_id")
            since_str = channel.get("since")
            try:
                since = parser.isoparse(since_str).astimezone(timezone.utc) if since_str else datetime.utcnow().replace(
                    hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
                )
            except Exception as e:
                print(f"[Collect] ⚠️ Erro ao interpretar 'since' para {chat_id}: {e}")
                continue
            try:
                tips = await collect_tips_until_date(chat_id, since)
                collected_tips.extend(tips)
            except Exception as e:
                print(f"[Collect] ❌ Erro ao coletar tips para {chat_id}: {e}")
        print(f"[Collect] ✅ Finalizando com {len(collected_tips)} tips.")
        print("[Collect] ✅ Enviando resposta...")
        return JSONResponse(content={"success": True, "tips": collected_tips})
    except Exception as e:
        print(f"[Collect] ❌ EXCEPTION inesperada em /collect-tips: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal error", "details": str(e)})
        
@app.post("/get-tipster-strategy")
async def get_tipster_strategy(request: Request, body: AnalyzeStrategyRequest, authorization: str = Header(None)):
    log_request(request, body.dict())
    if not is_authorized(authorization):
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    result = analyze_tipster_strategy_with_openai(body.tips)
    return { "success": True, "result": result }
