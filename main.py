from fastapi import FastAPI, Request
from twilio.rest import Client
from anthropic import Anthropic
import json
import os
from dotenv import load_dotenv
import base64
import requests
from datetime import datetime

load_dotenv()

app = FastAPI()

# ============ SETUP ============

# Twilio
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")
twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)

# Claude (Anthropic)
claude_client = Anthropic()

# Pinecone - Sintaxis para versión 2.2.4
pinecone_enabled = False
index = None

try:
    from pinecone import init
    init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west4-gcp")
    from pinecone import Index
    index = Index("wine-embeddings")
    pinecone_enabled = True
except Exception as e:
    print(f"Pinecone init error: {e}")
    pinecone_enabled = False

# ============ MULTI-USER DB HELPERS ============

def get_db_filename(sender):
    """Crea nombre único por usuario (número WhatsApp)."""
    clean_number = sender.replace("whatsapp:", "").replace("+", "")
    return f"wine_db_{clean_number}.json"

def load_db(sender):
    """Carga DB del usuario específico"""
    db_file = get_db_filename(sender)
    if os.path.exists(db_file):
        with open(db_file, "r") as f:
            return json.load(f)
    return {
        "inventory": {},
        "history": [],
        "user_preferences": {}
    }

def save_db(db, sender):
    """Guarda DB del usuario específico"""
    db_file = get_db_filename(sender)
    with open(db_file, "w") as f:
        json.dump(db, f, indent=2)

# ============ PINECONE HELPERS ============

def upsert_wine_to_pinecone(wine_id, wine_info):
    """Guarda wine embedding en Pinecone"""
    if not pinecone_enabled or index is None:
        return
    
    try:
        wine_desc = f"{wine_info['winery']} {wine_info['region']} {wine_info['varietal']} {wine_info['vintage']}"
        
        # Crear embedding simple basado en hash
        embedding = [hash(wine_desc + str(i)) % 256 / 256 for i in range(1536)]
        
        index.upsert(
            vectors=[
                (wine_id, embedding, wine_info)
            ]
        )
    except Exception as e:
        print(f"Pinecone upsert error: {e}")

def find_similar_wines(wine_id, top_k=3):
    """Busca vinos parecidos en Pinecone"""
    if not pinecone_enabled or index is None:
        return []
    
    try:
        results = index.query(
            id=wine_id,
            top_k=top_k,
            include_metadata=True
        )
        return results.get('matches', []) if results else []
    except Exception as e:
        print(f"Pinecone query error: {e}")
        return []

# ============ HANDLERS ============

async def handle_wine_photo(media_url, db, sender):
    """User envía foto de botella. Claude Vision extrae info."""
    try:
        # Descargar imagen
        img_response = requests.get(media_url)
        img_base64 = base64.b64encode(img_response.content).decode("utf-8")
        
        # Claude Vision extrae info
        message = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": """Extrae info de esta etiqueta de vino:
- Bodega/Winery
- Región
- Varietal (tipo de uva)
- Vintage (año)
- Cualquier nota que veas

Responde SOLO en JSON:
{"winery": "", "region": "", "varietal": "", "vintage": "", "notes": ""}

Si no ves algo, pon "unknown"."""
                        }
                    ],
                }
            ],
        )
        
        try:
            wine_info = json.loads(message.content[0].text)
        except:
            return "❌ No pude leer la etiqueta clarito. ¿Lo intentamos de nuevo?"
        
        wine_key = f"{wine_info['winery']}_{wine_info['vintage']}"
        
        if wine_key in db["inventory"]:
            current_qty = db["inventory"][wine_key].get("qty", 1)
            db["inventory"][wine_key]["qty"] = current_qty + 1
            response = f"✅ Ya tenías {wine_info['winery']} ({wine_info['vintage']})!\nAhora tienes {current_qty + 1} botellas."
        else:
            db["inventory"][wine_key] = {
                "winery": wine_info["winery"],
                "region": wine_info["region"],
                "varietal": wine_info["varietal"],
                "vintage": wine_info["vintage"],
                "notes": wine_info.get("notes", ""),
                "qty": 1,
