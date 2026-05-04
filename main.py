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

# Twilio
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")
twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)

# Claude
claude_client = Anthropic()

# Pinecone - Sintaxis 2.2.4 correcta
pinecone_enabled = False
index = None

try:
    from pinecone import init, Index
    init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west4-gcp")
    index = Index("wine-embeddings")
    pinecone_enabled = True
    print("Pinecone initialized successfully")
except Exception as e:
    print(f"Pinecone init failed: {e}")
    pinecone_enabled = False

# DB Helpers
def get_db_filename(sender):
    clean_number = sender.replace("whatsapp:", "").replace("+", "")
    return f"wine_db_{clean_number}.json"

def load_db(sender):
    db_file = get_db_filename(sender)
    if os.path.exists(db_file):
        with open(db_file, "r") as f:
            return json.load(f)
    return {"inventory": {}, "history": [], "user_preferences": {}}

def save_db(db, sender):
    db_file = get_db_filename(sender)
    with open(db_file, "w") as f:
        json.dump(db, f, indent=2)

# Pinecone helpers
def upsert_wine_to_pinecone(wine_id, wine_info):
    if not pinecone_enabled or index is None:
        return
    
    try:
        wine_desc = f"{wine_info['winery']} {wine_info['region']} {wine_info['varietal']} {wine_info['vintage']}"
        embedding = [hash(wine_desc + str(i)) % 256 / 256.0 for i in range(1536)]
        
        index.upsert(vectors=[(wine_id, embedding, wine_info)])
        print(f"Upserted {wine_id} to Pinecone")
    except Exception as e:
        print(f"Pinecone upsert error: {e}")

def find_similar_wines(wine_id, top_k=3):
    if not pinecone_enabled or index is None:
        return []
    
    try:
        results = index.query(id=wine_id, top_k=top_k, include_metadata=True)
        return results.get("matches", []) if results else []
    except Exception as e:
        print(f"Pinecone query error: {e}")
        return []

# Handlers
async def handle_wine_photo(media_url, db, sender):
    try:
        img_response = requests.get(media_url)
        img_base64 = base64.b64encode(img_response.content).decode("utf-8")
        
        message = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{
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
                        "text": """Extrae info de etiqueta de vino:
- Bodega/Winery
- Región
- Varietal
- Vintage (año)
- Notas

Responde SOLO en JSON:
{"winery": "", "region": "", "varietal": "", "vintage": "", "notes": ""}"""
                    }
                ],
            }],
        )
        
        try:
            wine_info = json.loads(message.content[0].text)
        except:
            return "❌ No pude leer la etiqueta. ¿Intentamos de nuevo?"
        
        wine_key = f"{wine_info['winery']}_{wine_info['vintage']}"
        
        if wine_key in db["inventory"]:
            current_qty = db["inventory"][wine_key].get("qty", 1)
            db["inventory"][wine_key]["qty"] = current_qty + 1
            return f"✅ Ya tenías {wine_info['winery']} ({wine_info['vintage']})!\nAhora tienes {current_qty + 1} botellas."
        else:
            db["inventory"][wine_key] = {
                "winery": wine_info["winery"],
                "region": wine_info["region"],
                "varietal": wine_info["varietal"],
                "vintage": wine_info["vintage"],
                "notes": wine_info.get("notes", ""),
                "qty": 1,
                "date_added": datetime.now().isoformat()
            }
            
            clean_number = sender.replace("whatsapp:", "").replace("+", "")
            wine_id = f"{clean_number}_{wine_key}".lower().replace(" ", "_")
            upsert_wine_to_pinecone(wine_id, db["inventory"][wine_key])
            
            return f"""📝 Agregué a tu inventario:
{wine_info['winery']} {wine_info['varietal']} {wine_info['vintage']}
Región: {wine_info['region']}"""
    except Exception as e:
        return f"❌ Error procesando foto: {str(e)}"

async def handle_inventory_query(query, db):
    search_term = query[1:].strip().lower()
    if not search_term:
        return "Uso: ?bodega (ej: ?Rioja)"
    
    matches = []
    for wine_key, wine in db["inventory"].items():
        if search_term in wine["winery"].lower() or search_term in wine["region"].lower():
            matches.append(wine)
    
    if not matches:
        return f"❌ No encuentro '{search_term}'"
    
    response = "🍷 **Lo que tienes:**\n"
    for wine in matches[:5]:
        response += f"• {wine['winery']} {wine['varietal']} ({wine['vintage']}) - {wine['qty']} bot.\n"
    return response

async def handle_recommendation(query, db):
    context = query.split(":", 1)[1].strip() if ":" in query else ""
    if not context:
        return "Uso: rec: comida (ej: rec: cordero)"
    if not db["inventory"]:
        return "📦 No tienes vinos aún. ¡Envía fotos!"
    
    inventory_str = json.dumps(db["inventory"], indent=2)
    message = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""Sommelier experto. Inventario:

{inventory_str}

Usuario quiere vino para: {context}

Recomienda 2-3 mejores. Explica por qué. Coloquial."""
        }],
    )
    return message.content[0].text

async def handle_prediction(wine_name, db):
    if not db["history"]:
        return "⭐ Aún sin calificaciones. Manda: rating: 5"
    
    liked_wines = [w for w in db["history"] if w.get("rating", 0) >= 4]
    if not liked_wines:
        return "⭐ Sin historial de vinos que te gusten."
    
    history_str = json.dumps(liked_wines, indent=2)
    message = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""¿Le va a gustar este vino?

Vinos que amó:
{history_str}

Candidato: {wine_name}

Analiza similitud y probabilidad."""
        }],
    )
    return message.content[0].text

async def handle_similar_wines(wine_name, db):
    if not db["inventory"]:
        return "📦 No tienes vinos aún."
    
    matching = [(k, v) for k, v in db["inventory"].items() if wine_name.lower() in f"{v['winery']} {v['vintage']}".lower()]
    if not matching:
        return f"❌ No encuentro '{wine_name}'"
    
    wine_key, wine = matching[0]
    
    if not pinecone_enabled:
        return "⚠️ Pinecone no conectado"
    
    similar = find_similar_wines(wine_key, top_k=3)
    if not similar:
        return f"🍷 Sin similares para {wine['winery']}"
    
    response = f"🍷 **Parecidos a {wine['winery']} {wine['vintage']}:**\n"
    for match in similar:
        if "metadata" in match:
            m = match["metadata"]
            response += f"• {m.get('winery', '?')} {m.get('varietal', '')} ({m.get('vintage', '')})\n"
    return response

async def handle_rating(rating_text, db):
    try:
        rating = int(rating_text.split(":")[1].strip())
        if not 1 <= rating <= 5:
            return "Rating debe ser 1-5"
        
        db["history"].append({
            "wine": "wine",
            "rating": rating,
            "date": datetime.now().isoformat()
        })
        
        return f"⭐ {rating}/5 - Anotado!"
    except:
        return "Uso: rating: 1 (o 2, 3, 4, 5)"

# Webhook
@app.post("/webhook")
async def webhook(request: Request):
    form_data = await request.form()
    incoming_msg = form_data.get("Body", "").strip()
    sender = form_data.get("From", "")
    num_media = int(form_data.get("NumMedia", 0))
    
    db = load_db(sender)
    response_text = ""
    
    try:
        if num_media > 0:
            media_url = form_data.get("MediaUrl0", "")
            response_text = await handle_wine_photo(media_url, db, sender)
        elif incoming_msg.lower().startswith("?"):
            response_text = await handle_inventory_query(incoming_msg, db)
        elif incoming_msg.lower().startswith("rec:"):
            response_text = await handle_recommendation(incoming_msg, db)
        elif incoming_msg.lower().startswith("pred:"):
            wine_name = incoming_msg.split(":", 1)[1].strip()
            response_text = await handle_prediction(wine_name, db)
        elif incoming_msg.lower().startswith("similar:"):
            wine_name = incoming_msg.split(":", 1)[1].strip()
            response_text = await handle_similar_wines(wine_name, db)
        elif incoming_msg.lower().startswith("rating:"):
            response_text = await handle_rating(incoming_msg, db)
        elif incoming_msg.lower() in ["help", "ayuda", "hola"]:
            response_text = """🍷 **WINE BOT:**

📸 Foto etiqueta → Extraigo + guardo

?Bodega → ¿Tengo en casa?

rec: comida → Recomendación

pred: vino → ¿Me va a gustar?

similar: vino → Parecidos

rating: 1-5 → Valora

inv → Inventario"""
        elif incoming_msg.lower() == "inv":
            if not db["inventory"]:
                response_text = "📦 No tienes vinos aún."
            else:
                response_text = "🍷 **Tu inventario:**\n"
                for wine_key, wine in db["inventory"].items():
                    response_text += f"• {wine['winery']} {wine['varietal']} ({wine['vintage']}) - {wine['qty']} bot.\n"
        else:
            response_text = "No entendí. Manda 'ayuda'"
    except Exception as e:
        response_text = f"❌ Error: {str(e)}"
        print(f"Error: {e}")
    
    save_db(db, sender)
    
    try:
        twilio_client.messages.create(from_=TWILIO_NUMBER, to=sender, body=response_text)
    except Exception as e:
        print(f"Twilio error: {e}")
    
    return {"status": "ok"}

@app.get("/")
def health():
    return {"status": "running ✅", "message": "Wine bot is alive!"}
