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
claude_client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

# DB Helpers
def get_db_filename(sender):
    clean_number = sender.replace("whatsapp:", "").replace("+", "")
    return f"wine_db_{clean_number}.json"

def load_db(sender):
    db_file = get_db_filename(sender)
    if os.path.exists(db_file):
        with open(db_file, "r") as f:
            return json.load(f)
    return {"inventory": {}, "history": [], "pending_wine": None}

def save_db(db, sender):
    db_file = get_db_filename(sender)
    with open(db_file, "w") as f:
        json.dump(db, f, indent=2)

# Handlers
async def handle_wine_photo(media_url, db, sender):
    try:
        img_response = requests.get(media_url, auth=(TWILIO_SID, TWILIO_TOKEN))
        img_base64 = base64.b64encode(img_response.content).decode("utf-8")
        
        message = claude_client.messages.create(
            model="claude-opus-4-6",
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
                        "text": """Extrae SOLO esto de la etiqueta:
- Bodega/Winery
- Región
- Varietal (uva)
- Vintage (año)

JSON: {"winery": "", "region": "", "varietal": "", "vintage": ""}"""
                    }
                ],
            }],
        )
        
        # Claude responde, intentamos parsear
        response_text = message.content[0].text.strip()
        
        # A veces Claude agrega texto antes/después del JSON
        try:
            # Intenta parsear directo
            wine_info = json.loads(response_text)
        except:
            # Si falla, busca el JSON dentro de la respuesta
            try:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = response_text[start:end]
                    wine_info = json.loads(json_str)
                else:
                    return "❌ No leo la etiqueta. ¿Foto más clara?"
            except:
                return "❌ No leo la etiqueta. ¿Foto más clara?"
        
        wine_key = f"{wine_info['winery']}_{wine_info['vintage']}"
        
        db["pending_wine"] = {
            "key": wine_key,
            "info": wine_info
        }
        
        if db["history"]:
            liked_wines = [w for w in db["history"] if w.get("rating", 0) >= 4]
            if liked_wines:
                history_str = json.dumps(liked_wines, indent=2)
                pred_message = claude_client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=150,
                    messages=[{
                        "role": "user",
                        "content": f"""¿Me va a gustar?

Vinos que amé:
{history_str}

Candidato: {wine_info['winery']} {wine_info['varietal']} {wine_info['vintage']}

Responde BREVE: probabilidad % y por qué."""
                    }],
                )
                prediction = pred_message.content[0].text
            else:
                prediction = "Sin historial aún para predecir"
        else:
            prediction = "Sin historial aún para predecir"
        
        response = f"""📸 Extraído:
{wine_info['winery']} {wine_info['varietal']} {wine_info['vintage']}
Región: {wine_info['region']}

🎯 {prediction}

¿Lo agregamos a inventario?
1️⃣ sí
2️⃣ no
"""
        return response
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

async def handle_yes_add(db):
    if not db.get("pending_wine"):
        return "❌ No hay vino pendiente. Envía una foto primero"
    
    wine_key = db["pending_wine"]["key"]
    wine_info = db["pending_wine"]["info"]
    
    if wine_key in db["inventory"]:
        current_qty = db["inventory"][wine_key].get("qty", 1)
        db["inventory"][wine_key]["qty"] = current_qty + 1
        response = f"✅ +1 botella. Total: {current_qty + 1}"
    else:
        db["inventory"][wine_key] = {
            "winery": wine_info["winery"],
            "region": wine_info["region"],
            "varietal": wine_info["varietal"],
            "vintage": wine_info["vintage"],
            "qty": 1,
            "date_added": datetime.now().isoformat()
        }
        response = f"✅ Agregado a inventario"
    
    db["pending_wine"] = None
    return response

async def handle_inventory_query(query, db):
    search_term = query[1:].strip().lower()
    if not search_term:
        return "Uso: ?bodega (ej: ?Rioja)"
    
    matches = []
    for wine_key, wine in db["inventory"].items():
        if search_term in wine["winery"].lower() or search_term in wine["region"].lower():
            matches.append(wine)
    
    if not matches:
        return f"❌ No tienes '{search_term}'"
    
    response = "🍷 Lo que tienes:\n"
    for wine in matches[:5]:
        response += f"• {wine['winery']} {wine['varietal']} ({wine['vintage']}) - {wine['qty']} bot.\n"
    return response

async def handle_recommendation(query, db):
    context = query.split(":", 1)[1].strip() if ":" in query else ""
    
    if not context:
        return "Uso: rec: comida (ej: rec: cordero)"
    if not db["inventory"]:
        return "📦 Sin vinos en inventario. ¡Envía fotos!"
    
    inventory_str = json.dumps(db["inventory"], indent=2)
    message = claude_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"""Sommelier. Inventario:

{inventory_str}

Quiero vino para: {context}

Recomienda los 2 mejores. Breve. Coloquial."""
        }],
    )
    return message.content[0].text

async def handle_rating(rating_text, db):
    try:
        rating = int(rating_text.split(":")[1].strip())
        if not 1 <= rating <= 5:
            return "Rating 1-5"
        
        wine_name = db.get("pending_wine", {}).get("info", {}).get("winery", "wine")
        
        db["history"].append({
            "wine": wine_name,
            "rating": rating,
            "date": datetime.now().isoformat()
        })
        
        return f"⭐ {rating}/5 - Anotado! Mejora tus recomendaciones"
    except:
        return "Uso: rating: 5 (1-5)"

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
        
        elif incoming_msg.lower() in ["sí", "si", "yes", "1"]:
            response_text = await handle_yes_add(db)
        
        elif incoming_msg.lower() in ["no", "2"]:
            db["pending_wine"] = None
            response_text = "❌ No agregado"
        
        elif incoming_msg.lower().startswith("?"):
            response_text = await handle_inventory_query(incoming_msg, db)
        
        elif incoming_msg.lower().startswith("rec:"):
            response_text = await handle_recommendation(incoming_msg, db)
        
        elif incoming_msg.lower().startswith("rating:"):
            response_text = await handle_rating(incoming_msg, db)
        
        elif incoming_msg.lower() == "inv":
            if not db["inventory"]:
                response_text = "📦 Sin vinos"
            else:
                response_text = "🍷 Tu inventario:\n"
                for wine_key, wine in db["inventory"].items():
                    response_text += f"• {wine['winery']} {wine['varietal']} ({wine['vintage']}) - {wine['qty']} bot.\n"
        
        elif incoming_msg.lower() in ["menu", "menú"]:
            response_text = """🍷 WINE BOT - MENÚ

📸 ENVÍA FOTO
Lee etiqueta + Predice si te gusta

🔍 ?bodega
¿Tengo en casa? (ej: ?Rioja)

🍽️ rec: comida
Recomendación (ej: rec: cordero)

⭐ rating: 5
Califica después de probar (1-5)

📦 inv
Ver todo tu inventario"""
        
        else:
            response_text = """¿Qué quieres?

📸 Envía foto de vino

Manda 'menú' para ver todas las opciones"""
    
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
