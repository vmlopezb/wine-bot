from fastapi import FastAPI, Request
from twilio.rest import Client
from anthropic import Anthropic
from pinecone import Pinecone
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
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# Pinecone (embeddings)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("wine-embeddings")

# ============ MULTI-USER DB HELPERS ============

def get_db_filename(sender):
    """
    Crea nombre único por usuario (número WhatsApp).
    sender es como: "whatsapp:+506XXXXXXXX"
    Extrae solo el número para crear archivo único.
    """
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

def get_embedding(text):
    """Pide a Claude que genere embedding (vector) de texto"""
    message = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": f"Generate a 1536-dimensional embedding for: {text}. Return ONLY a JSON array of 1536 floats, nothing else."
            }
        ]
    )
    try:
        embedding = json.loads(message.content[0].text)
        return embedding
    except:
        return [0.0] * 1536

def upsert_wine_to_pinecone(wine_id, wine_info):
    """Guarda wine embedding en Pinecone para búsquedas similares"""
    wine_desc = f"{wine_info['winery']} {wine_info['region']} {wine_info['varietal']} {wine_info['vintage']}"
    embedding = get_embedding(wine_desc)
    
    index.upsert(
        vectors=[
            {
                "id": wine_id,
                "values": embedding,
                "metadata": wine_info
            }
        ]
    )

def find_similar_wines(wine_id, top_k=3):
    """Busca vinos parecidos en Pinecone"""
    try:
        results = index.query(
            id=wine_id,
            top_k=top_k,
            include_metadata=True
        )
        return results['matches']
    except:
        return []

# ============ HANDLERS ============

async def handle_wine_photo(media_url, db, sender):
    """
    User envía foto de botella.
    Claude Vision extrae: bodega, región, varietal, vintage.
    Guarda en inventory del usuario.
    """
    try:
        # Descargar imagen
        img_response = requests.get(media_url)
        img_base64 = base64.b64encode(img_response.content).decode("utf-8")
        
        # Claude Vision extrae info de la foto
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
- Cualquier nota que veas (ej: "reserva", "gran reserva")

Responde SOLO en JSON con exactamente esto:
{"winery": "", "region": "", "varietal": "", "vintage": "", "notes": ""}

Si no ves algo, pon "unknown"."""
                        }
                    ],
                }
            ],
        )
        
        # Parse respuesta
        try:
            wine_info = json.loads(message.content[0].text)
        except:
            return "❌ No pude leer la etiqueta clarito. ¿Lo intentamos de nuevo?"
        
        # Crear wine ID único POR USUARIO
        clean_number = sender.replace("whatsapp:", "").replace("+", "")
        wine_id = f"{clean_number}_{wine_info['winery'].replace(' ', '_')}_{wine_info['vintage']}".lower()
        
        # Check si ya lo tienes
        wine_key = f"{wine_info['winery']}_{wine_info['vintage']}"
        if wine_key in db["inventory"]:
            current_qty = db["inventory"][wine_key].get("qty", 1)
            db["inventory"][wine_key]["qty"] = current_qty + 1
            response = f"✅ Ya tenías {wine_info['winery']} ({wine_info['vintage']})!\nAhora tienes {current_qty + 1} botellas."
        else:
            # Agregar a inventory
            db["inventory"][wine_key] = {
                "winery": wine_info["winery"],
                "region": wine_info["region"],
                "varietal": wine_info["varietal"],
                "vintage": wine_info["vintage"],
                "notes": wine_info.get("notes", ""),
                "qty": 1,
                "date_added": datetime.now().isoformat()
            }
            
            # Guarda embedding en Pinecone
            upsert_wine_to_pinecone(wine_id, db["inventory"][wine_key])
            
            response = f"""📝 Agregué a tu inventario:
{wine_info['winery']} {wine_info['varietal']} {wine_info['vintage']}
Región: {wine_info['region']}

Ya puedo recomendarte vinos parecidos! 🍷"""
        
        return response
    
    except Exception as e:
        return f"❌ Error procesando foto: {str(e)}"

async def handle_inventory_query(query, db):
    """
    User pregunta: "?Rioja" - ¿Tengo Rioja?
    """
    search_term = query[1:].strip().lower()
    
    if not search_term:
        return "Uso: ?bodega (ej: ?Rioja)"
    
    matches = []
    for wine_key, wine in db["inventory"].items():
        if search_term in wine["winery"].lower() or search_term in wine["region"].lower() or search_term in wine["varietal"].lower():
            matches.append(wine)
    
    if not matches:
        return f"❌ No encuentro '{search_term}' en tu inventario."
    
    response = "🍷 **Lo que tienes:**\n"
    for wine in matches[:5]:
        response += f"• {wine['winery']} {wine['varietal']} ({wine['vintage']}) - {wine['qty']} bot.\n"
    
    return response

async def handle_recommendation(query, db):
    """
    User pregunta: "rec: cordero" - Dame vino para X comida
    """
    context = query.split(":", 1)[1].strip() if ":" in query else ""
    
    if not context:
        return "Uso: rec: comida (ej: rec: cordero asado)"
    
    if not db["inventory"]:
        return "📦 No tienes vinos en inventario aún. ¡Envía fotos de botellas!"
    
    # Prepara inventory
    inventory_str = json.dumps(db["inventory"], indent=2)
    
    # Claude analiza y recomienda
    message = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": f"""Sos sommelier experto. Tu usuario tiene estos vinos:

{inventory_str}

Quiere un vino para comer: {context}

Recomienda los 2-3 MEJORES que tiene.
Explica breve por qué van bien.
Sé coloquial y amigable.
Si tiene perfecto, destácalo."""
            }
        ],
    )
    
    return message.content[0].text

async def handle_prediction(wine_name, db):
    """
    User pregunta: "pred: Tempranillo 2019" 
    ¿Me va a gustar este vino?
    """
    
    if not db["history"]:
        return "⭐ Aún no has calificado vinos. Cuando pruebes algo, manda: rating: 5 (o 1-5)"
    
    # Junta vinos que te gustaron
    liked_wines = [w for w in db["history"] if w.get("rating", 0) >= 4]
    
    if not liked_wines:
        return "⭐ Aún no tengo historial de vinos que te gusten. Valora alguno con: rating: 5"
    
    history_str = json.dumps(liked_wines, indent=2)
    
    # Claude predice
    message = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": f"""Analiza si este vino le va a gustar al usuario.

VINOS QUE AMASTE (rating 4-5):
{history_str}

VINO CANDIDATO: {wine_name}

Analiza:
1. ¿Tiene características similares?
2. Probabilidad de que le guste (0-100%)
3. ¿Lo compraría?

Responde como sommelier que lo conoce. Sé honesto."""
            }
        ],
    )
    
    return message.content[0].text

async def handle_similar_wines(wine_name, db):
    """
    User pregunta: "similar: Rioja 2018"
    Dame vinos parecidos a este
    """
    
    if not db["inventory"]:
        return "📦 No tienes vinos aún."
    
    # Busca en inventory
    matching_wines = []
    for wine_key, wine in db["inventory"].items():
        if wine_name.lower() in f"{wine['winery']} {wine['vintage']}".lower():
            matching_wines.append((wine_key, wine))
    
    if not matching_wines:
        return f"❌ No encuentro '{wine_name}' en tu inventario."
    
    # Toma el primero que match
    wine_key, wine = matching_wines[0]
    
    # Busca similares en Pinecone
    try:
        similar = find_similar_wines(wine_key, top_k=3)
        
        if not similar:
            return f"🍷 No encontré vinos similares para {wine['winery']}"
        
        response = f"🍷 **Parecidos a {wine['winery']} {wine['vintage']}:**\n"
        for match in similar:
            if match['metadata']:
                m = match['metadata']
                response += f"• {m['winery']} {m['varietal']} ({m['vintage']}) - Similitud: {match['score']:.0%}\n"
        
        return response
    except Exception as e:
        return f"Error buscando similares: {str(e)}"

async def handle_rating(rating_text, db, wine_name="last_wine"):
    """
    User manda: "rating: 5" para calificar último vino
    """
    try:
        rating = int(rating_text.split(":")[1].strip())
        
        if rating < 1 or rating > 5:
            return "Rating debe ser 1-5"
        
        # Agrega a history
        db["history"].append({
            "wine": wine_name,
            "rating": rating,
            "date": datetime.now().isoformat()
        })
        
        if rating >= 4:
            return f"⭐ {rating}/5 - ¡Anotado! Esto me ayuda a recomendarte mejor."
        else:
            return f"⭐ {rating}/5 - Anotado. Buscaré algo mejor para vos."
    
    except:
        return "Uso: rating: 1 (o 2, 3, 4, 5)"

# ============ WEBHOOK (El endpoint que Twilio llama) ============

@app.post("/webhook")
async def webhook(request: Request):
    """
    Twilio manda request cuando user envía mensaje.
    Cada usuario tiene su propio historial gracias a 'sender'.
    """
    form_data = await request.form()
    incoming_msg = form_data.get("Body", "").strip()
    sender = form_data.get("From", "")
    num_media = int(form_data.get("NumMedia", 0))
    
    # Carga DB específico del usuario
    db = load_db(sender)
    response_text = ""
    
    try:
        if num_media > 0:
            # 📸 User envió FOTO
            media_url = form_data.get("MediaUrl0", "")
            response_text = await handle_wine_photo(media_url, db, sender)
        
        elif incoming_msg.lower().startswith("?"):
            # ❓ "¿Tengo este vino?"
            response_text = await handle_inventory_query(incoming_msg, db)
        
        elif incoming_msg.lower().startswith("rec:"):
            # 🍽️ "Recomendación para comida"
            response_text = await handle_recommendation(incoming_msg, db)
        
        elif incoming_msg.lower().startswith("pred:"):
            # 🎯 "¿Me va a gustar?"
            wine_name = incoming_msg.split(":", 1)[1].strip()
            response_text = await handle_prediction(wine_name, db)
        
        elif incoming_msg.lower().startswith("similar:"):
            # 🍷 "Dame parecidos a..."
            wine_name = incoming_msg.split(":", 1)[1].strip()
            response_text = await handle_similar_wines(wine_name, db)
        
        elif incoming_msg.lower().startswith("rating:"):
            # ⭐ "Valora vino"
            response_text = await handle_rating(incoming_msg, db)
        
        elif incoming_msg.lower() in ["help", "ayuda", "hola"]:
            response_text = """🍷 **WINE BOT - Guía:**

📸 Envía foto de etiqueta → Extraigo + guardo

?Bodega → ¿Tengo en casa? (ej: ?Rioja)

rec: comida → Recomendación (ej: rec: cordero)

pred: vino → ¿Me va a gustar? (ej: pred: Tempranillo 2019)

similar: vino → Parecidos (ej: similar: Rioja 2018)

rating: 1-5 → Valora último vino

inv → Ver inventario completo"""
        
        elif incoming_msg.lower() == "inv":
            if not db["inventory"]:
                response_text = "📦 No tienes vinos aún. ¡Envía fotos!"
            else:
                response_text = "🍷 **Tu inventario:**\n"
                for wine_key, wine in db["inventory"].items():
                    response_text += f"• {wine['winery']} {wine['varietal']} ({wine['vintage']}) - {wine['qty']} bot.\n"
        
        else:
            response_text = "No entendí. Manda 'ayuda' para ver opciones."
    
    except Exception as e:
        response_text = f"❌ Error: {str(e)}"
        print(f"Error: {e}")
    
    # Guarda DB específico del usuario
    save_db(db, sender)
    
    # Envía respuesta por WhatsApp
    try:
        twilio_client.messages.create(
            from_=TWILIO_NUMBER,
            to=sender,
            body=response_text
        )
    except Exception as e:
        print(f"Error enviando Twilio: {e}")
    
    return {"status": "ok"}

# ============ HEALTH CHECK ============

@app.get("/")
def health():
    """Endpoint para verificar que el servidor está vivo"""
    return {"status": "running ✅", "message": "Wine bot is alive!"}
