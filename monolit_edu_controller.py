# =============================================================================
# IMPORTS
# =============================================================================
# Biblioteca estándar
import os
import time
import traceback
from io import BytesIO

# Third-party: ML y embeddings
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# Third-party: API y almacenamiento
import requests
from openai import OpenAI
from supabase import create_client, Client

# Third-party: UI
import gradio as gr

# Variables sensibles: definir en entorno (ej. .env) HF_TOKEN, SUPABASE_URL, SUPABASE_KEY
## eliminadas por seguridad

# =============================================================================
# CARGA DE MODELOS (SentenceTransformer + CLIP)
# =============================================================================
print("Cargando modelos...")
model_text = SentenceTransformer("all-MiniLM-L6-v2")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Modelos cargados correctamente (REST).")


# =============================================================================
# DATASET — Documento JSON único para inserción en BD
# =============================================================================
# Este "document" es el único punto de datos que el cuaderno usará para insertar la BD.
# He usado artistas y álbumes reales (títulos), pero NO incluye letras completas.
# Las "letras" son resúmenes.
document = {
    "metadata": {
        "fuente": "dataset_semi_real_unico",
        "descripcion": "Dataset semi-real con informacion de los integrantes."
    },


    # USUARIOS

    "usuarios": [
        {"nombre": "Juan", "correo": "juan@example.com", "plan_suscripcion": "premium", "tiempo_escucha": 1250.5, "id_portada": None},
        {"nombre": "Isabella", "correo": "isabella@example.com", "plan_suscripcion": "free", "tiempo_escucha": 320.0, "id_portada": None},
        {"nombre": "Alejandro", "correo": "alejandro@example.com", "plan_suscripcion": "family", "tiempo_escucha": 780.25, "id_portada": None}
    ],


    # GÉNEROS

    "generos": [
        {"nombre": "Pop", "descripcion": "Música popular y melódica."},
        {"nombre": "Rock", "descripcion": "Guitarras y estructuras de rock."},
        {"nombre": "Electronic", "descripcion": "Sonidos electrónicos y sintetizados."},
        {"nombre": "Alternative", "descripcion": "Sonidos alternativos y experimentales."},
        {"nombre": "R&B", "descripcion": "Ritmo y blues contemporáneo."},
        {"nombre": "Hip-Hop", "descripcion": "Rap y música urbana rítmica."}
    ],


    # ARTISTAS

    "artistas": [
        {"nombre": "Adele", "pais": "United Kingdom", "descripcion": "Cantautora británica."},
        {"nombre": "Coldplay", "pais": "United Kingdom", "descripcion": "Banda pop/rock melódico."},
        {"nombre": "Radiohead", "pais": "United Kingdom", "descripcion": "Rock alternativo experimental."},
        {"nombre": "Daft Punk", "pais": "France", "descripcion": "Dúo electrónico francés."},
        {"nombre": "Beyonce", "pais": "United States", "descripcion": "Artista R&B/pop global."},
        {"nombre": "Ed Sheeran", "pais": "United Kingdom", "descripcion": "Cantautor pop/folk."},
        {"nombre": "Taylor Swift", "pais": "United States", "descripcion": "Cantautora pop/folk."},
        {"nombre": "Kendrick Lamar", "pais": "United States", "descripcion": "Rapper influyente."},
        {"nombre": "The Weeknd", "pais": "Canada", "descripcion": "Artista R&B y pop alternativo."},
        {"nombre": "Sia", "pais": "Australia", "descripcion": "Cantautora australiana."}
    ],

    # RELACIÓN N:M ARTISTA–GÉNERO 

    "artista_genero": [
        {"artista": "Adele", "genero": "Pop"},
        {"artista": "Coldplay", "genero": "Alternative"},
        {"artista": "Coldplay", "genero": "Rock"},
        {"artista": "Radiohead", "genero": "Alternative"},
        {"artista": "Radiohead", "genero": "Rock"},
        {"artista": "Daft Punk", "genero": "Electronic"},
        {"artista": "Beyonce", "genero": "R&B"},
        {"artista": "Ed Sheeran", "genero": "Pop"},
        {"artista": "Taylor Swift", "genero": "Pop"},
        {"artista": "Kendrick Lamar", "genero": "Hip-Hop"},
        {"artista": "The Weeknd", "genero": "R&B"},
        {"artista": "Sia", "genero": "Pop"}
    ],

    # ÁLBUMES
    
    "albumes": [
        {"titulo": "21", "anio": 2011, "descripcion": "Álbum emblemático de Adele.", "artista": "Adele", "ruta_portada": None},
        {"titulo": "25", "anio": 2015, "descripcion": "Continuación poderosa.", "artista": "Adele", "ruta_portada": None},

        {"titulo": "Parachutes", "anio": 2000, "descripcion": "Debut de Coldplay.", "artista": "Coldplay", "ruta_portada": None},
        {"titulo": "A Rush of Blood to the Head", "anio": 2002, "descripcion": "Éxito mundial.", "artista": "Coldplay", "ruta_portada": None},

        {"titulo": "OK Computer", "anio": 1997, "descripcion": "Rock alternativo profundo.", "artista": "Radiohead", "ruta_portada": None},
        {"titulo": "Kid A", "anio": 2000, "descripcion": "Experimental.", "artista": "Radiohead", "ruta_portada": None},

        {"titulo": "Discovery", "anio": 2001, "descripcion": "Clásico electrónico.", "artista": "Daft Punk", "ruta_portada": None},
        {"titulo": "Random Access Memories", "anio": 2013, "descripcion": "Retro futurista.", "artista": "Daft Punk", "ruta_portada": None},

        {"titulo": "Lemonade", "anio": 2016, "descripcion": "Conceptual y poderosa.", "artista": "Beyonce", "ruta_portada": None},
        {"titulo": "4", "anio": 2011, "descripcion": "R&B moderno.", "artista": "Beyonce", "ruta_portada": None},

        {"titulo": "Multiply", "anio": 2014, "descripcion": "Hits globales.", "artista": "Ed Sheeran", "ruta_portada": None},
        {"titulo": "Divide", "anio": 2017, "descripcion": "Álbum comercial exitoso.", "artista": "Ed Sheeran", "ruta_portada": None},

        {"titulo": "1989", "anio": 2014, "descripcion": "Synth-pop moderno.", "artista": "Taylor Swift", "ruta_portada": None},
        {"titulo": "Folklore", "anio": 2020, "descripcion": "Indie folk íntimo.", "artista": "Taylor Swift", "ruta_portada": None},

        {"titulo": "good kid, m.A.A.d city", "anio": 2012, "descripcion": "Narrativa urbana.", "artista": "Kendrick Lamar", "ruta_portada": None},
        {"titulo": "DAMN.", "anio": 2017, "descripcion": "Hip-Hop moderno.", "artista": "Kendrick Lamar", "ruta_portada": None},

        {"titulo": "Beauty Behind the Madness", "anio": 2015, "descripcion": "Éxito global.", "artista": "The Weeknd", "ruta_portada": None},
        {"titulo": "After Hours", "anio": 2020, "descripcion": "Synthwave moderno.", "artista": "The Weeknd", "ruta_portada": None},

        {"titulo": "1000 Forms of Fear", "anio": 2014, "descripcion": "Poder vocal intenso.", "artista": "Sia", "ruta_portada": None},
        {"titulo": "This Is Acting", "anio": 2016, "descripcion": "Canciones para otros artistas.", "artista": "Sia", "ruta_portada": None}
    ],


    # CANCIONES

    "canciones": [
        {"titulo": "Rolling in the Deep", "artista": "Adele", "album": "21", "genero": "Pop", "duracion": 228, "letra": "Balada poderosa."},
        {"titulo": "Someone Like You", "artista": "Adele", "album": "21", "genero": "Pop", "duracion": 285, "letra": "Desamor nostálgico."},
        {"titulo": "Set Fire to the Rain", "artista": "Adele", "album": "21", "genero": "Pop", "duracion": 242, "letra": "Explosión emocional."},

        {"titulo": "Hello", "artista": "Adele", "album": "25", "genero": "Pop", "duracion": 295, "letra": "Reconciliación."},
        {"titulo": "When We Were Young", "artista": "Adele", "album": "25", "genero": "Pop", "duracion": 289, "letra": "Memoria y juventud."},
        {"titulo": "Water Under the Bridge", "artista": "Adele", "album": "25", "genero": "Pop", "duracion": 238, "letra": "Calma después de tormenta."},

        # Coldplay
        {"titulo": "Yellow", "artista": "Coldplay", "album": "Parachutes", "genero": "Alternative", "duracion": 269, "letra": "Devoción melódica."},
        {"titulo": "Shiver", "artista": "Coldplay", "album": "Parachutes", "genero": "Alternative", "duracion": 301, "letra": "Melancolía rock."},
        {"titulo": "Trouble", "artista": "Coldplay", "album": "Parachutes", "genero": "Alternative", "duracion": 250, "letra": "Reflexión."}
    ],


    # EMOCIONES

    "emociones": [
        {"nombre": "Feliz", "descripcion": "Alegría o bienestar."},
        {"nombre": "Triste", "descripcion": "Melancolía."},
        {"nombre": "Nostalgia", "descripcion": "Recuerdo emocional."},
        {"nombre": "Energia", "descripcion": "Motivación."},
        {"nombre": "Relajado", "descripcion": "Calma."},
        {"nombre": "Enfocado", "descripcion": "Concentración."}
    ],


    # EVENTOS

    "eventos": [
        {"usuario_correo": "juan@example.com", "titulo_cancion": "Someone Like You", "emocion": "Nostalgia", "tipo_relacion": "reproducida"},
        {"usuario_correo": "juan@example.com", "titulo_cancion": "Hello", "emocion": "Triste", "tipo_relacion": "reproducida"},

        {"usuario_correo": "isabella@example.com", "titulo_cancion": "Yellow", "emocion": "Feliz", "tipo_relacion": "favorita"},
        {"usuario_correo": "isabella@example.com", "titulo_cancion": "Trouble", "emocion": "Triste", "tipo_relacion": "buscada"},

        {"usuario_correo": "alejandro@example.com", "titulo_cancion": "Rolling in the Deep", "emocion": "Energia", "tipo_relacion": "reproducida"},
        {"usuario_correo": "alejandro@example.com", "titulo_cancion": "Shiver", "emocion": "Nostalgia", "tipo_relacion": "reproducida"}
    ],


    # PLAYLISTS

    "playlists": [
        {"titulo": "Juan Chill", "descripcion": "Relajación profunda", "usuario_correo": "juan@example.com",
         "canciones": ["Someone Like You", "Trouble", "Shiver"]},

        {"titulo": "Isabella Pop", "descripcion": "Hits suaves", "usuario_correo": "isabella@example.com",
         "canciones": ["Yellow", "Hello", "When We Were Young"]},

        {"titulo": "Alejandro Energy", "descripcion": "Potencia total", "usuario_correo": "alejandro@example.com",
         "canciones": ["Rolling in the Deep", "Yellow", "Set Fire to the Rain"]}
    ],


    # RESEÑAS

    "resenas": [
        {"usuario_correo": "juan@example.com", "tipo_objeto": "cancion",
         "titulo_objeto": "Someone Like You", "calificacion": 5, "comentario": "Muy emotiva."},

        {"usuario_correo": "isabella@example.com", "tipo_objeto": "album",
         "titulo_objeto": "Parachutes", "calificacion": 4, "comentario": "Suave y melódico."},

        {"usuario_correo": "alejandro@example.com", "tipo_objeto": "playlist",
         "titulo_objeto": "Alejandro Energy", "calificacion": 5, "comentario": "Perfecta para entrenar."}
    ],


    # CONSULTAS

    "consultas": [
        {"usuario_correo": "juan@example.com", "texto_pregunta": "Canciones para llorar con estilo"},
        {"usuario_correo": "isabella@example.com", "texto_pregunta": "Pop suave para estudiar"},
        {"usuario_correo": "alejandro@example.com", "texto_pregunta": "Canciones con energía para correr"}
    ]
}

# =============================================================================
# LLM — Cliente OpenAI compatible (HuggingFace Inference Router)
# =============================================================================
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Define la variable de entorno HF_TOKEN.")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

# -----------------------------------------------------------------------------
# Función generadora de texto (Llama 3.1 vía HuggingFace)
# -----------------------------------------------------------------------------
def generar_llm(prompt: str):
    """
    Genera texto usando Llama 3.1 8B Instruct vía HuggingFace Inference Router.
    Usa el SDK de OpenAI como cliente universal.
    """

    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content":
                 "Eres un asistente experto en análisis de datos, SQL y RAG. "
                 "Siempre usas el contexto entregado por el sistema."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.1,
            top_p=0.95
        )

        return completion.choices[0].message["content"]

    except Exception as e:
        return f"[ERROR LLM] {str(e)}"





# =============================================================================
# PORTADAS — iTunes, CLIP, Supabase Storage e inserción
# =============================================================================
# ----------------------------
# Config (thumbnails, bucket, credenciales Supabase)
# ----------------------------
THUMB_SIZE = (256, 256)
JPEG_QUALITY = 70
SUPABASE_BUCKET = "portadas"

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Define las variables de entorno SUPABASE_URL y SUPABASE_KEY.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# -----------------------------------------------------------------------------
# 1) Buscar portada en iTunes
# -----------------------------------------------------------------------------
def obtener_portada_itunes(artist_name: str, album_title: str = None):
    if not artist_name and not album_title:
        return None

    base = "https://itunes.apple.com/search"
    query = f"{artist_name or ''} {album_title or ''}".strip()

    try:
        r = requests.get(base, params={
            "term": query,
            "entity": "album",
            "limit": 10,
            "media": "music"
        }, timeout=10)

        r.raise_for_status()
        results = r.json().get("results", [])

        # coincidencia exacta si existe
        if album_title:
            for it in results:
                if it.get("collectionName", "").lower() == album_title.lower():
                    url = it.get("artworkUrl100")
                    if url:
                        return url.replace("100x100bb.jpg", "600x600bb.jpg")

        # fallback al primero
        if results:
            url = results[0].get("artworkUrl100")
            if url:
                return url.replace("100x100bb.jpg", "600x600bb.jpg")
        return None

    except Exception:
        return None


# -----------------------------------------------------------------------------
# 2) Descargar y procesar imagen (thumbnail JPEG)
# -----------------------------------------------------------------------------
def download_and_process_image(url: str):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()

        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.thumbnail(THUMB_SIZE, Image.LANCZOS)

        bio = BytesIO()
        img.save(bio, format="JPEG", quality=JPEG_QUALITY)
        bio.seek(0)

        return bio.read(), img

    except Exception:
        return None, None


# -----------------------------------------------------------------------------
# 3) Embedding de imagen (CLIP, normalizado)
# -----------------------------------------------------------------------------
def get_image_embedding_from_pil(pil_img: Image.Image):
    try:
        inputs = clip_processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)

        v = feats[0].cpu().numpy().astype(float)
        v = v / (np.linalg.norm(v) + 1e-12)   # normalización
        return v.tolist()

    except Exception:
        return None


# -----------------------------------------------------------------------------
# 4) Subir imagen a Supabase Storage
# -----------------------------------------------------------------------------
def upload_image_to_supabase(client, bucket: str, path: str, bytes_data: bytes):
    try:
        # borrar si ya existe
        try:
            client.storage.from_(bucket).remove([path])
        except:
            pass

        # subir archivo optimizado
        client.storage.from_(bucket).upload(
            path,
            bytes_data,
            file_options={"content-type": "image/jpeg"}
        )

        # url pública
        url = client.storage.from_(bucket).get_public_url(path)
        return url.get("publicURL") if isinstance(url, dict) else url

    except Exception:
        return None


# -----------------------------------------------------------------------------
# 5) Pipeline completo: iTunes → descarga → CLIP → upload → insert portada
# -----------------------------------------------------------------------------
def insertar_portada_real(client, artist_name=None, album_title=None, prefix="item"):
    """
    iTunes → download → compress → CLIP → upload → insert → return id_portada
    """

    # buscar imagen
    url = obtener_portada_itunes(artist_name, album_title)

    if not url:
        return None

    # descargar imagen
    img_bytes, pil_img = download_and_process_image(url)
    if not img_bytes or not pil_img:
        return None

    # embedding normalizado
    emb = get_image_embedding_from_pil(pil_img)

    # subir storage
    safe_prefix = prefix.lower().replace(" ", "_")
    path = f"{safe_prefix}.jpg"
    ruta = upload_image_to_supabase(client, SUPABASE_BUCKET, path, img_bytes)

    if not ruta:
        return None

    # insertar fila en tabla portada
    res = client.table("portada").insert({
        "ruta_archivo": f"{SUPABASE_BUCKET}/{path}",
        "descripcion": None,
        "emb_imagen": emb
    }).execute()

    return res.data[0]["id_portada"]




# =============================================================================
# EMBEDDINGS DE TEXTO Y UTILIDADES DE BD
# =============================================================================
# -----------------------------------------------------------------------------
# Embedding de texto (SentenceTransformer, normalizado)
# -----------------------------------------------------------------------------
def embed(text):
    if not text or not isinstance(text, str):
        return None

    v = model_text.encode(text)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.astype(float).tolist()


# -----------------------------------------------------------------------------
# Select eficiente (una fila por filtros, sin leer toda la tabla)
# -----------------------------------------------------------------------------
def _select_one(table, filters):
    q = supabase.table(table).select("*")
    for col, val in filters.items():
        q = q.eq(col, val)
    r = q.limit(1).execute()
    return r.data[0] if r.data else None


# =============================================================================
# INSERCIÓN DEL DATASET COMPLETO (REST, seguro, con portadas reales)
# =============================================================================
def insertar_dataset_rest_safe(doc):
    print("=== Iniciando inserción REST (SAFE + imágenes reales) ===")
    t0 = time.time()

    map_usuario = {}
    map_genero = {}
    map_artista = {}
    map_album = {}
    map_cancion = {}
    map_emocion = {}
    map_playlist = {}

    # UTIL: normalizar texto MUY largo (letras)
    def safe_lyric(txt: str):
        if txt and len(txt.split()) > 200:  # límite razonable
            # resumimos pero sin LLM (rápido)
            return " ".join(txt.split()[:200])
        return txt

    try:
        # ----------------------------------------------------------
        # 1) USUARIOS
        # ----------------------------------------------------------
        print("Insertando usuarios...")

        for usr in doc.get("usuarios", []):
            # placeholder portada
            ph = _select_one("portada", {"ruta_archivo": "/img/placeholder.png"})
            if ph:
                pid = ph["id_portada"]
            else:
                pid = supabase.table("portada").insert({
                    "ruta_archivo": "/img/placeholder.png",
                    "descripcion": None,
                    "emb_imagen": None
                }).execute().data[0]["id_portada"]

            usr["id_portada_real"] = pid

        # insertar usuario
        for usr in doc.get("usuarios", []):
            ex = _select_one("usuario", {"correo": usr["correo"]})

            if ex:
                uid = ex["id_usuario"]
            else:
                uid = supabase.table("usuario").insert({
                    "nombre": usr["nombre"],
                    "correo": usr["correo"],
                    "plan_suscripcion": usr.get("plan_suscripcion", "free"),
                    "tiempo_escucha": usr.get("tiempo_escucha", 0.0),
                    "id_portada": usr["id_portada_real"]
                }).execute().data[0]["id_usuario"]

            map_usuario[usr["correo"]] = uid


        # ----------------------------------------------------------
        # 2) GENEROS
        # ----------------------------------------------------------
        print("Insertando géneros...")

        for g in doc.get("generos", []):
            ex = _select_one("genero", {"nombre": g["nombre"]})
            gid = ex["id_genero"] if ex else supabase.table("genero").insert({
                "nombre": g["nombre"],
                "descripcion": g.get("descripcion")
            }).execute().data[0]["id_genero"]

            map_genero[g["nombre"]] = gid


        # ----------------------------------------------------------
        # 3) ARTISTAS
        # ----------------------------------------------------------
        print("Insertando artistas...")

        for a in doc.get("artistas", []):
            ex = _select_one("artista", {"nombre": a["nombre"]})
            aid = ex["id_artista"] if ex else supabase.table("artista").insert({
                "nombre": a["nombre"],
                "pais": a.get("pais"),
                "descripcion": a.get("descripcion"),
                "emb_descripcion": embed(a.get("descripcion"))
            }).execute().data[0]["id_artista"]

            map_artista[a["nombre"]] = aid


        # ----------------------------------------------------------
        # 4) ALBUMES (con portada real)
        # ----------------------------------------------------------
        print("Insertando álbumes...")

        for alb in doc.get("albumes", []):
            artist_id = map_artista[alb["artista"]]

            ex = _select_one("album", {"titulo": alb["titulo"], "id_artista": artist_id})
            if ex:
                alid = ex["id_album"]
            else:
                pid = insertar_portada_real(
                    supabase,
                    artist_name=alb["artista"],
                    album_title=alb["titulo"],
                    prefix=f"album_{alb['titulo']}"
                )

                if pid is None:  # fallback
                    pid = supabase.table("portada").insert({
                        "ruta_archivo": "/img/placeholder.png",
                        "descripcion": None,
                        "emb_imagen": None
                    }).execute().data[0]["id_portada"]

                alid = supabase.table("album").insert({
                    "titulo": alb["titulo"],
                    "anio": alb.get("anio"),
                    "descripcion": alb.get("descripcion"),
                    "id_artista": artist_id,
                    "id_portada": pid,
                    "emb_descripcion": embed(alb.get("descripcion"))
                }).execute().data[0]["id_album"]

            map_album[alb["titulo"]] = alid


        # ----------------------------------------------------------
        # 5) CANCIONES
        # ----------------------------------------------------------
        print("Insertando canciones...")

        for c in doc.get("canciones", []):
            artist_id = map_artista[c["artista"]]

            ex = _select_one("cancion", {"titulo": c["titulo"], "id_artista": artist_id})
            if ex:
                cid = ex["id_cancion"]
            else:
                letra_proc = safe_lyric(c.get("letra"))
                cid = supabase.table("cancion").insert({
                    "titulo": c["titulo"],
                    "id_artista": artist_id,
                    "id_album": map_album.get(c["album"]),
                    "id_genero": map_genero.get(c["genero"]),
                    "duracion": c.get("duracion"),
                    "letra": letra_proc,
                    "emb_letra": embed(letra_proc)
                }).execute().data[0]["id_cancion"]

            map_cancion[c["titulo"]] = cid


        # ----------------------------------------------------------
        # 6) EMOCIONES
        # ----------------------------------------------------------
        print("Insertando emociones...")

        for e in doc.get("emociones", []):
            ex = _select_one("emocion", {"nombre": e["nombre"]})
            eid = ex["id_emocion"] if ex else supabase.table("emocion").insert(e).execute().data[0]["id_emocion"]
            map_emocion[e["nombre"]] = eid


        # ----------------------------------------------------------
        # 7) EVENTOS
        # ----------------------------------------------------------
        print("Insertando eventos...")

        for ev in doc.get("eventos", []):
            supabase.table("usuario_cancion_emocion").insert({
                "id_usuario": map_usuario[ev["usuario_correo"]],
                "id_cancion": map_cancion[ev["titulo_cancion"]],
                "id_emocion": map_emocion[ev["emocion"]],
                "tipo_relacion": ev.get("tipo_relacion")
            }).execute()


        # ----------------------------------------------------------
        # 8) PLAYLISTS
        # ----------------------------------------------------------
        print("Insertando playlists...")

        for p in doc.get("playlists", []):
            uid = map_usuario[p["usuario_correo"]]
            ex = _select_one("playlist", {"titulo": p["titulo"], "id_usuario": uid})

            if ex:
                pid = ex["id_playlist"]
            else:
                pid_cover = insertar_portada_real(
                    supabase,
                    artist_name=None,
                    album_title=p["titulo"],
                    prefix=f"playlist_{p['titulo']}"
                )

                if pid_cover is None:
                    pid_cover = supabase.table("portada").insert({
                        "ruta_archivo": "/img/placeholder.png",
                        "descripcion": None,
                        "emb_imagen": None
                    }).execute().data[0]["id_portada"]

                pid = supabase.table("playlist").insert({
                    "titulo": p["titulo"],
                    "descripcion": p.get("descripcion"),
                    "id_usuario": uid,
                    "id_portada": pid_cover
                }).execute().data[0]["id_playlist"]

            map_playlist[p["titulo"]] = pid

            # relaciones playlist-cancion
            for cname in p.get("canciones", []):
                cid = map_cancion[cname]
                if not _select_one("playlist_cancion", {"id_playlist": pid, "id_cancion": cid}):
                    supabase.table("playlist_cancion").insert({
                        "id_playlist": pid,
                        "id_cancion": cid
                    }).execute()


        # ----------------------------------------------------------
        # 9) RESEÑAS
        # ----------------------------------------------------------
        print("Insertando reseñas...")

        for r in doc.get("resenas", []):
            tipo = r["tipo_objeto"]
            titulo = r["titulo_objeto"]

            if tipo == "cancion":
                id_obj = map_cancion[titulo]
            elif tipo == "album":
                id_obj = map_album[titulo]
            else:
                id_obj = map_playlist[titulo]

            comentario = r.get("comentario")
            emb_com = embed(comentario)

            supabase.table("resena").insert({
                "id_usuario": map_usuario[r["usuario_correo"]],
                "tipo_objeto": tipo,
                "id_objeto": id_obj,
                "calificacion": r.get("calificacion"),
                "comentario": comentario,
                "emb_comentario": emb_com
            }).execute()


        # ----------------------------------------------------------
        # 10) CONSULTAS + EMBEDDINGS
        # ----------------------------------------------------------
        print("Insertando consultas...")

        for q in doc.get("consultas", []):
            # consulta
            ins = supabase.table("consulta").insert({
                "id_usuario": map_usuario[q["usuario_correo"]],
                "texto_pregunta": q["texto_pregunta"]
            }).execute()

            idc = ins.data[0]["id_consulta"]

            # embedding normalizado
            qemb = embed(q["texto_pregunta"])

            supabase.table("query_embedding").insert({
                "id_consulta": idc,
                "vector_embedding": qemb
            }).execute()


        print("Inserción finalizada en", round(time.time() - t0, 2), "s")

    except Exception as e:
        print("ERROR GENERAL:")
        traceback.print_exc()



# Ejecutar la inserción del dataset
insertar_dataset_rest_safe(document)


# =============================================================================
# CENTRO RPC — Búsquedas vectoriales y funciones analíticas (Supabase)
# =============================================================================
# ------------------------------------------------------------
# 1) Búsquedas vectoriales (RPC)
# ------------------------------------------------------------

def rpc_match_canciones_por_letra(texto, top_k=5):
    emb = embed(texto)
    return supabase.rpc(
        "match_canciones_por_letra",
        {"query_embedding": emb, "top_k": top_k}
    ).execute().data


def rpc_match_album_por_texto(texto, top_k=5):
    emb = embed(texto)
    return supabase.rpc(
        "match_album_por_texto",
        {"query_embedding": emb, "top_k": top_k}
    ).execute().data


def rpc_match_portadas_por_imagen(pil_img, top_k=5):
    emb = get_image_embedding_from_pil(pil_img)
    return supabase.rpc(
        "match_portadas_por_imagen",
        {"query_embedding": emb, "top_k": top_k}
    ).execute().data

def rpc_match_canciones_por_letra_arr(texto, top_k=5):
    emb = embed(texto)
    return supabase.rpc("match_canciones_por_letra_arr", {"query_embedding_arr": emb, "top_k": top_k}).execute().data

def rpc_match_album_por_texto_arr(texto, top_k=5):
    emb = embed(texto)
    return supabase.rpc("match_album_por_texto_arr", {"query_embedding_arr": emb, "top_k": top_k}).execute().data

def rpc_match_portadas_por_imagen_arr(pil_img, top_k=5):
    emb = get_image_embedding_from_pil(pil_img)
    return supabase.rpc("match_portadas_por_imagen_arr", {"query_embedding_arr": emb, "top_k": top_k}).execute().data

def rpc_match_artista_por_texto_v2(texto, top_k=5):
    emb = embed(texto)
    return supabase.rpc("match_artista_por_texto_v2", {"query_embedding_arr": emb, "top_k": top_k}).execute().data

def rpc_resenas_por_cancion_v2(id_cancion):
    return supabase.rpc("resenas_por_cancion_v2", {"p_id_cancion": id_cancion}).execute().data

def rpc_usuarios_cualificados_upgrade_v2(min_tiempo=2000):
    return supabase.rpc("usuarios_cualificados_upgrade_v2", {"p_min_tiempo": min_tiempo}).execute().data


# ------------------------------------------------------------
# 2) Funciones analíticas (RPC)
# ------------------------------------------------------------

def rpc_plan_mas_usado():
    return supabase.rpc("plan_mas_usado").execute().data


def rpc_estadisticas_generales():
    return supabase.rpc("estadisticas_generales").execute().data


def rpc_get_consultas_usuario(id_usuario, p_limit=50):
    return supabase.rpc(
        "get_consultas_usuario",
        {"p_id_usuario": id_usuario, "p_limit": p_limit}
    ).execute().data


def rpc_emocion_dominante_usuario(id_usuario, since_days=365):
    return supabase.rpc(
        "obtener_emocion_dominante_usuario",
        {"p_id_usuario": id_usuario, "p_since_days": since_days}
    ).execute().data


def rpc_top_playlists_usuario(id_usuario, limit=10):
    return supabase.rpc(
        "top_playlists_usuario",
        {"p_id_usuario": id_usuario, "p_limit": limit}
    ).execute().data


def rpc_top_artistas_por_usuario(id_usuario, limit=10):
    return supabase.rpc(
        "top_artistas_por_usuario",
        {"p_id_usuario": id_usuario, "p_limit": limit}
    ).execute().data


# ------------------------------------------------------------
# 3) Funciones atómicas (modifican datos de forma segura)
# ------------------------------------------------------------

def rpc_incrementar_tiempo_escucha(id_usuario, minutos):
    return supabase.rpc(
        "incrementar_tiempo_escucha_atomic",
        {"p_id_usuario": id_usuario,
         "p_minutos": minutos}
    ).execute().data


def rpc_crear_resena_atomic(id_usuario, tipo_objeto, id_objeto,
                            calificacion, comentario):
    emb = embed(comentario)
    return supabase.rpc(
        "crear_resena_atomic",
        {"p_id_usuario": id_usuario,
         "p_tipo_objeto": tipo_objeto,
         "p_id_objeto": id_objeto,
         "p_calificacion": calificacion,
         "p_comentario": comentario,
         "p_emb_comentario": emb}
    ).execute().data


def rpc_crear_consulta_con_embedding_atomic(id_usuario, texto):
    emb = embed(texto)
    return supabase.rpc(
        "crear_consulta_con_embedding_atomic",
        {"p_id_usuario": id_usuario,
         "p_texto_pregunta": texto,
         "p_vector": emb}
    ).execute().data


# ------------------------------------------------------------
# 4) Ejecutor RPC genérico (para Gradio y llamadas dinámicas)
# ------------------------------------------------------------

def ejecutar_rpc(nombre, parametros=None):
    """Llamada genérica a cualquier RPC."""
    if parametros:
        return supabase.rpc(nombre, parametros).execute().data
    else:
        return supabase.rpc(nombre).execute().data


print("Centro RPC + RAG inicializado correctamente.")



# Tests rápidos RPC (opcional)
res = rpc_match_canciones_por_letra_arr("amor y desamor en la calle", top_k=3)
print(res)
res2 = rpc_match_album_por_texto_arr("álbum acústico", top_k=3)
print(res2)
res3 = rpc_match_artista_por_texto_v2("cantante indie", top_k=3)
print(res3)
r_res = rpc_resenas_por_cancion_v2(1)
print(r_res)
r_users = rpc_usuarios_cualificados_upgrade_v2(1000)
print(r_users)


# =============================================================================
# RAG GLOBAL — Canciones + Álbumes + Artistas + LLM
# =============================================================================
def rag_global(pregunta, top_k=5):
    """
    RAG Global mejorado:
    - Embedding → Supabase (canciones, álbumes, artistas)
    - Combina evidencias
    - Construye contexto exhaustivo
    - Llama a LLM con prompt robusto
    """

    try:
        # ------------------------------------------------------------
        # 1. EMBEDDING
        # ------------------------------------------------------------
        embedding = embed(pregunta)

        # ------------------------------------------------------------
        # 2. RECUPERACIÓN VECTORIAL (RPC)
        # ------------------------------------------------------------
        ev_canciones = rpc_match_canciones_por_letra(pregunta, top_k)
        ev_albumes   = rpc_match_album_por_texto(pregunta, top_k)
        ev_artistas  = rpc_match_artista_por_texto_v2(pregunta, top_k)

        # Unificar resultados con etiquetas
        evidencias = []

        for e in ev_canciones:
            e["tipo"] = "cancion"
            evidencias.append(e)

        for e in ev_albumes:
            e["tipo"] = "album"
            evidencias.append(e)

        for e in ev_artistas:
            e["tipo"] = "artista"
            evidencias.append(e)

        if not evidencias:
            return "No encontré evidencia relevante en la base de datos.", []

        # ------------------------------------------------------------
        # 3. CONSTRUIR CONTEXTO REAL A PARTIR DE SQL
        # ------------------------------------------------------------
        bloques = []

        for ev in evidencias:
            tipo = ev["tipo"]

            if tipo == "cancion":
                fila = (
                    supabase.table("cancion")
                    .select("*")
                    .eq("id_cancion", ev["id_cancion"])
                    .execute()
                    .data[0]
                )
                bloques.append(
                    f"[CANCIÓN]\nTítulo: {fila['titulo']}\nLetra: {fila.get('letra','')[:400]}\n"
                )

            elif tipo == "album":
                fila = (
                    supabase.table("album")
                    .select("*")
                    .eq("id_album", ev["id_album"])
                    .execute()
                    .data[0]
                )
                bloques.append(
                    f"[ÁLBUM]\nNombre: {fila['nombre_album']}\nDescripción: {fila.get('descripcion','')[:400]}\n"
                )

            elif tipo == "artista":
                fila = (
                    supabase.table("artista")
                    .select("*")
                    .eq("id_artista", ev["id_artista"])
                    .execute()
                    .data[0]
                )
                bloques.append(
                    f"[ARTISTA]\nNombre: {fila['nombre']}\nBio: {fila.get('bio','')[:400]}\n"
                )

        contexto = "\n\n".join(bloques)

        # ------------------------------------------------------------
        # 4. PROMPT ROBUSTO PARA LLM
        # ------------------------------------------------------------
        prompt = f"""
Eres un sistema de Respuesta Aumentada por Recuperación (RAG).
Debes responder **solo** usando el siguiente contexto proveniente de Supabase.

Si la respuesta NO está explícitamente en el contexto,
responde exactamente: "No hay información suficiente en la base de datos."

---------------- CONTEXTO ----------------
{contexto}
------------------------------------------

PREGUNTA:
{pregunta}

RESPUESTA:
"""

        # ------------------------------------------------------------
        # 5. LLM FINAL
        # ------------------------------------------------------------
        respuesta = generar_llm(prompt)

        return respuesta, evidencias

    except Exception as e:
        return f"ERROR en RAG Global: {str(e)}", []


# =============================================================================
# UI GRADIO — Panel completo (buscador, RAG, analítica, consultas, reseñas)
# =============================================================================
# -----------------------------------------------------------------------------
# Formateador de evidencias para mostrar en la UI
# -----------------------------------------------------------------------------
def prettify_evidencias(evidencias):
    if not evidencias:
        return "Sin evidencias."

    salida = []
    for ev in evidencias:
        block = []

        if "id_cancion" in ev:
            block.append(f"Cancion {ev['id_cancion']}")
        if "id_album" in ev:
            block.append(f"Album {ev['id_album']}")

        if "titulo" in ev:
            block.append(f"Título: {ev['titulo']}")
        if "score" in ev:
            block.append(f"Score: {round(ev['score'],4)}")

        salida.append("\n".join(block))

    return "\n\n".join(salida)



# -----------------------------------------------------------------------------
# Wrappers RPC para la UI
# -----------------------------------------------------------------------------
def ui_buscar_canciones(texto, top_k):
    try:
        return rpc_match_canciones_por_letra(texto, top_k)
    except Exception as e:
        return [{"error": str(e)}]

def ui_buscar_albumes(texto, top_k):
    try:
        return rpc_match_album_por_texto(texto, top_k)
    except Exception as e:
        return [{"error": str(e)}]

def ui_plan_mas_usado():
    try:
        return rpc_plan_mas_usado()
    except Exception as e:
        return [{"error": str(e)}]

def ui_buscar_portadas_imagen(img, top_k):
    if img is None:
        return [{"error": "Debes subir una imagen"}], []

    try:
        items = rpc_match_portadas_por_imagen(img, top_k)

        urls = []
        for itm in items:
            r = (
                supabase.table("portada")
                .select("*")
                .eq("id_portada", itm["id_portada"])
                .execute()
                .data
            )
            if r:
                ruta = r[0]["ruta_archivo"]
                url = supabase.storage.from_("portadas").get_public_url(
                    ruta.replace("portadas/", "")
                )
                urls.append(url)

        return items, urls
    except Exception as e:
        return [{"error": str(e)}], []



# -----------------------------------------------------------------------------
# RAG híbrido (canciones + álbumes + LLM)
# -----------------------------------------------------------------------------
def rag_hibrido(pregunta, top_k):
    try:
        # Canciones
        ev_cancion = rpc_match_canciones_por_letra(pregunta, top_k)
        # Álbumes
        ev_album = rpc_match_album_por_texto(pregunta, top_k)

        evidencias = []
        for e in ev_cancion:
            e["tipo"] = "cancion"
            evidencias.append(e)
        for e in ev_album:
            e["tipo"] = "album"
            evidencias.append(e)

        if not evidencias:
            return "No se encontró evidencia.", []

        # Construcción del contexto
        bloques = []
        for ev in evidencias:
            if ev["tipo"] == "cancion":
                fila = (
                    supabase.table("cancion")
                    .select("*")
                    .eq("id_cancion", ev["id_cancion"])
                    .execute()
                    .data[0]
                )
                bloques.append(
                    f"[CANCIÓN]\nTítulo: {fila['titulo']}\nLetra: {fila.get('letra','')[:400]}"
                )

            if ev["tipo"] == "album":
                fila = (
                    supabase.table("album")
                    .select("*")
                    .eq("id_album", ev["id_album"])
                    .execute()
                    .data[0]
                )
                bloques.append(
                    f"[ÁLBUM]\nTítulo: {fila['titulo']}\nDescripción: {fila.get('descripcion','')[:400]}"
                )

        contexto = "\n\n".join(bloques)

        # Prompt LLM
        prompt = f"""
Usa exclusivamente el siguiente contexto para responder.
Si la respuesta no está en el contexto, responde:
"No hay información suficiente en la base de datos."

CONTEXT:
{contexto}

PREGUNTA:
{pregunta}

RESPUESTA:
"""
        respuesta = generar_llm(prompt)
        return respuesta, evidencias

    except Exception as e:
        return f"ERROR en RAG híbrido: {e}", []



# -----------------------------------------------------------------------------
# Wrappers RPC analítica para UI
# -----------------------------------------------------------------------------
def ui_stats_globales():
    try:
        return rpc_estadisticas_generales()
    except Exception as e:
        return {"error": str(e)}

def ui_top_artistas(id_usuario, k):
    try:
        return rpc_top_artistas_por_usuario(id_usuario, k)
    except Exception as e:
        return [{"error": str(e)}]

def ui_top_playlists(id_usuario, k):
    try:
        return rpc_top_playlists_usuario(id_usuario, k)
    except Exception as e:
        return [{"error": str(e)}]



# -----------------------------------------------------------------------------
# Wrappers RPC transaccionales (crear consulta, crear reseña)
# -----------------------------------------------------------------------------
def ui_crear_consulta(uid, texto):
    try:
        return rpc_crear_consulta_con_embedding_atomic(uid, texto)
    except Exception as e:
        return {"error": str(e)}

def ui_crear_resena(uid, tipo, oid, cal, comentario):
    try:
        return rpc_crear_resena_atomic(uid, tipo, oid, cal, comentario)
    except Exception as e:
        return {"error": str(e)}



# -----------------------------------------------------------------------------
# Definición del bloque Gradio y lanzamiento
# -----------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# **Spotify RAG + Supabase — Panel Completo**")

    # ---------------------------------------------
    # Buscador Vectorial
    # ---------------------------------------------
    with gr.Tab("Buscador Vectorial"):
        with gr.Tabs():

            with gr.Tab("Canciones por letra"):
                q = gr.Textbox(label="Texto")
                k = gr.Slider(1, 10, value=5)
                out = gr.JSON()
                gr.Button("Buscar").click(ui_buscar_canciones, [q, k], out)

            with gr.Tab("Album por texto"):
                qa = gr.Textbox(label="Descripción")
                ka = gr.Slider(1, 10, value=5)
                outa = gr.JSON()
                gr.Button("Buscar").click(ui_buscar_albumes, [qa, ka], outa)

            with gr.Tab("Portadas por imagen"):
                img = gr.Image(type="pil")
                kp = gr.Slider(1, 10, value=5)
                meta = gr.JSON()
                gal = gr.Gallery()
                gr.Button("Buscar").click(
                    ui_buscar_portadas_imagen, [img, kp], [meta, gal]
                )

    # ---------------------------------------------
    # RAG Hibrido
    # ---------------------------------------------
    with gr.Tab("RAG SQL + LLM"):
        pregunta = gr.Textbox(label="Pregunta")
        k_rag = gr.Slider(1, 10, value=3)
        out = gr.Markdown()
        evid = gr.Markdown()

        def run_rag(q, k):
            r = rag_hibrido(q, k)
            return r[0], prettify_evidencias(r[1])

        gr.Button("Consultar").click(run_rag, [pregunta, k_rag], [out, evid])

    # Analítica
    with gr.Tab("Analítica SQL"):
        uid = gr.Number(label="ID Usuario", value=1)
        k_top = gr.Slider(1, 10, value=5)

        out_stats = gr.JSON()
        gr.Button("Estadísticas Globales").click(ui_stats_globales, None, out_stats)

        out_plan = gr.JSON()
        gr.Button("Plan más usado").click(ui_plan_mas_usado, None, out_plan)

        out_art = gr.JSON()
        gr.Button("Top Artistas").click(ui_top_artistas, [uid, k_top], out_art)

        out_pl = gr.JSON()
        gr.Button("Top Playlists").click(ui_top_playlists, [uid, k_top], out_pl)


    # Crear Consulta
    with gr.Tab("Crear Consulta"):
        uid2 = gr.Number(label="ID Usuario")
        txt2 = gr.Textbox(label="Texto de la consulta")
        out2 = gr.JSON()

        gr.Button("Guardar consulta").click(
            ui_crear_consulta, [uid2, txt2], out2
        )

    # Crear Reseña
    with gr.Tab("Crear Reseña"):
        uid3 = gr.Number(label="ID Usuario")
        tipo3 = gr.Dropdown(["cancion", "album", "playlist"], label="Tipo de objeto")
        oid3 = gr.Number(label="ID Objeto")
        cal3 = gr.Slider(1, 5, step=1, label="Calificación")
        com3 = gr.Textbox(label="Comentario")
        out3 = gr.JSON()

        gr.Button("Crear reseña").click(
            ui_crear_resena, [uid3, tipo3, oid3, cal3, com3], out3
        )

demo.launch(share=True)

