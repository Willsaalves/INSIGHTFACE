from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import insightface
import cv2
import numpy as np
import os
import uuid
import time

from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="AllParty Face API",
    version="2.0.0"
)

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# AUTH (opcional)
# Se FACE_API_KEY estiver definida, toda rota exceto "/" exige o
# header X-API-Key com o mesmo valor. Sem a variável, nada muda.
# =========================================================

FACE_API_KEY = os.environ.get("FACE_API_KEY", "").strip()

@app.middleware("http")
async def require_api_key(request: Request, call_next):

    if (
        FACE_API_KEY
        and request.method != "OPTIONS"
        and request.url.path != "/"
        and request.headers.get("x-api-key") != FACE_API_KEY
    ):

        return JSONResponse(
            {"success": False, "error": "unauthorized"},
            status_code=401
        )

    return await call_next(request)

# =========================================================
# MODEL
# =========================================================

model = insightface.app.FaceAnalysis(
    name="buffalo_sc",
    providers=["CPUExecutionProvider"]
)

model.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

# Fotos de grupo: rostos pequenos somem com det_size 320.
# Mesmo modelo de reconhecimento -> embeddings no mesmo espaço.
multi_model = insightface.app.FaceAnalysis(
    name="buffalo_sc",
    providers=["CPUExecutionProvider"]
)

MULTI_DET_SIZE = int(os.environ.get("MULTI_DET_SIZE", "1280"))

multi_model.prepare(
    ctx_id=0,
    det_size=(MULTI_DET_SIZE, MULTI_DET_SIZE)
)

MULTI_MAX_SIDE = int(os.environ.get("MULTI_MAX_SIDE", "1920"))

# =========================================================
# MEMORY DATABASE (MVP)
# depois trocar por postgres/pgvector
# =========================================================

FACESETS = {}

# =========================================================
# HELPERS
# =========================================================

def resize_image(img, max_width):

    h, w = img.shape[:2]

    if w > max_width:

        scale = max_width / w

        new_w = int(w * scale)
        new_h = int(h * scale)

        img = cv2.resize(
            img,
            (new_w, new_h)
        )

    return img

def get_main_face(faces):

    largest_face = None
    largest_area = 0

    for face in faces:

        x1, y1, x2, y2 = face.bbox

        width = x2 - x1
        height = y2 - y1

        area = width * height

        if area > largest_area:

            largest_area = area
            largest_face = face

    return largest_face

def extract_embedding_from_image(img):

    img = resize_image(img, 640)

    faces = model.get(img)

    if len(faces) == 0:
        return None

    main_face = get_main_face(faces)

    embedding = main_face.embedding.astype(
        np.float32
    )

    return {
        "face": main_face,
        "embedding": embedding
    }

def generate_face_token():

    return str(uuid.uuid4())

# =========================================================
# REQUEST MODELS
# =========================================================

class CompareRequest(BaseModel):

    embedding1: list
    embedding2: list

class CreateFaceSetRequest(BaseModel):

    display_name: str

class AddFaceRequest(BaseModel):

    faceset_token: str
    user_id: str
    embedding: list

class SearchRequest(BaseModel):

    faceset_token: str
    embedding: list

# =========================================================
# HOME
# =========================================================

@app.get("/")

def home():

    return {
        "status": "online",
        "api": "AllParty Face API",
        "version": "2.0.0"
    }

# =========================================================
# DETECT
# =========================================================

@app.post("/detect")

async def detect(file: UploadFile = File(...)):

    start = time.time()

    try:

        contents = await file.read()

        nparr = np.frombuffer(
            contents,
            np.uint8
        )

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if img is None:

            return {
                "success": False,
                "error": "invalid image"
            }

        img = resize_image(img, 1280)

        faces = model.get(img)

        response_faces = []

        for face in faces:

            face_token = generate_face_token()

            response_faces.append({

                "face_token": face_token,

                "bbox": {
                    "x1": float(face.bbox[0]),
                    "y1": float(face.bbox[1]),
                    "x2": float(face.bbox[2]),
                    "y2": float(face.bbox[3]),
                },

                "confidence": float(
                    face.det_score
                )
            })

        end = time.time()

        return {

            "success": True,

            "faces_detected": len(
                response_faces
            ),

            "processing_time_seconds": round(
                end - start,
                2
            ),

            "faces": response_faces
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

# =========================================================
# EXTRACT EMBEDDING
# =========================================================

@app.post("/extract-embedding")

async def extract_embedding(
    file: UploadFile = File(...)
):

    try:

        contents = await file.read()

        nparr = np.frombuffer(
            contents,
            np.uint8
        )

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if img is None:

            return {
                "success": False,
                "error": "invalid image"
            }

        result = extract_embedding_from_image(
            img
        )

        if result is None:

            return {
                "success": False,
                "error": "no face found"
            }

        face = result["face"]

        face_token = generate_face_token()

        return {

            "success": True,

            "face": {

                "face_token": face_token,

                "bbox": {
                    "x1": float(face.bbox[0]),
                    "y1": float(face.bbox[1]),
                    "x2": float(face.bbox[2]),
                    "y2": float(face.bbox[3]),
                },

                "confidence": float(
                    face.det_score
                ),

                "embedding": result[
                    "embedding"
                ].tolist()
            }
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

# =========================================================
# EXTRACT MAIN FACE
# =========================================================

@app.post("/extract-main-face")

async def extract_main_face(
    file: UploadFile = File(...)
):

    try:

        contents = await file.read()

        nparr = np.frombuffer(
            contents,
            np.uint8
        )

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if img is None:

            return {
                "success": False,
                "error": "invalid image"
            }

        result = extract_embedding_from_image(
            img
        )

        if result is None:

            return {
                "success": False,
                "error": "no face found"
            }

        face = result["face"]

        return {

            "success": True,

            "main_face": {

                "bbox": {
                    "x1": float(face.bbox[0]),
                    "y1": float(face.bbox[1]),
                    "x2": float(face.bbox[2]),
                    "y2": float(face.bbox[3]),
                },

                "confidence": float(
                    face.det_score
                ),

                "embedding": result[
                    "embedding"
                ].tolist()
            }
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

# =========================================================
# EXTRACT FACES (todos os rostos + embedding, uma chamada)
# bbox volta na escala da imagem ORIGINAL enviada.
# =========================================================

@app.post("/extract-faces")

async def extract_faces(
    file: UploadFile = File(...),
    min_face_size: int = 40,
    min_det_score: float = 0.5
):

    start = time.time()

    try:

        contents = await file.read()

        nparr = np.frombuffer(
            contents,
            np.uint8
        )

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if img is None:

            return {
                "success": False,
                "error": "invalid image"
            }

        orig_h, orig_w = img.shape[:2]

        scale = 1.0

        if max(orig_h, orig_w) > MULTI_MAX_SIDE:

            scale = MULTI_MAX_SIDE / max(orig_h, orig_w)

            img = cv2.resize(
                img,
                (int(orig_w * scale), int(orig_h * scale))
            )

        faces = multi_model.get(img)

        response_faces = []
        skipped = 0

        for face in faces:

            x1, y1, x2, y2 = [float(v) / scale for v in face.bbox]

            w = x2 - x1
            h = y2 - y1

            det_score = float(face.det_score)

            if min(w, h) < min_face_size or det_score < min_det_score:

                skipped += 1
                continue

            response_faces.append({

                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },

                "confidence": det_score,

                "embedding": face.embedding.astype(
                    np.float32
                ).tolist()
            })

        return {

            "success": True,

            "faces_found": len(response_faces),

            "faces_skipped": skipped,

            "image_size": {
                "width": orig_w,
                "height": orig_h
            },

            "processing_time_seconds": round(
                time.time() - start,
                2
            ),

            "faces": response_faces
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

# =========================================================
# COMPARE
# =========================================================

@app.post("/compare")

def compare_faces(data: CompareRequest):

    try:

        emb1 = np.array(
            data.embedding1
        ).reshape(1, -1)

        emb2 = np.array(
            data.embedding2
        ).reshape(1, -1)

        similarity = cosine_similarity(
            emb1,
            emb2
        )[0][0]

        return {

            "success": True,

            "similarity": round(
                float(similarity),
                4
            ),

            "is_match": bool(similarity > 0.45)
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

# =========================================================
# CREATE FACESET
# =========================================================

@app.post("/faceset/create")

def create_faceset(
    data: CreateFaceSetRequest
):

    try:

        faceset_token = str(
            uuid.uuid4()
        )

        FACESETS[
            faceset_token
        ] = {

            "display_name": data.display_name,

            "faces": []
        }

        return {

            "success": True,

            "faceset_token": faceset_token,

            "display_name": data.display_name
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

# =========================================================
# ADD FACE
# =========================================================

@app.post("/faceset/addface")

def add_face(data: AddFaceRequest):

    try:

        if data.faceset_token not in FACESETS:

            return {
                "success": False,
                "error": "faceset not found"
            }

        face_token = generate_face_token()

        FACESETS[
            data.faceset_token
        ]["faces"].append({

            "face_token": face_token,

            "user_id": data.user_id,

            "embedding": data.embedding
        })

        return {

            "success": True,

            "face_token": face_token
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

# =========================================================
# SEARCH
# =========================================================

@app.post("/search")

def search_face(data: SearchRequest):

    try:

        if data.faceset_token not in FACESETS:

            return {
                "success": False,
                "error": "faceset not found"
            }

        query_embedding = np.array(
            data.embedding
        ).reshape(1, -1)

        results = []

        for stored_face in FACESETS[
            data.faceset_token
        ]["faces"]:

            stored_embedding = np.array(
                stored_face["embedding"]
            ).reshape(1, -1)

            similarity = cosine_similarity(
                query_embedding,
                stored_embedding
            )[0][0]

            results.append({

                "user_id": stored_face[
                    "user_id"
                ],

                "face_token": stored_face[
                    "face_token"
                ],

                "confidence": round(
                    float(similarity),
                    4
                )
            })

        results = sorted(
            results,
            key=lambda x: x["confidence"],
            reverse=True
        )

        return {

            "success": True,

            "results": results[:5]
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }
