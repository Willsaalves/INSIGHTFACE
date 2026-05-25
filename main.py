from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import insightface
import cv2
import numpy as np
import time

app = FastAPI()

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODEL
# =========================

model = insightface.app.FaceAnalysis(
    name="buffalo_sc",
    providers=["CPUExecutionProvider"]
)

model.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

# =========================
# HOME
# =========================

@app.get("/")
def home():

    return {
        "status": "online",
        "model": "buffalo_sc"
    }

# =========================================================
# ENDPOINT 1
# EXTRAIR EMBEDDING DA SELFIE
# =========================================================

@app.post("/extract-embedding")

async def extract_embedding(file: UploadFile):

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
                "error": "imagem inválida"
            }

        # resize selfie
        img = resize_image(img, 640)

        faces = model.get(
            img,
            max_num=1
        )

        if len(faces) == 0:

            return {
                "success": False,
                "error": "nenhum rosto encontrado"
            }

        # PEGA MAIOR ROSTO
        main_face = get_main_face(faces)

        end = time.time()

        return {

            "success": True,

            "processing_time_seconds": round(
                end - start,
                2
            ),

            "face": {

                "bbox": {
                    "x1": float(main_face.bbox[0]),
                    "y1": float(main_face.bbox[1]),
                    "x2": float(main_face.bbox[2]),
                    "y2": float(main_face.bbox[3]),
                },

                "score": float(
                    main_face.det_score
                ),

                # embedding completo
                "embedding": main_face.embedding.astype(
                    np.float32
                ).tolist()
            }
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

# =========================================================
# ENDPOINT 2
# DETECTAR PRINCIPAL PESSOA DA FOTO
# =========================================================

@app.post("/extract-main-face")

async def extract_main_face(file: UploadFile):

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
                "error": "imagem inválida"
            }

        # resize evento
        img = resize_image(img, 960)

        # múltiplos rostos
        faces = model.get(img)

        if len(faces) == 0:

            return {
                "success": False,
                "error": "nenhum rosto encontrado"
            }

        # MAIOR ROSTO DA FOTO
        main_face = get_main_face(faces)

        end = time.time()

        return {

            "success": True,

            "processing_time_seconds": round(
                end - start,
                2
            ),

            "faces_found": len(faces),

            "main_face": {

                "bbox": {
                    "x1": float(main_face.bbox[0]),
                    "y1": float(main_face.bbox[1]),
                    "x2": float(main_face.bbox[2]),
                    "y2": float(main_face.bbox[3]),
                },

                "score": float(
                    main_face.det_score
                ),

                # embedding da pessoa principal
                "embedding": main_face.embedding.astype(
                    np.float32
                ).tolist()
            }
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

# =========================================================
# FUNÇÃO
# REDIMENSIONAR
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

# =========================================================
# FUNÇÃO
# PEGAR MAIOR ROSTO
# =========================================================

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
