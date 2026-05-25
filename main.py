from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import insightface
import cv2
import numpy as np
import time

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODELO OTIMIZADO
model = insightface.app.FaceAnalysis(
    name="buffalo_sc",
    providers=["CPUExecutionProvider"]
)

model.prepare(
    ctx_id=0,
    det_size=(320, 320)
)

@app.get("/")
def home():

    return {
        "status": "online",
        "model": "buffalo_sc"
    }

@app.post("/recognize")
async def recognize(file: UploadFile):

    start = time.time()

    try:

        # LER ARQUIVO
        contents = await file.read()

        # CONVERTER PARA NUMPY
        nparr = np.frombuffer(
            contents,
            np.uint8
        )

        # DECODIFICAR IMAGEM
        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        # VALIDAR
        if img is None:

            return {
                "success": False,
                "error": "imagem inválida"
            }

        # REDIMENSIONAR
        max_width = 640

        h, w = img.shape[:2]

        if w > max_width:

            scale = max_width / w

            new_w = int(w * scale)
            new_h = int(h * scale)

            img = cv2.resize(
                img,
                (new_w, new_h)
            )

        # PROCESSAR
        faces = model.get(
            img,
            max_num=1
        )

        result = []

        for face in faces:

            # NORMALIZAR EMBEDDING
            embedding = face.embedding.astype(
                np.float32
            )

            result.append({

                "bbox": {
                    "x1": float(face.bbox[0]),
                    "y1": float(face.bbox[1]),
                    "x2": float(face.bbox[2]),
                    "y2": float(face.bbox[3]),
                },

                "score": float(face.det_score),

                # EMBEDDING COMPLETO
                "embedding": embedding.tolist()
            })

        end = time.time()

        return {
            "success": True,
            "faces_found": len(result),
            "processing_time_seconds": round(end - start, 2),
            "faces": result
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }
