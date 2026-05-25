from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import insightface
import cv2
import numpy as np

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODELO MAIS LEVE
model = insightface.app.FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"]
)

# DETECTION SIZE MENOR = MAIS RÁPIDO
model.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

@app.get("/")
def home():
    return {
        "status": "online"
    }

@app.post("/recognize")
async def recognize(file: UploadFile):

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

        # VALIDAÇÃO
        if img is None:
            return {
                "error": "imagem inválida"
            }

        # REDIMENSIONAMENTO
        max_width = 1280

        h, w = img.shape[:2]

        if w > max_width:

            scale = max_width / w

            new_w = int(w * scale)
            new_h = int(h * scale)

            img = cv2.resize(
                img,
                (new_w, new_h)
            )

        # PROCESSAMENTO
        faces = model.get(img)

        result = []

        for face in faces:

            result.append({

                # posição do rosto
                "bbox": face.bbox.tolist(),

                # confiança
                "det_score": float(face.det_score),

                # embedding resumido
                # NÃO RETORNE O VETOR TODO
                "embedding_preview": face.embedding[:5].tolist()
            })

        return {
            "success": True,
            "faces_found": len(result),
            "faces": result
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }
