from fastapi import FastAPI, UploadFile
import insightface
import cv2
import numpy as np

app = FastAPI()

model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0)

@app.get("/")
def home():
    return {"status": "online"}

@app.post("/recognize")
async def recognize(file: UploadFile):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    faces = model.get(img)

    result = []

    for face in faces:
        result.append({
            "bbox": face.bbox.tolist(),
            "embedding": face.embedding.tolist()
        })

    return {
        "faces_found": len(result),
        "faces": result
    }
