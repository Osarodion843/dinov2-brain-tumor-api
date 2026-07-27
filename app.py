import os
import cv2
import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Explicitly define HTTPS server URL for Render
app = FastAPI(
    title="DINOv2 Brain Tumor API",
    servers=[
        {
            "url": "https://dinov2-brain-tumor-api.onrender.com",
            "description": "Production Server (HTTPS)"
        }
    ]
)

# Open CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load lightweight ONNX model
session = ort.InferenceSession("model.onnx")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

@app.get("/")
def home():
    return {"status": "online", "message": "API running on lightweight ONNX engine"}

@app.post("/predict")
async def predict_mri(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
            
        # Image preprocessing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img_tensor = np.transpose(img, (2, 0, 1))[np.newaxis, :]
        
        # Inference via ONNX
        outputs = session.run(None, {"input": img_tensor})
        prob = float(outputs[0][0][0])
        
        label = "tumor" if prob > 0.5 else "healthy"
        
        return {
            "filename": file.filename,
            "tumor_probability": f"{round(prob * 100, 2)}%",
            "prediction": label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)