import os
import io
import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

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

# Load lightweight quantized ONNX model
MODEL_PATH = "dinov2_mri_int8.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Dynamically pull exact input/output names from ONNX graph ("input_image", "tumor_probability")
INPUT_NAME = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

@app.get("/")
def home():
    return {"status": "online", "message": "API running on lightweight ONNX engine"}

@app.post("/predict")
async def predict_mri(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        
        # Image loading and conversion using PIL (avoids OpenCV system library issues on Render)
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image = image.resize((224, 224))
        
        # Normalize and reshape to (1, 3, 224, 224)
        img_arr = np.array(image, dtype=np.float32) / 255.0
        img_arr = (img_arr - MEAN) / STD
        img_tensor = np.transpose(img_arr, (2, 0, 1))[np.newaxis, :]
        
        # Inference via ONNX Runtime
        outputs = session.run([OUTPUT_NAME], {INPUT_NAME: img_tensor})
        
        # Safely extract probability from array
        prob = float(outputs[0].flat[0])
        
        label = "tumor" if prob >= 0.5 else "healthy"
        
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