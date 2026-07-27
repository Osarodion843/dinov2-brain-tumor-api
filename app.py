import os
import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException

# --- CPU Runtime Setup ---
device = torch.device('cpu')      
torch.set_num_threads(1)          
os.environ["DNNL_MAX_CPU_ISA"] = "NONE"

# =====================================================================
# 1. MODEL ARCHITECTURE & LOAD
# =====================================================================
class DINOv2Classifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features)

sys.modules['__main__'].DINOv2Classifier = DINOv2Classifier

QUANTIZED_MODEL = "int8_dynamic_full_model.pth"

try:
    _ = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)
    model = torch.load(QUANTIZED_MODEL, map_location=device, weights_only=False)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# =====================================================================
# 2. FASTAPI ENDPOINT
# =====================================================================
app = FastAPI(title="DINOv2 Brain Tumor Classifier API")

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

@app.get("/")
def home():
    return {"status": "online", "message": "DINOv2 API is live! Go to /docs to test inference."}

@app.post("/predict")
async def predict_mri(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype("float32") / 255.0
        img = (img - MEAN) / STD
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            prob = model(img_tensor).item()
            
        label = "tumor" if prob > 0.5 else "healthy"
        prob_percentage = f"{round(prob * 100, 2)}%"
        
        return {
            "filename": file.filename,
            "tumor_probability": prob_percentage,
            "prediction": label
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Render assigns dynamic port numbers via environment variables
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)