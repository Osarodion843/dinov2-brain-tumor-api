import os
import torch
import torch.nn as nn
from onnxruntime.quantization import quantize_dynamic, QuantType

# ================= 1. MODEL ARCHITECTURE =================
class DINOv2Classifier(nn.Module):
    def __init__(self, backbone_model):
        super().__init__()
        self.backbone = backbone_model
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        for param in self.backbone.blocks[-2:].parameters():
            param.requires_grad = True
            
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features).squeeze(-1)

class DINOv2DeploymentWrapper(nn.Module):
    """Wraps the model to output probability values (0 to 1) for ONNX inference."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        return torch.sigmoid(logits)

# ================= 2. LOCAL FILE PATH RESOLUTION =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(SCRIPT_DIR, "dinov2_seed_12.pth")

TEMP_ONNX_PATH = os.path.join(SCRIPT_DIR, "dinov2_temp.onnx")
FINAL_ONNX_PATH = os.path.join(SCRIPT_DIR, "dinov2_mri_int8.onnx")

def export_local_checkpoint():
    device = torch.device('cpu')

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"❌ Could not find 'dinov2_seed_12.pth' in your working directory:\n"
            f"Expected Location: {CKPT_PATH}"
        )

    print(f"📦 Found local checkpoint at: {CKPT_PATH}")
    print("⏳ Instantiating DINOv2 backbone and loading model weights...")
    
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    base_model = DINOv2Classifier(backbone)
    
    state_dict = torch.load(CKPT_PATH, map_location=device)
    base_model.load_state_dict(state_dict)

    export_model = DINOv2DeploymentWrapper(base_model)
    export_model.eval()  # Enforce evaluation mode on wrapped module

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    print(f"⚡ Exporting float32 ONNX graph to: {TEMP_ONNX_PATH}")
    
    # Use legacy exporter (dynamo=False) & opset_version=18 to prevent shape inference corruption
    torch.onnx.export(
        export_model,
        dummy_input,
        TEMP_ONNX_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["tumor_probability"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "tumor_probability": {0: "batch_size"}
        },
        dynamo=False
    )

    print(f"🗜️ Quantizing model to INT8: {FINAL_ONNX_PATH}")
    quantize_dynamic(
        model_input=TEMP_ONNX_PATH,
        model_output=FINAL_ONNX_PATH,
        weight_type=QuantType.QInt8
    )

    if os.path.exists(TEMP_ONNX_PATH):
        os.remove(TEMP_ONNX_PATH)

    print(f"\n🎉 Success! Exported '{os.path.basename(FINAL_ONNX_PATH)}' into your local directory.")

if __name__ == '__main__':
    export_local_checkpoint()