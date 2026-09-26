import os
import torch
import torch.nn as nn
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# ================= 1. MODEL ARCHITECTURE DEFINITION =================
class DINOv2Classifier(nn.Module):
    def __init__(self, backbone_model):
        super().__init__()
        self.backbone = backbone_model
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
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
    """Wraps model to output probability values (0.0 to 1.0) for ONNX inference."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        return torch.sigmoid(logits)

# ================= 2. PATH RESOLUTION & EXPORT =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FP32_MODEL_PATH = os.path.join(SCRIPT_DIR, "full_model_fp32.pth")
TEMP_ONNX_PATH  = os.path.join(SCRIPT_DIR, "dinov2_temp.onnx")
FINAL_ONNX_PATH = os.path.join(SCRIPT_DIR, "dinov2_mri_int8.onnx")

def export_fp32_to_onnx_int8():
    device = torch.device('cpu')

    if not os.path.exists(FP32_MODEL_PATH):
        raise FileNotFoundError(
            f"❌ 'full_model_fp32.pth' was not found in:\n{SCRIPT_DIR}\n"
            f"Please copy 'full_model_fp32.pth' into this folder before running the script."
        )

    print(f"📦 Loading FP32 file from: {FP32_MODEL_PATH}")
    checkpoint = torch.load(FP32_MODEL_PATH, map_location=device)

    # Check if loaded object is a full model instance or a state dictionary
    if isinstance(checkpoint, torch.nn.Module):
        base_model = checkpoint
    else:
        print("⏳ Instantiating DINOv2 backbone and loading parameters...")
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        base_model = DINOv2Classifier(backbone)
        
        # Remap state dict keys to remove the extra 'head.' prefix
        cleaned_state_dict = {}
        for key, val in checkpoint.items():
            if key.startswith("head.classifier."):
                cleaned_state_dict[key.replace("head.classifier.", "classifier.")] = val
            elif key.startswith("head."):
                cleaned_state_dict[key.replace("head.", "")] = val
            else:
                cleaned_state_dict[key] = val

        base_model.load_state_dict(cleaned_state_dict)

    base_model.eval()

    # Wrap model with Sigmoid for deployment
    export_model = DINOv2DeploymentWrapper(base_model)
    export_model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # Export intermediate FP32 ONNX graph
    print(f"⚡ Exporting Float32 ONNX graph to: {TEMP_ONNX_PATH}")
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

    # Dynamically quantize ONNX graph to INT8
    print(f"🗜️ Dynamically quantizing ONNX model to INT8: {FINAL_ONNX_PATH}")
    quantize_dynamic(
        model_input=TEMP_ONNX_PATH,
        model_output=FINAL_ONNX_PATH,
        weight_type=QuantType.QInt8
    )

    # Consolidate external data tensors into a single file
    print("🔗 Consolidating into a single unified ONNX file...")
    quant_model = onnx.load(FINAL_ONNX_PATH)
    onnx.save_model(quant_model, FINAL_ONNX_PATH, save_as_external_data=False)

    # Clean up temporary split files
    for temp_file in [TEMP_ONNX_PATH, f"{TEMP_ONNX_PATH}.data", f"{FINAL_ONNX_PATH}.data"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print(f"\n🎉 Success! Render-ready file created: '{os.path.basename(FINAL_ONNX_PATH)}' (~23.5 MB)")

if __name__ == '__main__':
    export_fp32_to_onnx_int8()