import os
import sys
import torch
import torch.nn as nn
from onnxruntime.quantization import quantize_dynamic, QuantType

device = torch.device('cpu')

# 1. Model Architecture
class DINOv2Classifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.classifier(features)

sys.modules['__main__'].DINOv2Classifier = DINOv2Classifier

# Path to your saved weights file
pth_file_path = "dinov2_seed_12.pth"

print("📦 Instantiating model architecture and loading weights...")
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)
model = DINOv2Classifier(backbone)

# Load weights into the model
checkpoint = torch.load(pth_file_path, map_location=device)
if isinstance(checkpoint, dict) and not isinstance(checkpoint, nn.Module):
    model.load_state_dict(checkpoint)
else:
    model = checkpoint

model.eval()

# 2. Helper function to re-align layers for ONNX export
def convert_quantized_to_float(module):
    for name, child in list(module.named_children()):
        convert_quantized_to_float(child)
        if 'quantized' in type(child).__module__.lower():
            try:
                w = child.weight().dequantize()
                b = child.bias() if hasattr(child, 'bias') else None
                in_f = child.in_features
                out_f = child.out_features
                
                new_linear = nn.Linear(in_f, out_f)
                new_linear.weight = nn.Parameter(w)
                if b is not None:
                    new_linear.bias = nn.Parameter(b)
                
                setattr(module, name, new_linear)
            except Exception:
                pass

print("🔄 Re-aligning layers for ONNX export...")
convert_quantized_to_float(model)

# 3. Export to ONNX format using legacy exporter (dynamo=False) and opset 18
dummy_input = torch.randn(1, 3, 224, 224, device=device)
temp_onnx = "temp_model.onnx"
final_onnx = "model.onnx"

print("⚡ Exporting model graph to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    temp_onnx,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    dynamo=False
)

print("🗜️ Applying ONNX dynamic INT8 quantization...")
quantize_dynamic(
    temp_onnx,
    final_onnx,
    weight_type=QuantType.QUInt8
)

# Clean up uncompressed temporary file
if os.path.exists(temp_onnx):
    os.remove(temp_onnx)

print("✅ Success! 'model.onnx' created in your project directory.")