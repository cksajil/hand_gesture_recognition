import torch
from app import load_model


model = load_model("config.json")
model.eval()


# Create dummy input with 1 channel
dummy_input = torch.randn(1, 3, 18, 84, 84)


torch.onnx.export(
    model,
    dummy_input,
    "./models/model.onnx",
    verbose=True,
    input_names=["input"],
    output_names=["output"],
)
