import torch
from ultralytics.nn.tasks import DetectionModel  # Import the custom class

# Allowlist the custom class
torch.serialization.add_safe_globals([DetectionModel])

# Load the saved PyTorch model from a .pt file
model_path = "yolo11n.pt"  # Replace with the path to your .pt file
torch_model = torch.load(model_path)
torch_model.eval()  # Set the model to evaluation mode

# Create example inputs for exporting the model
example_inputs = torch.randn(1, 3, 640, 640)  # Adjust input shape as needed

# Export the model to ONNX
torch.onnx.export(
    torch_model,                     # The model to be exported
    example_inputs,                  # Example input tensor
    "yolo11n_model.onnx",            # Output ONNX file path
    export_params=True,              # Store the trained parameter weights
    opset_version=11,                # ONNX opset version
    do_constant_folding=True,        # Optimize constant folding
    input_names=["input"],           # Input tensor names
    output_names=["output"],         # Output tensor names
    dynamic_axes={                   # Dynamic axes for variable input sizes
        "input": {0: "batch_size"}, 
        "output": {0: "batch_size"}
    }
)

print("Model has been successfully exported to ONNX format.")