# General imports used throughout the tutorial
import tensorflow as tf
from IPython.display import SVG

# import the ClientRunner class from the hailo_sdk_client package
from hailo_sdk_client import ClientRunner

chosen_hw_arch = "hailo8"
# For Hailo-15 devices, use 'hailo15h'
# For Mini PCIe modules or Hailo-8R devices, use 'hailo8r'

onnx_model_name = "yolo11n"
onnx_path = "yolo11n.onnx"

runner = ClientRunner(hw_arch=chosen_hw_arch)
hn, npz = runner.translate_onnx_model(
    onnx_path,
    onnx_model_name,
    start_node_names=["images"],  # Keep this as the correct input node name
    end_node_names=[
    "/model.23/cv2.0/cv2.0.2/Conv",
    "/model.23/cv3.0/cv3.0.2/Conv",
    "/model.23/cv3.1/cv3.1.2/Conv",
    "/model.23/cv2.1/cv2.1.2/Conv",
    "/model.23/cv3.2/cv3.2.2/Conv",
    "/model.23/cv2.2/cv2.2.2/Conv",],  # Update to the suggested end node
    net_input_shapes={"images": [1, 3, 640, 640]},  # Ensure this matches your ONNX model's input shape
)

#runner.save_hn("yolo11n.hn")
runner.save_har("yolo11n.har")
print(f"HAR file generated: ")