from hailo_sdk_client import ClientRunner

# Create runner with Hailo8 architecture
runner = ClientRunner(hw_arch="hailo8")

# Load your previously converted .har file
runner.load_har("yolo11n.har")

# Define full optimization config (with top-level 'optimization' key)
optimization_config = {
    "optimization": {
        "calib_set": {
            "type": "raw_images",
            "path": "/home/user/heat_seeking_missile/yolo/calib_images_resized",
            "size": 100
        }
    }
}

# Perform optimization
optimized_har = runner.optimize("optimize_config.yaml")
print(f"✅ Optimized HAR saved to: {optimized_har}")
