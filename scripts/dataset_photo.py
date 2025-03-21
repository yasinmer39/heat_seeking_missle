from picamera2 import Picamera2
import time
import os

# Initialize camera
picam2 = Picamera2(1)

# Configure and start the camera
picam2.configure(picam2.create_still_configuration())
picam2.start()

# Create an output directory
output_dir = "captured_images"
os.makedirs(output_dir, exist_ok=True)

# Capture 200 images in 60 seconds (~every 0.3 seconds)
num_photos = 200
interval = 60 / num_photos  # Time between each capture

print("Starting capture...")
start_time = time.time()

for i in range(num_photos):
    timestamp = time.time() - start_time
    filename = os.path.join(output_dir, f"image_{i+1:03d}.jpg")
    picam2.capture_file(filename)
    print(f"Captured {filename} at {timestamp:.2f} seconds")
    time.sleep(interval)

print("Capture complete.")
picam2.stop()
