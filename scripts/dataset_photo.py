from picamera2 import Picamera2
import time
import os
import cv2  # Import OpenCV

# Initialize cameras
picam1 = Picamera2(0)
picam1.configure(picam1.create_still_configuration())
picam1.start()

# Create output directories
output_dir = "captured_images_3"
os.makedirs(output_dir, exist_ok=True)

# Capture parameters
num_photos = 200
interval = 60 / num_photos  # Time between each capture

print("Starting capture...")
start_time = time.time()

for i in range(num_photos):
    frame = picam1.capture_array()  # Capture frame from camera 2
    frame_resized = cv2.resize(frame, (640, 480))  # Adjust size as needed
    
    timestamp = time.time() - start_time  # FIXED indentation

    filename = os.path.join(output_dir, f"image_{i+1:03d}.jpg")

    picam1.capture_file(filename)
    
    print(f"Captured {filename} at {timestamp:.2f} seconds")

    # Show frame
    cv2.imshow("Camera 2", frame_resized)
    cv2.waitKey(1)  # Needed to refresh the OpenCV window

    time.sleep(interval)

print("Capture complete.")
picam1.stop()

cv2.destroyAllWindows()  # Close OpenCV windows
