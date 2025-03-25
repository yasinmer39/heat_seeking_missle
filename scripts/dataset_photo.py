from picamera2 import Picamera2
import time
import os
import cv2  # Import OpenCV

# Initialize cameras
picam1 = Picamera2(0)
picam1.configure(picam1.create_still_configuration())
picam1.start()

picam2 = Picamera2(1)
picam2.configure(picam2.create_still_configuration())
picam2.start()

# Create output directories
output_dir_left = "captured_images_left"
output_dir_right = "captured_images_right"
os.makedirs(output_dir_left, exist_ok=True)
os.makedirs(output_dir_right, exist_ok=True)

# Capture parameters
num_photos = 20
interval = 60 / num_photos  # Time between each capture

print("Starting capture...")
start_time = time.time()

for i in range(num_photos):
    frame = picam2.capture_array()  # Capture frame from camera 2
    frame_resized = cv2.resize(frame, (640, 480))  # Adjust size as needed
    
    timestamp = time.time() - start_time  # FIXED indentation

    filename_left = os.path.join(output_dir_left, f"image_{i+1:03d}.jpg")
    filename_right = os.path.join(output_dir_right, f"image_{i+1:03d}.jpg")

    picam1.capture_file(filename_left)
    picam2.capture_file(filename_right)
    
    print(f"Captured {filename_left} at {timestamp:.2f} seconds")
    print(f"Captured {filename_right} at {timestamp:.2f} seconds")

    # Show frame
    cv2.imshow("Camera 2", frame_resized)
    cv2.waitKey(1)  # Needed to refresh the OpenCV window

    time.sleep(interval)

print("Capture complete.")
picam1.stop()
picam2.stop()

cv2.destroyAllWindows()  # Close OpenCV windows
