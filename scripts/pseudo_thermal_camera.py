from picamera2 import Picamera2
import cv2
import numpy as np

# Initialize camera
picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)
picam2.start()

while True:
    frame = picam2.capture_array()
    
    # Convert to grayscale (intensity-based)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a false thermal colormap (e.g., COLORMAP_JET)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    # Show the result
    cv2.imshow("Pseudo Thermal Camera", thermal)
    
    if cv2.waitKey(1) == ord('q'):
        break

picam2.close()
cv2.destroyAllWindows()
