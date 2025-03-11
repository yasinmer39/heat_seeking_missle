from picamera2 import Picamera2
import cv2
import numpy as np

# Initialize NoIr camera
noir_cam = Picamera2(0)
video_config_0 = noir_cam.create_video_configuration()
noir_cam.configure(video_config_0)
noir_cam.start()

# Initialize Visual camera
visual_cam = Picamera2(1)
video_config_1 = visual_cam.create_video_configuration()
visual_cam.configure(video_config_1)
visual_cam.start()

while True:
    frame = noir_cam.capture_array()
    frame1 = visual_cam.capture_array()
    
    #BGR RGB yap
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale (intensity-based)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a false thermal colormap (e.g., COLORMAP_JET)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    # Show the result
    cv2.imshow("Pseudo Thermal Camera", thermal)
    cv2.imshow("Visual CAM", frame1)
    
    if cv2.waitKey(1) == ord('q'):
        break

noir_cam.close()
visual_cam.close()
cv2.destroyAllWindows()

