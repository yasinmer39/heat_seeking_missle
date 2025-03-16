import cv2
import numpy as np
import tensorflow.lite as tflite
from PIL import Image
from picamera2 import Picamera2

# Initialize cameras
noir_cam = Picamera2(0)
visual_cam = Picamera2(1)

video_config_0 = noir_cam.create_video_configuration()
video_config_1 = visual_cam.create_video_configuration()

noir_cam.configure(video_config_0)
visual_cam.configure(video_config_1)

noir_cam.start()
visual_cam.start()

# Load TFLite model
interpreter = tflite.Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label map
with open("labelmap.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ORB feature detector for alignment
orb = cv2.ORB_create(nfeatures=500)

def align_images(img1, img2):
    """Align NoIR image to the RGB image using feature matching."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if len(matches) < 10:
        return img1  # Return unaligned image if not enough matches

    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned_noir = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

    return aligned_noir

def detect_objects(image):
    """Run object detection on the given image."""
    img_resized = cv2.resize(image, (300, 300))
    input_data = np.expand_dims(img_resized, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    height, width, _ = image.shape
    detected = False
    plane_detected = False
    fire_detected = False

    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            ymin, xmin, ymax, xmax = boxes[i]
            xmin, xmax, ymin, ymax = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)
            class_name = labels[int(classes[i])]
            confidence = scores[i]

            # Draw bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, f"{class_name} ({confidence:.2f})", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detected = True
            if "plane" in class_name.lower():
                plane_detected = True
            if "fire" in class_name.lower():
                fire_detected = True

    return image, detected, plane_detected, fire_detected

while True:
    frame_noir = noir_cam.capture_array()
    frame_visual = visual_cam.capture_array()

    frame_visual = cv2.cvtColor(frame_visual, cv2.COLOR_BGR2RGB)
    frame_noir = align_images(frame_noir, frame_visual)

    gray_noir = cv2.cvtColor(frame_noir, cv2.COLOR_BGR2GRAY)
    thermal_noir = cv2.applyColorMap(gray_noir, cv2.COLORMAP_JET)

    mean_intensity = np.mean(gray_noir)
    threshold_value = max(200, mean_intensity + 30)
    _, fire_mask = cv2.threshold(gray_noir, threshold_value, 255, cv2.THRESH_BINARY)

    fire_mask_colored = cv2.cvtColor(fire_mask, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            cv2.drawContours(fire_mask_colored, [cnt], -1, (0, 0, 0), thickness=cv2.FILLED)

    fire_region = cv2.bitwise_and(thermal_noir, fire_mask_colored)
    fused_result = cv2.addWeighted(frame_visual, 1, fire_region, 0.7, 0)

    # Run object detection
    detected_img, detected, plane_detected, fire_detected = detect_objects(fused_result)

    # Classification logic
    if plane_detected and fire_detected:
        classification = "SUCCESSFUL TARGET"
        color = (0, 255, 0)
    elif fire_detected:
        classification = "CHAFF FLARE"
        color = (0, 0, 255)
    else:
        classification = "NO THREAT DETECTED"
        color = (255, 255, 255)

    cv2.putText(detected_img, classification, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display results
    cv2.imshow("Fused Camera", detected_img)

    if cv2.waitKey(1) == ord('q'):
        break

noir_cam.close()
visual_cam.close()
cv2.destroyAllWindows()
