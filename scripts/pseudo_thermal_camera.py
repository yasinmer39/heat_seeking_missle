from picamera2 import Picamera2
import cv2
import numpy as np
import socket

# # UDP socket (alıcı) setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('localhost', 5005))

noir_cam = Picamera2(1)
video_config_1 = noir_cam.create_video_configuration()
noir_cam.configure(video_config_1)
noir_cam.start()

def nothing(x):
    pass

while True:
    frame = noir_cam.capture_array()
    frame = cv2.resize(frame, (640, 480))  # Resize if needed

    data, addr = sock.recvfrom(1024)  # 1024 byte buffer
    message = data.decode()
    print(f"Received detection: {message}")

    # Parse the message to extract confidence and bounding box
    try:
        label, confidence, bbox = message.split(',', 2)  # Split into label, confidence, and bbox
        confidence = float(confidence)

        # Remove parentheses and split bbox into individual coordinates
        bbox = bbox.strip('()')
        x_min, y_min, x_max, y_max = map(float, bbox.split(','))

        # Scale normalized coordinates to image dimensions
        x_min = int(x_min * frame.shape[1])
        y_min = int(y_min * frame.shape[0])
        x_max = int(x_max * frame.shape[1])
        y_max = int(y_max * frame.shape[0])

        # HSL color space conversion
        hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        # Set HSL range
        lower_bound = np.array([0, 175, 0])
        upper_bound = np.array([11, 255, 8])

        # Mask and find the result
        mask = cv2.inRange(hsl, lower_bound, upper_bound)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Draw the bounding box on the result frame in green
        cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(result, f"{label} Conf: {confidence:.2f}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    except ValueError as ve:
        print(f"Error parsing message: {ve}")
        continue
    except Exception as e:
        print(f"Unexpected error: {e}")
        continue

    # Display the result
    cv2.imshow('Result', result)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
noir_cam.close()
cv2.destroyAllWindows()