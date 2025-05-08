from picamera2 import Picamera2
import cv2
import numpy as np
import socket
import serial
import threading

ser = serial.Serial('/dev/ttyAMA10', 115200, timeout=1)

# UDP socket (alıcı) setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('localhost', 5005))

# Bu dict içine en son gelen detection mesajını yazacağız
detection_data = {'message': None}

# UDP dinleyici thread
def udp_listener():
    while True:
        data, _ = sock.recvfrom(1024)
        detection_data['message'] = data.decode()

# Thread’i başlat
udp_thread = threading.Thread(target=udp_listener, daemon=True)
udp_thread.start()

noir_cam = Picamera2(1)
video_config_1 = noir_cam.create_video_configuration()
noir_cam.configure(video_config_1)
noir_cam.start()

def nothing(x):
    pass

while True:
    frame = noir_cam.capture_array()
    frame = cv2.resize(frame, (640, 480))  # Resize if needed

    message = detection_data.get('message')
    if message:
        print(f"Received detection: {message}")

        try:
            label, confidence, bbox = message.split(',', 2)  # Split into label, confidence, bbox
            confidence = float(confidence)

            # Remove parentheses and split bbox into individual coordinates
            bbox = bbox.strip('()')
            x_min, y_min, x_max, y_max = map(float, bbox.split(','))

            # Scale normalized coordinates to image dimensions
            x_min = int(x_min * frame.shape[1])
            y_min = int(y_min * frame.shape[0])
            x_max = int(x_max * frame.shape[1])
            y_max = int(y_max * frame.shape[0])

            # Apply horizontal offset (positive = shift right, negative = shift left)
            pixel_offset_x = 110  # ~4 cm yatay kayma (deneyerek ayarlayabilirsin)
            #pixel_offset_y = -10  # ~4 cm yatay kayma (deneyerek ayarlayabilirsin)
            x_min = min(max(x_min + pixel_offset_x, 0), frame.shape[1])
            x_max = min(max(x_max + pixel_offset_x, 0), frame.shape[1])
            #y_min = min(max(x_min + pixel_offset_y, 0), frame.shape[1])
            #y_max = min(max(x_max + pixel_offset_y, 0), frame.shape[1])

            # HSL color space conversion
            hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            lower_bound = np.array([0, 175, 0])
            upper_bound = np.array([11, 255, 8])
            mask = cv2.inRange(hsl, lower_bound, upper_bound)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            result = cv2.flip(result, 1)

            # Draw a line from the center of the frame to the center of the bounding box
            bbox_center_x = (x_min + x_max) // 2
            bbox_center_y = (y_min + y_max) // 2
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            cv2.line(result, (frame_center_x, frame_center_y), (bbox_center_x, bbox_center_y), (255, 0, 0), 2)

            # Draw bounding box
            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(result, f"{label} Conf: {confidence:.2f}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Send center coordinates over serial
            ser.write(f"{bbox_center_x},{bbox_center_y}\n".encode())

        except ValueError as ve:
            print(f"Error parsing message: {ve}")
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue

    # Display the result
    cv2.imshow('Result', frame if message is None else result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
noir_cam.close()
cv2.destroyAllWindows()
