from picamera2 import Picamera2
import cv2
import numpy as np
import socket

# UDP socket (alıcı) setup
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
    frame = cv2.resize(frame, (640, 480))  # İstersen boyutlandır

    data, addr = sock.recvfrom(1024)  # 1024 byte buffer
    message = data.decode()
    print(f"Received detection: {message}")

    # HSL renk uzayına çevir
    hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # HSL aralıklarını ayarla
    lower_bound = np.array([0, 175, 0])
    upper_bound = np.array([11, 255, 8])

    # Maskele ve sonucu bul
    mask = cv2.inRange(hsl, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Ekrana bas
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    # 'q' tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Her şeyi kapat
noir_cam.close()
cv2.destroyAllWindows()