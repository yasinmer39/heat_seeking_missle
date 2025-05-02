from picamera2 import Picamera2
import cv2
import numpy as np

# Kamera init kısmı
noir_cam = Picamera2(1)
visual_cam = Picamera2(0)

video_config_1 = noir_cam.create_video_configuration()
video_config_0 = visual_cam.create_video_configuration()

noir_cam.configure(video_config_1)
visual_cam.configure(video_config_0)

noir_cam.start()
visual_cam.start()

# Hizalama için özellik dedektörü
orb = cv2.ORB_create(nfeatures=500)

def align_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if len(matches) < 10:
        print("yeterli eslesme yok babafungo")
        return img1

    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned_noir = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
    
    return aligned_noir

def nothing(x):
    pass

# Trackbar'ları oluştur
cv2.namedWindow('Trackbars')
cv2.createTrackbar('H Lower', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('H Upper', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('L Lower', 'Trackbars', 0, 255, nothing)  # S yerine L
cv2.createTrackbar('L Upper', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('S Lower', 'Trackbars', 0, 255, nothing)  # V yerine S
cv2.createTrackbar('S Upper', 'Trackbars', 255, 255, nothing)

while True:
    frame = noir_cam.capture_array()
    frame = cv2.resize(frame, (640, 480))  # İstersen boyutlandır
    frame_visual = visual_cam.capture_array()

    # HSL renk uzayına çevir
    hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # Trackbar'lardan değerleri oku
    h_lower = cv2.getTrackbarPos('H Lower', 'Trackbars')
    h_upper = cv2.getTrackbarPos('H Upper', 'Trackbars')
    l_lower = cv2.getTrackbarPos('L Lower', 'Trackbars')
    l_upper = cv2.getTrackbarPos('L Upper', 'Trackbars')
    s_lower = cv2.getTrackbarPos('S Lower', 'Trackbars')
    s_upper = cv2.getTrackbarPos('S Upper', 'Trackbars')

    # HSL aralıklarını ayarla
    lower_bound = np.array([h_lower, l_lower, s_lower])
    upper_bound = np.array([h_upper, l_upper, s_upper])

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
visual_cam.close()
cv2.destroyAllWindows()
