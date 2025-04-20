from picamera2 import Picamera2
import cv2
import numpy as np

#kamera init kismi
noir_cam = Picamera2(0)
visual_cam = Picamera2(1)

video_config_0 = noir_cam.create_video_configuration()
video_config_1 = visual_cam.create_video_configuration()

noir_cam.configure(video_config_0)
visual_cam.configure(video_config_1)

noir_cam.start()
visual_cam.start()

#hizalama icin ozellik dedektoru
orb = cv2.ORB_create(nfeatures=500)

def align_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    #en yakin komsu arama
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    
    if len(matches) < 10:
        print("yeterli eslesme yok babafungo")
        return img1

    #mesafeye dayali eslesmeleri sort et
    matches = sorted(matches, key=lambda x: x.distance)

    #eslesmeye karsilik gelen noktalari cikart
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    #perspective transform yap ve noir goruntuye uygula
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned_noir = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
    
    return aligned_noir

while True:
    frame_noir = noir_cam.capture_array()
    frame_visual = visual_cam.capture_array()
    
    #bgr->rgb visual icin
    frame_visual = cv2.cvtColor(frame_visual, cv2.COLOR_BGR2RGB)
    
    #noir ve visual hizala, noir graycsale donustur ve thermal kamera gibi yapan colormap ekle
    frame_noir = align_images(frame_noir, frame_visual)
    gray_noir = cv2.cvtColor(frame_noir, cv2.COLOR_BGR2GRAY)
    thermal_noir = cv2.applyColorMap(gray_noir, cv2.COLORMAP_JET)

    # Fire detection using adaptive thresholding
    mean_intensity = np.mean(gray_noir)
    threshold_value = max(200, mean_intensity + 30)  # Dynamically adjust
    _, fire_mask = cv2.threshold(gray_noir, threshold_value, 255, cv2.THRESH_BINARY)

    # Convert fire mask to 3-channel
    fire_mask_colored = cv2.cvtColor(fire_mask, cv2.COLOR_GRAY2BGR)

    # Reduce false positives (small regions only)
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # Ignore small noise
            cv2.drawContours(fire_mask_colored, [cnt], -1, (0, 0, 0), thickness=cv2.FILLED)

    # Extract fire region from the thermal image
    fire_region = cv2.bitwise_and(thermal_noir, fire_mask_colored)

    # Overlay fire region onto the visual camera feed
    fused_result = cv2.addWeighted(frame_visual, 1, fire_region, 0.7, 0)

    # Display results
    cv2.imshow("Fused Camera", fused_result)

    if cv2.waitKey(1) == ord('q'):
        break

noir_cam.close()
visual_cam.close()
cv2.destroyAllWindows()
