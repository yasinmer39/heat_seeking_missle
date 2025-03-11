from picamera2 import Picamera2
import cv2
import numpy as np

# NoIr camera init
noir_cam = Picamera2(0)
video_config_0 = noir_cam.create_video_configuration()
noir_cam.configure(video_config_0)
noir_cam.start()

# Visual camera init
visual_cam = Picamera2(1)
video_config_1 = visual_cam.create_video_configuration()
visual_cam.configure(video_config_1)
visual_cam.start()

# ORB feature detector for alignment
orb = cv2.ORB_create(500)

# BFMatcher for feature matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def align_images(thermal, frame1):
    """Aligns the thermal image to match the perspective of the visual camera."""
    gray_thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray_thermal, None)
    kp2, des2 = orb.detectAndCompute(gray_frame1, None)

    if des1 is None or des2 is None:
        print("Warning: No features detected.")
        return thermal  # Return original if no features are found

    # Match descriptors
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        print("Warning: Not enough matches.")
        return thermal  # Return original if not enough matches

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("Warning: Homography computation failed.")
        return thermal  # Return original if homography fails

    # Warp the thermal image
    aligned_thermal = cv2.warpPerspective(thermal, H, (frame1.shape[1], frame1.shape[0]))

    return aligned_thermal

while True:
    frame0 = noir_cam.capture_array()
    frame1 = visual_cam.capture_array()

    # Ensure frame1 has 3 channels
    if frame1.shape[-1] == 4:  
        frame1 = frame1[:, :, :3]  # Convert to 3 channels if necessary

    # Align the thermal image
    frame0 = align_images(frame0, frame1)

    ## 🟢 VISUAL CAMERA PROCESSING (YUV CONVERSION)
    frame1_yuv = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)

    ## 🔹 HIGH-PASS FILTERING (EDGE DETECTION)
    y_channel = frame1_yuv[:, :, 0]  
    blurred_y = cv2.GaussianBlur(y_channel, (3, 3), 0)
    sobel_x = cv2.Sobel(blurred_y, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_y, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_magnitude = np.uint8(255 * edge_magnitude / np.max(edge_magnitude))

    frame1_yuv_highpass = frame1_yuv.copy()
    frame1_yuv_highpass[:, :, 0] = edge_magnitude

    ## 🔹 LOW-PASS FILTERING (SMOOTHING)
    y_channel_lowpass = cv2.GaussianBlur(frame1_yuv[:, :, 0], (5, 5), 0)
    frame1_yuv_lowpass = frame1_yuv.copy()
    frame1_yuv_lowpass[:, :, 0] = y_channel_lowpass

    ## 🟢 NOIR CAMERA PROCESSING
    gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    ## 🔹 HIGH-PASS FILTERING (EDGE DETECTION)
    y_channel_noir = thermal[:, :, 0]  
    blurred_y_noir = cv2.GaussianBlur(y_channel_noir, (3, 3), 0)
    sobel_x_noir = cv2.Sobel(blurred_y_noir, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y_noir = cv2.Sobel(blurred_y_noir, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude_noir = np.sqrt(sobel_x_noir**2 + sobel_y_noir**2)
    edge_magnitude_noir = np.uint8(255 * edge_magnitude_noir / np.max(edge_magnitude_noir))

    frame0_highpass = thermal.copy()
    frame0_highpass[:, :, 0] = edge_magnitude_noir

    ## 🔹 LOW-PASS FILTERING (SMOOTHING)
    y_channel_lowpass_noir = cv2.GaussianBlur(thermal[:, :, 0], (5, 5), 0)
    frame0_lowpass = thermal.copy()
    frame0_lowpass[:, :, 0] = y_channel_lowpass_noir

    ## 🟢 WEIGHTED FUSION
    denominator = (frame1_yuv_highpass + frame1_yuv_lowpass + frame0_highpass + frame0_lowpass).astype(np.float32)
    denominator[denominator == 0] = 1  # Prevent divide by zero

    wi = (frame1_yuv_highpass + frame1_yuv_lowpass) / denominator
    wt = (frame0_highpass + frame0_lowpass) / denominator

    fused = (wt * thermal) + (wi * frame1)
    fused1 = (wt * thermal) + ((1 - wt) * frame1)

    ## 🟢 DISPLAY RESULTS
    cv2.imshow("fused", np.uint8(fused))
    cv2.imshow("fused1", np.uint8(fused1))

    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
noir_cam.close()
visual_cam.close()
cv2.destroyAllWindows()
