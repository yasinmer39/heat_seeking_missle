from picamera2 import Picamera2
import cv2
import numpy as np

def align_images(img1, img2):
    """Aligns img1 (NoIR) to img2 (Visual) using ORB feature matching & homography."""
    
    # Convert to grayscale for feature detection
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # ORB feature detector
    orb = cv2.ORB_create(200)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Feature matching
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)
    
    if len(matches) < 10:
        print("Not enough matches to align images!")
        return img1  # Return original if alignment fails

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography (perspective transformation)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp img1 (NoIR) to align with img2 (Visual)
    aligned_img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
    
    return aligned_img1

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
    # Capture frames
    frame_noir = noir_cam.capture_array()
    frame_vis = visual_cam.capture_array()

    # Convert Visual camera BGR -> RGB
    frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)

    # Convert NoIR frame to grayscale (intensity-based)
    gray_noir = cv2.cvtColor(frame_noir, cv2.COLOR_BGR2GRAY)

    # Apply false thermal colormap
    thermal_noir = cv2.applyColorMap(gray_noir, cv2.COLORMAP_JET)

    # Align NoIR thermal image with Visual camera
    aligned_noir = align_images(thermal_noir, frame_vis)

    # Compute edge-based weight maps for adaptive fusion
    noir_edges = cv2.Canny(cv2.cvtColor(aligned_noir, cv2.COLOR_BGR2GRAY), 50, 150).astype(float)
    vis_edges = cv2.Canny(cv2.cvtColor(frame_vis, cv2.COLOR_RGB2GRAY), 50, 150).astype(float)

    # Normalize weights
    weight_noir = noir_edges / (noir_edges + vis_edges + 1e-6)
    weight_vis = 1 - weight_noir

    # Perform pixelwise adaptive fusion
    fused_frame = (weight_noir[:, :, None] * aligned_noir + 
                   weight_vis[:, :, None] * frame_vis).astype(np.uint8)

    # Show the result
    cv2.imshow("Pseudo Thermal Camera", aligned_noir)
    cv2.imshow("Visual CAM", frame_vis)
    cv2.imshow("Fused Image", fused_frame)  # Display the fused result

    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
noir_cam.close()
visual_cam.close()
cv2.destroyAllWindows()
