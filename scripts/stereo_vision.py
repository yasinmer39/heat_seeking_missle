import cv2
import numpy as np

# Camera parameters from calibration (replace with your values)
focal_length = 3.04  # Example value in mm, update after calibration
baseline = 3.0       # Baseline distance in cm

# Load left and right images
left_img = cv2.imread("left.jpg", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread("right.jpg", cv2.IMREAD_GRAYSCALE)

if left_img is None or right_img is None:
    print("Error: Could not load images. Check file paths.")
    exit()

# StereoSGBM settings (tuned for Raspberry Pi Camera)
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,  # Increase this (must be a multiple of 16)
    blockSize=9,         # Try reducing (e.g., 5, 7, 9)
    P1=8 * 3 * 9 ** 2,
    P2=32 * 3 * 9 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=5,    # Lower this (e.g., 5-10)
    speckleWindowSize=50, # Reduce this
    speckleRange=32
)


disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
disparity[disparity <= 0] = 0.1  # Avoid division by zero

print("Disparity min:", disparity.min(), "Disparity max:", disparity.max())

disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

cv2.imshow("Disparity Map", disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
