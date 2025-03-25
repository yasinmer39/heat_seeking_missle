import numpy as np
import cv2
import glob

# Chessboard settings
chessboard_size = (9, 6)  # Change this based on your chessboard
square_size = 2.5  # Size of a square in cm or mm (does not matter for ratios)

# Prepare object points (3D world coordinates)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Lists to store points
objpoints = []  # 3D points
imgpoints_left = []  # 2D points from left camera
imgpoints_right = []  # 2D points from right camera

# Load images
left_images = sorted(glob.glob("captured_images_left/*.jpg"))  # Adjust extension if needed
right_images = sorted(glob.glob("captured_images_right/*.jpg"))

if not left_images or not right_images:
    print("Error: No images found in the dataset.")
    exit()

if len(left_images) != len(right_images):
    print("Error: Mismatch in the number of left and right images.")
    exit()

# Read first image to get image size
imgL = cv2.imread(left_images[0])
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
image_size = grayL.shape[::-1]  # (width, height)

# Process each pair of images
for left_img_path, right_img_path in zip(left_images, right_images):
    imgL = cv2.imread(left_img_path)
    imgR = cv2.imread(right_img_path)
    
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)

    if retL and retR:
        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)
    else:
        print(f"Chessboard not found in {left_img_path} or {right_img_path}")

# Ensure there are enough points
if len(objpoints) == 0:
    print("Error: No valid points for calibration found.")
    exit()

# Run camera calibration
print("Calibrating left camera...")
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, image_size, None, None)
print("Calibrating right camera...")
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, image_size, None, None)

# Save calibration results
np.savez("stereo_calibration_data.npz", 
         mtxL=mtxL, distL=distL, 
         mtxR=mtxR, distR=distR)

print("Stereo calibration completed successfully. Data saved to 'stereo_calibration_data.npz'.")
