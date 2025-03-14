import cv2
import numpy as np

# Initialize computer camera
cap = cv2.VideoCapture(0)

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30)

class KalmanFilter:
    def __init__(self):
        # Initialize the Kalman Filter
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1., 0., 1., 0.],
                                                 [0., 1., 0., 1.],
                                                 [0., 0., 1., 0.],
                                                 [0., 0., 0., 1.]], np.float32)
        self.kalman.measurementMatrix = 1. * np.eye(2, 4, dtype=np.float32)
        self.kalman.processNoiseCov = 1e-5 * np.eye(4, 4, dtype=np.float32)
        self.kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2, dtype=np.float32)
        self.kalman.errorCovPost = 1. * np.ones((4, 4), dtype=np.float32)
        self.kalman.statePost = 0.1 * np.random.randn(4, 1).astype(np.float32)

    def predict(self, measurement):
        # Predict the new state
        prediction = self.kalman.predict()
        self.kalman.correct(measurement)
        return prediction

class MeanShiftTracker:
    def __init__(self):
        # Initialize the Mean Shift Tracker
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.track_window = None

    def track(self, frame, roi_hist):
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Track the object using the Mean Shift algorithm
        back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, self.track_window = cv2.meanShift(back_proj, self.track_window, self.term_crit)
        return self.track_window
    
mean_shift_tracker = MeanShiftTracker()

kf = KalmanFilter()

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale (intensity-based)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a false thermal colormap (e.g., COLORMAP_JET)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Object detection
    mask = object_detector.apply(thermal)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            measurement = np.array([[np.float32(x + w / 2)], [np.float32(y + h / 2)]], dtype=np.float32)
            prediction = kf.predict(measurement)
            pred_x, pred_y = int(prediction[0]), int(prediction[1])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, 'Object', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.circle(frame, (pred_x, pred_y), 5, (0, 0, 255), -1)  # Draw the predicted point

            # Initialize the Mean Shift Tracker with the first detected object
            if mean_shift_tracker.track_window is None:
                mean_shift_tracker.track_window = (x, y, w, h)
                roi = frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            else:
                mean_shift_tracker.track_window = (x, y, w, h)
                track_window = mean_shift_tracker.track(frame, roi_hist)
                # Draw the tracking window
                x, y, w, h = track_window
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(frame, 'Mean Shift', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Frame', frame)
    
    # Show the result
    cv2.imshow("Pseudo Thermal Camera", thermal)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

