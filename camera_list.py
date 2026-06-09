import cv2

for i in range(10):  # Check IDs 0-9
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at ID: {i}")
        cap.release()
