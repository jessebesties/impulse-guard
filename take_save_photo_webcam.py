import cv2

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    raise IOError("Couldn't open webcam or video")

ret, frame = video_capture.read()

if not ret:
    raise IOError("Couldn't capture frame")

cv2.imshow("captured_image.jpg", frame)