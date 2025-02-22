import cv2
from datetime import datetime

now = datetime.now()

formatted_date_time = now.strftime("%Y-%m-%d %H_%M_%S")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise IOError("Could not open webcam")

result, frame = camera.read()

if result:
    cv2.imwrite(formatted_date_time + "webcam.jpg", frame)

camera.release()

