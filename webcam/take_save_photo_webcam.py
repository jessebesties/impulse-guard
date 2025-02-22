import cv2
from datetime import datetime
import os

def take_save_photo_webcam():
    now = datetime.now()

    formatted_date_time = now.strftime("%Y-%m-%d %H_%M_%S")

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise IOError("Could not open webcam")

    result, frame = camera.read()

    if result:
        output_path = os.path.join("images", f"{formatted_date_time}webcam.jpg")
        cv2.imwrite(output_path, frame)
        return output_path

    camera.release()