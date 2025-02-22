from webcam import take_save_photo_webcam
from gcp_face_recognition import face_detection

def take_picture_webcam():
    webcam_output = take_save_photo_webcam.take_save_photo_webcam()
    if webcam_output:
        face_detection.detect_and_box_faces(webcam_output, 1, 1)

face_detection.detect_and_box_faces("img.jpeg", 1, 1)