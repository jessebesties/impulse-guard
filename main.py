import take_save_photo_webcam
from gcp_face_recognition import face_detection

filename = take_save_photo_webcam.take_save_photo_webcam()
face_detection.detect_and_box_faces(filename, 1, 1)