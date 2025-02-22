from webcam import take_save_photo_webcam
from gcp_face_recognition import face_detection

webcam_output = take_save_photo_webcam.take_save_photo_webcam()
face_detection.detect_and_box_faces(webcam_output, 1, 1)