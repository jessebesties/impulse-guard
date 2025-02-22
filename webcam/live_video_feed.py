import cv2

camera = cv2.VideoCapture(0)

def generate_display_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'webcam_feed: image/jpeg\r\n\r\n' + frame + b'\r\n')

