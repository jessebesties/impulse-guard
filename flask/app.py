import time
import threading
from flask import Flask, render_template, Response, send_file

from gcp_face_recognition.face_detection import detect_and_box_faces
from webcam import live_video_feed, take_save_photo_webcam
from gcp_face_recognition import face_detection

app = Flask(__name__)

@app.route('/')
def purchase():
    return render_template('purchase.html')

@app.route('/success')
def success():
    return render_template('success.html')

@app.route('/video_feed')
def video_feed():
    try:
        return Response(live_video_feed.generate_display_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except:
        pass

@app.route('/photo_feed')
#fix this, then add the feed
def capture_photo():
    while True:
        try:
            return detect_and_box_faces(take_save_photo_webcam.take_save_photo_webcam(), 1, 1)
        except:
            pass
        time.sleep(2)

if __name__ == '__main__':
    photo_thread = threading.Thread(target=capture_photo, daemon=True)
    photo_thread.start()

    app.run(debug=True)
