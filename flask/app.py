from flask import Flask, render_template, Response
from webcam import live_video_feed

app = Flask(__name__)

@app.route('/')
def purchase():
    return render_template('purchase.html')

@app.route('/success')
def success():
    return render_template('success.html')

@app.route('/video_feed')
def video_feed():
    return Response(live_video_feed.generate_display_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
