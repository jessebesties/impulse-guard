import sys

from flask import Flask, render_template, Response, send_file, request, jsonify
import cv2
from datetime import datetime, time
import os
from google.cloud import vision
from PIL import Image, ImageDraw

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from webcam.take_save_photo_webcam import take_save_photo_webcam
from gcp_face_recognition.face_detection import detect_and_box_faces

app = Flask(__name__)

# Connect to Client
client = vision.ImageAnnotatorClient()

# Ensure uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')


@app.route('/')
def purchase():
    return render_template('purchase.html')


@app.route('/success')
def success():
    return render_template('success.html')


@app.route('/video_feed')
def video_feed():
    try:
        # return Response(generate_display_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        pass
    except:
        pass


@app.route('/photo_feed')
def capture_photo():
    while True:
        print("Capturing photo...")
        try:
            output_path = take_save_photo_webcam()
            detect_and_box_faces(output_path, 1, 1)
        except Exception as e:
            print("Error: ", e)
        print("Sleeping for 5 seconds...")
        time.sleep(5)


@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    print("Uploading audio...")

    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]

    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    input_path = os.path.join('uploads', audio_file.filename)
    try:
        audio_file.save(input_path)
        print(f"File saved to {input_path}")
        return jsonify({"message": "File uploaded successfully"}), 200
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({"error": "Failed to save file"}), 500
    
# ---------------------------- Helper Functions ---------------------------- #

if __name__ == '__main__':
    # photo_thread = threading.Thread(target=capture_photo, daemon=True)
    # photo_thread.start()

    app.run(debug=True)
