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

import librosa
import numpy as np
import tensorflow as tf
from scipy import signal # Import signal
import resampy

# Load the model
model = tf.keras.models.model_from_json(open('model.json', 'r').read())
model.load_weights('model.h5')

# Parameters for spectrogram calculation (adjust as needed)
nperseg = 512
noverlap = 256


import subprocess

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
    if "audio" not in request.files:
        print("No audio file in request")
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]

    if audio_file.filename == '':
        print("No file selected")
        return jsonify({"error": "No selected file"}), 400

    file_name = "recording_" + datetime.now().strftime("%Y%m%d%H%M%S")
    input_path = os.path.join('uploads', f"{file_name}.webp")
    
    try:
        audio_file.save(input_path)
        print(f"File saved to {input_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({"error": "Failed to save file"}), 500
    
    subprocess.run(["ffmpeg", "-i", input_path, "-vn", "-acodec", "pcm_s16le", os.path.join('uploads', f"{file_name}.wav")])
    
    os.remove(os.path.join('uploads', file_name + ".webp"))
    
    emotion = predict_emotion(os.path.join('uploads', f"{file_name}.wav"))
    print(emotion)
    
    return jsonify({"audio-emotion": emotion}), 200
    
# ---------------------------- Helper Functions ---------------------------- #

def extract_feature(file_name):
    """Extract MFCC features from an audio file."""
    try:
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=3, sr=22050 * 2, offset=0.5)
        sample_rate = np.array(sample_rate)

        # Calculate spectrogram
        freqs, times, spectrogram = signal.spectrogram(
            X,
            fs=sample_rate,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False
        )

        # Standardize spectrogram (if you did this in Colab)
        mean = np.mean(spectrogram, axis=0)
        std = np.std(spectrogram, axis=0)
        spectrogram = (spectrogram - mean) / std

        # Calculate MFCCs
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)

        # If shape of mfccs is not (259,), pad or trim it to match
        if mfccs.shape != (259,):
            if mfccs.shape[0] < 259:
                # Pad with zeros if smaller
                mfccs = np.pad(mfccs, (0, 259 - mfccs.shape[0]), 'constant')
            else:
                # Trim if larger
                mfccs = mfccs[:259]

        feature = mfccs
        return feature
    except Exception as e:
        print("Error encountered while parsing file: ", file_name, e)
        return None

def predict_emotion(audio_path):
    """Predict emotion using your pre-trained model."""
    feature = extract_feature(audio_path)
    if feature is not None:
        feature = feature.reshape(1, -1)  # Reshape for model input
        feature = np.expand_dims(feature, axis=2)
        prediction = model.predict(feature)
        predicted_class = np.argmax(prediction)
        emotion_labels = ['angry', 'calm', 'fearful', 'happy', 'sad']  # Adjust labels
        predicted_emotion = emotion_labels[predicted_class]
        return predicted_emotion
    else:
        return None

if __name__ == '__main__':
    # photo_thread = threading.Thread(target=capture_photo, daemon=True)
    # photo_thread.start()

    app.run(debug=True)
