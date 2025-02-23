import time
import threading
from flask import Flask, render_template, Response, send_file, request, jsonify
import sys

import cv2
from datetime import datetime, time
import os
from google.cloud import vision
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torchvision import models, transforms
import torch.nn as nn

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Connect to Client
client = vision.ImageAnnotatorClient()

camera = cv2.VideoCapture(0)

# Load Model
# Load the model architecture
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
vgg16.classifier = nn.Sequential(
    nn.Linear(25088, 4096),  # VGG16's original FC layer
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),   # VGG16's original FC layer
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 7)      # New output layer for 9 classes
)

# Load the fine-tuned weights
vgg16.load_state_dict(torch.load('vgg16_finetuned_pytorch.pth', map_location=torch.device('cpu')))

vgg16.eval()

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

@app.route('/get_emotion')
def get_emotion():
    emotion, is_happy = detect_and_box_faces(take_save_photo_webcam(), 1, 1)
    return jsonify({
        "emotion": emotion,
        "gcp_emotion": "Happy" if is_happy else "Not happy",
    })

@app.route('/photo_feed')
def capture_photo():
    # while True:
        print("Capturing photo...")
        try:
            detect_and_box_faces(take_save_photo_webcam(), 1, 1)
        except Exception as e:
            print("Error: ", e)
        print("Done...")
            
        # break


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
    
    return jsonify({"audio_emotion": emotion}), 200
    
# ---------------------------- Helper Functions ---------------------------- #

def detect_faces(image_file, max_results=999):
    """
    Detects faces in an image and returns the detected faces as a list of
    google.cloud.vision.Face objects.

    Args:
        image_file: A file object containing the image data.
        max_results: The maximum number of faces to detect. Defaults to 999.

    Returns:
        A list of google.cloud.vision.Face objects, each representing a detected
        face.

    Raises:
        google.cloud.vision.exceptions.GoogleAPICallError: If the API call fails.
    """
    # Read Image content
    content = image_file.read()
    
    # Create image
    image = vision.Image(content=content)
    
    # Detect Faces
    result = client.face_detection(image=image, max_results=max_results)
    
    return result

def highlight_faces(image_path, faces, output_filename):
    """
    Highlights detected faces in an image and saves it to a new image.

    Args:
        image_path: The path to the image file.
        faces: A list of google.cloud.vision.Face objects.
        output_filename: The filename of the output image.

    Returns:
        None
    Raises:
        IOError: If the image could not be opened.
    """
    # Open Image
    im = Image.open(image_path)
    
    # Drawing API
    draw = ImageDraw.Draw(im)
    
    # Draw Faces
    for face in faces:
        box = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill="#00ff00")
        draw.text(
            (
                (face.bounding_poly.vertices)[0].x,
                (face.bounding_poly.vertices)[0].y - 30,
            ),
            str(format(face.detection_confidence, ".3f")) + "%",
            fill="#FF0000",
        )
    im.save(os.path.join('images-drawn', output_filename))
    
def crop_faces(image_path, faces):
    """
    Crop a face from an image and save it to a new image.

    Args:
        image_path (str): The path to the image file.
        faces (list): A list of google.cloud.vision.Face objects.

    Returns:
        list: A list of PIL.Image objects, each representing a cropped face.
        
    Raises:
        IOError: If the image could not be opened.
    """
    # Open Image
    im = Image.open(image_path)
    
    # Crop Faces
    cropped_images = []
    print(faces)
    for i, face in enumerate(faces):
        xs = [vertex.x for vertex in face.bounding_poly.vertices]
        ys = [vertex.y for vertex in face.bounding_poly.vertices]
        left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        cropped_im = im.crop((left, top, right, bottom))
        base_name, _ = os.path.splitext(os.path.basename(image_path))
        cropped_filename = f"{base_name}_cropped_{i}.png"
        cropped_im.save(os.path.join('images-cropped', cropped_filename))
        cropped_images.append(cropped_im)
        break
    return cropped_filename
    # return cropped_images

def detect_and_box_faces(input_filename, max_results, activate_crop):
    """
    Detects faces in an image and draws boxes around them.
    Args:
        input_filename: The filename of the image to process.
        max_results: The maximum number of faces to detect.
        activate_crop: If True, the detected faces will be cropped
            and saved to a new image.

    Returns:
        None
    """
    with open(input_filename, "rb") as image_file:
        faces = detect_faces(image_file, max_results)
        
        happy = (faces.face_annotations[0].joy_likelihood)
        sorrow = faces.face_annotations[0].sorrow_likelihood
        anger = faces.face_annotations[0].anger_likelihood

        print("current happy: ", happy)
        print("current not_happy: ", max(sorrow, anger))
        print("happier? ", happy > max(anger, sorrow))
        is_happy = happy > max(sorrow, anger)
        
    file_name = input_filename.split('/')[-1]
    highlight_faces(input_filename, faces.face_annotations, file_name)
    if activate_crop:
        cropped_filename = crop_faces(input_filename, faces.face_annotations)
        
    predicted_label = pytorch_vgg_inference(cropped_filename)
    
    os.remove(input_filename)

    os.remove(os.path.join('images-drawn', file_name))

    base_name, _ = os.path.splitext(os.path.basename(input_filename))
    os.remove(os.path.join('images-cropped', cropped_filename))
    
    return predicted_label, is_happy
    

def take_save_photo_webcam():
    """
    Take a photo from the webcam and save it to the images folder with a timestamped filename.

    Returns:
        str: The path to the saved image file.

    Raises:
        IOError: If the webcam could not be opened.
    """
    now = datetime.now()

    formatted_date_time = now.strftime("%Y-%m-%d %H_%M_%S")

    if not camera.isOpened():
        raise IOError("Could not open webcam")

    result, frame = camera.read()

    if result:
        output_path = os.path.join("images", f"{formatted_date_time}webcam.jpg")
        cv2.imwrite(output_path, frame)
        return output_path

    camera.release()
    
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
        break

    camera.release()

def pytorch_vgg_inference(file_name):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Inferencing...")

    # Load and preprocess the image
    while True:
        try:
            image = Image.open(f"./images-cropped/{file_name}")
            break
        except Exception as e:
            print("Error: ", e)
    print("Image Loaded!")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    print("Running Inference")

    # Run inference
    with torch.no_grad():
        output = vgg16(input_batch)
        
    print("Converting to probabilities")

    # Convert output to probabilities (optional)
    probabilities = F.softmax(output, dim=1)

    # Get the predicted class
    _, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()

    # Map the predicted class to a label
    class_labels = ["afraid", "angry", "disappointed", "happy", "neutral", "sad", "surprised"]
    predicted_label = class_labels[predicted_class]

    # Print the result
    print(f"Predicted class: {predicted_label}")
    print(f"Probabilities: {probabilities}")
    
    return predicted_label

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

# ---------------------------- Helper Functions ---------------------------- #

if __name__ == '__main__':
    # photo_thread = threading.Thread(target=capture_photo, daemon=True)
    # photo_thread.start()

    app.run(debug=True)
