import time
import threading
from flask import Flask, render_template, Response, send_file, request

import cv2
from datetime import datetime
import os
from google.cloud import vision
from PIL import Image, ImageDraw

app = Flask(__name__)

# Connect to Cliend
client = vision.ImageAnnotatorClient()

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
    return client.face_detection(image=image, max_results=max_results)

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
    for i, face in enumerate(faces):
        xs = [vertex.x for vertex in face.bounding_poly.vertices]
        ys = [vertex.y for vertex in face.bounding_poly.vertices]
        left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        cropped_im = im.crop((left, top, right, bottom))
        base_name, _ = os.path.splitext(os.path.basename(image_path))
        cropped_filename = f"{base_name}_cropped_{i}.png"
        cropped_im.save(os.path.join('images-cropped', cropped_filename))
        cropped_images.append(cropped_im)
    return cropped_images

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
        print("Result: ", faces)
        print("Face Annotations!!!: ", faces.face_annotations)
    file_name = input_filename.split('/')[-1]
    highlight_faces(input_filename, faces.face_annotations, file_name)
    if activate_crop:
        crop_faces(input_filename, faces.face_annotations)

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
    
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise IOError("Could not open webcam")

    result, frame = camera.read()

    if result:
        output_path = os.path.join("images", f"{formatted_date_time}webcam.jpg")
        cv2.imwrite(output_path, frame)
        return output_path

    camera.release()
    
def generate_display_frames():
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'webcam_feed: image/jpeg\r\n\r\n' + frame + b'\r\n')


    
# ---------------------------- Helper Functions ---------------------------- #

if __name__ == '__main__':
    photo_thread = threading.Thread(target=capture_photo, daemon=True)
    photo_thread.start()

    app.run(debug=True)
