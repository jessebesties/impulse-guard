from PIL import Image, ImageDraw
from google.cloud import vision
import os

output_dir = "output_images"
input_dir = "input_images"

os.makedirs(output_dir, exist_ok=True)

def detect_faces(image_file, max_results=999):
    client = vision.ImageAnnotatorClient()
    content = image_file.read()
    image = vision.Image(content=content)
    return client.face_detection(image=image, max_results=max_results).face_annotations

def highlight_faces(image_path, faces, output_filename):
    im = Image.open(image_path)
    draw = ImageDraw.Draw(im)
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
    im.save(os.path.join(output_dir, output_filename))

def detect_and_box_faces(input_filename, max_results, activate_crop):
    image_path = os.path.join(input_dir, input_filename)
    with open(image_path, "rb") as image_file:
        faces = detect_faces(image_file, max_results)
    highlight_faces(image_path, faces, "output_" + input_filename)
    if activate_crop:
        crop_faces(image_path, faces)

def crop_faces(image_path, faces):
    im = Image.open(image_path)
    cropped_images = []
    for i, face in enumerate(faces):
        xs = [vertex.x for vertex in face.bounding_poly.vertices]
        ys = [vertex.y for vertex in face.bounding_poly.vertices]
        left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
        cropped_im = im.crop((left, top, right, bottom))
        base_name, _ = os.path.splitext(os.path.basename(image_path))
        cropped_filename = f"{base_name}_cropped_{i}.png"
        cropped_im.save(os.path.join(output_dir, cropped_filename))
        cropped_images.append(cropped_im)
    return cropped_images
