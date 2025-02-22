from PIL import Image, ImageDraw

def detect_faces(source_file, max_results=999):
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    content = source_file.read()
    image = vision.Image(content=content)

    return client.face_detection(image=image, max_results=max_results).face_annotations

def highlight_faces(image, faces, output_filename):
    im = Image.open(image)
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
    im.save(output_filename)