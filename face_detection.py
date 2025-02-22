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

def detect_and_box_faces(input_filename, max_results):
    with open(input_filename, "rb") as image:
        faces = detect_faces(image, max_results)
        image.seek(0)
        highlight_faces(image, faces, "output_" + input_filename)

detect_and_box_faces("asianpeople.png", 999)