from flask import Flask, render_template, request, redirect, url_for
import face_recognition
import cv2
import os
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
DATABASE_FOLDER = 'database/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Load known faces
def load_known_faces():
    faces_dict = {}
    for filename in os.listdir(DATABASE_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            person_name = os.path.splitext(filename)[0]
            file_path = os.path.join(DATABASE_FOLDER, filename)
            image = face_recognition.load_image_file(file_path)
            try:
                encoding = face_recognition.face_encodings(image)[0]
                faces_dict[person_name] = encoding
            except IndexError:
                print(f'Face not detected in {filename}')
    return faces_dict


known_faces = load_known_faces()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Perform face recognition
    image = face_recognition.load_image_file(file_path)
    face_encodings = face_recognition.face_encodings(image)
    recognized_faces = []

    if face_encodings:
        for face_encoding in face_encodings:
            results = {name: face_recognition.compare_faces([encoding], face_encoding)[0]
                       for name, encoding in known_faces.items()}
            matched_names = [name for name, match in results.items() if match]
            recognized_faces.extend(matched_names)

    return render_template('result.html', faces=recognized_faces, image_url=file_path)


if __name__ == '__main__':
    app.run(debug=True)
