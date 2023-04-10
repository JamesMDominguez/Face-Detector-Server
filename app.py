import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64

app = Flask(__name__)
cors = CORS(app)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/', methods=['POST'])
def detect_faces():
    # get the image from the request
    print("this endpoint has been used-----------------------------------------------")
    
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    #img2 = cv2.imread('face.jpg')

    # detect faces in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # loop through each face and save it as a separate image
    face_images = []
    for (x, y, w, h) in faces:
        new_w = int(w * 1.7)
        new_h = int(h * 1.7)

    # draw rectangle around the face with the new coordinates and size
        face_img = img[y-50:y+new_h, x-70:x+new_w]
        face_images.append(face_img)
    # convert each face image to bytes and store them in a list
    face_bytes = []
    for face_img in face_images:
        x, buffer = cv2.imencode('.jpg', face_img)
        face_bytes.append(buffer.tobytes())
    
    # return the face images as a JSON response
    images_b64 = [base64.b64encode(img).decode('utf-8') for img in face_bytes]

    # Create JSON response with image data
    response_data = {'images': images_b64}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
