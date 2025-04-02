import cv2
import os

# Haar cascade for face detection again
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
#Thinking i should first isolate the face so that i dont pick up any random circles in the image.

raw_image_path = "/Users/jakobildstad/Documents/Projects/opencv_facial_recognition/misc/raw_test_img.png"
image = cv2.imread(raw_image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.11,
            minNeighbors=6,
            minSize=(200, 200)
        )

x, y, w, h = faces[0]

face_roi = image[y:y+h, x:x+w]
# ok so now we have the face.

