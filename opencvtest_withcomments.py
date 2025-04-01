import numpy as np
import cv2


# Load Haar cascade, which is a pre-trained classifier. 
# A pre-trainer classifier is basically a type of pre-trained machine learning model,
# which in this instance is used to recognize faces in general. 
# face_cascade is assigned this class and becomes an object with its own methods.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# capture becomes a object of the class VideoCapture from the cv2 library. 
# it talks to the pc and establishes a connection to the camera. 
capture = cv2.VideoCapture(0)

# capture.isOpened() returns true if cv2.VideoCapture(0) 
# successfully established a connection to the Pc's camera.
if not capture.isOpened():
    print("Error: Could not open camera.")
    exit()

# An infinite loop, which is needed to display a continuous video. 
while True:
    ret, frame = capture.read() #.read() reads the current frame of the camera.
    # ret (return) is a boolean value symbolizing whether a frame is successfully returned.
    # frame is a n x m x 3 array (standard image array) that includes all information about the frame at that instance. 


    # Exits the camera if a frame is missed, probably because a frame missed means something is wrong.
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # defines a gray-scale version of frame using the cvtColor method.
    # This is a normalizing method making it easier for the face detection algorithm to work on it.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # faces is an array of tuples (x, y, w, h) where x, y is position and w, h is width and height.
    # this uses the cascade to give faces in the captured frame.
    # "multiscale" means the faces can be different sizes
    # argument gray tells it that the frame is grayscale
    # scaleFactor: a smaller scale factor makes the det. slower but more accurate.
    # minNeighbors: increasing reduces false positives (FAR: False acceptence rate)
    # minsize: tells the minimum size of the faces. the bigger the faster.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # draws rectangle(s) around the faces.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # creates or updates window that displays the resulting video.
    cv2.imshow('Camera Feed', frame)

    # looks for button press, which breaks the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# closes camera and closes windows. 
capture.release()
cv2.destroyAllWindows()