import numpy as np
import cv2

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the LBPH face recognizer and your pre-trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_face_model.yml")  # Ensure this path is correct

# Set a threshold to map the confidence (distance) to a certainty score (this value may need tuning)
threshold = 100.0

# Open the video capture (default camera)
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around each detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the region of interest (the face) from the grayscale image
        face_roi = gray[y:y+h, x:x+w]

        # Optionally, resize the face ROI if your training data images were of a specific size:
        # face_roi = cv2.resize(face_roi, (desired_width, desired_height))

        # Use the recognizer to predict the label and obtain a confidence value.
        # Note: With LBPH, a lower confidence value means a better match.
        label, confidence = recognizer.predict(face_roi)

        # Map the confidence to a certainty score between 0 and 1.
        # Here we assume that a confidence of 0 corresponds to a perfect match (certainty 1.0)
        # and a confidence equal to or above the threshold corresponds to certainty 0.
        certainty = max(0, min(1, 1 - confidence / threshold))

        # Assuming label 0 is Jakob, adjust if you have more labels.
        if label == 0:
            text = "Jakob: {:.2f}".format(certainty)
        else:
            text = "Unknown"

        # Display the text below the face rectangle
        cv2.putText(frame, text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # Display the video feed with the drawn rectangles and text
    cv2.imshow('Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()