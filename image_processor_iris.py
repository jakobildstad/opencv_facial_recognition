import cv2
import os
import numpy as np

# Haar cascade for face detection again
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
#Thinking i should first isolate the face so that i dont pick up any random circles in the image.

#trying it first with a testing img
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

#doing some processing. Not sure how much I should, because of color and details being important.
gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
gray_roi = cv2.medianBlur(gray_roi, 5)

#Detect circles
circles = cv2.HoughCircles(
    gray_roi,
    cv2.HOUGH_GRADIENT,
    dp=1,  # The inverse ratio of resolution
    minDist=gray_roi.shape[0] / 8,  # Minimum distance between detected circles
    param1=50,  # Higher threshold for the Canny edge detector
    param2=30,  # Threshold for center detection in HoughCircles
    minRadius=20,  # Minimum radius to consider (tune this)
    maxRadius=60   # Maximum radius to consider (tune this)
)

#This is really for displaying in camera
"""
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        center_x, center_y, radius = circle
        # Draw the outer circle on the face ROI for visualization
        cv2.circle(face_roi, (center_x, center_y), radius, (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(face_roi, (center_x, center_y), 2, (0, 0, 255), 3)
    print("Iris detected.")
else:
    print("No iris detected.")"
"""