# pseudo:
#
# for img in raw_training_data_jakob (folder)
#       make grayscale
#       detect face, cut it out (is this right?)
#       normalize (Can you explain this a bit more?)
#       add the now processed image to the folder processed_training_data_jakob

import cv2
import os

# Path to the Haar cascade for face detection.
# Adjust this path if needed or use another cascade.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

name = "jakob"

# Folders
raw_folder = "/Users/jakobildstad/Documents/Projects/opencv_facial_recognition_private/raw_training_data/" + name
processed_folder = "/Users/jakobildstad/Documents/Projects/opencv_facial_recognition_private/processed_training_data/" + name


# Iterate through each file in the raw_folder
for filename in os.listdir(raw_folder):
    # Check if the file is an image (basic check)

    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Build the full path to the image
        raw_image_path = os.path.join(raw_folder, filename)
        processed_image_path = os.path.join(processed_folder, "processed_" + filename)

        # Read the image
        image = cv2.imread(raw_image_path) #gives m x n x 3 array cool function
        if image is None:
            print(f"Could not read {raw_image_path}, skipping.")
            continue

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces. Same function as in opencvtest.py
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.11,
            minNeighbors=6,
            minSize=(200, 200)
        )

        # If no faces are found, you can skip or handle it differently
        if len(faces) == 0:
            print(f"No face detected in {filename}, skipping.")
            continue

        # For simplicity, let's just process the first detected face
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w] #extracts region from the array gray where the face is

        # Normalization: 
        #    - Here we use histogram equalization to improve contrast.
        #    - Another common approach is scaling pixel values to [0, 1].
        # ok so basically this makes a histogram of the "insensity" (0-255) of each pixel.
        # if the histogram is really concetrated, the image has low contrast, and naturally, the face will be harder to detect.
        # The equalizeHist methot "normalizes" the histogram, making sure the contrast is good enough
        # to get predictable results. 
        face_roi_normalized = cv2.equalizeHist(face_roi)

        # Save the processed face
        cv2.imwrite(processed_image_path, face_roi_normalized)
        print(f"Processed face saved to {processed_image_path}")