import cv2
import numpy as np
import os

# Path to your processed training data
data_path = "/Users/jakobildstad/Documents/Projects/opencv_facial_recognition/processed_training_data"

# Lists to hold face samples and corresponding labels
face_samples = []
labels = []

# For each person in your dataset, assume each person has their own folder.
label_id = 0
label_dict = {}  # Map label IDs to person names

for person_name in os.listdir(data_path):
    person_folder = os.path.join(data_path, person_name)
    if not os.path.isdir(person_folder):
        continue
    label_dict[label_id] = person_name
    for image_name in os.listdir(person_folder):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(person_folder, image_name)
            # Read image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            face_samples.append(img)
            labels.append(label_id)
    label_id += 1

# Convert labels list to NumPy array
labels = np.array(labels)

# Create the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Train the recognizer on the face samples and labels
recognizer.train(face_samples, labels)

# Optionally, save the trained model for later use
recognizer.save("trained_face_model.yml")
print("Training complete!")