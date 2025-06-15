import cv2
import os
import numpy as np

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare training data
dataset_path = "dataset"

faces, labels, label_names = [], [], []
for label, person in enumerate(os.listdir("dataset")):
    label_names.append(person)
    for img_name in os.listdir(f"dataset/{person}"):
        img = cv2.imread(f"dataset/{person}/{img_name}", cv2.IMREAD_GRAYSCALE)
        if img is not None:
            faces.append(cv2.resize(img, (160, 160)))
            labels.append(label)

faces_array = np.array(faces)
labels_array = np.array(labels)

# Train the recognizer
recognizer.train(faces_array, labels_array)

# Load test image
test_img_path = "test.jpg"
test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)

if test_img is None:
    print("❌ Error: Could not read test image. Check path and file name.")
    exit()

test_img_resized = cv2.resize(test_img, (160, 160))

# Predict
label, confidence = recognizer.predict(test_img_resized)
predicted_name = label_names[label]

print(f"✅ Predicted: {predicted_name} with confidence {confidence}")
