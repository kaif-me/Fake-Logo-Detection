import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

# Load Dataset
def load_data(data_dir):
    images, labels = [], []
    for label, category in enumerate(["real", "fake"]):
        category_path = os.path.join(data_dir, category)
        for image_file in os.listdir(category_path):
            img = cv2.imread(os.path.join(category_path, image_file))
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)

    return np.array(images) / 255.0, np.array(labels)

# Load Data
X, y = load_data("dataset")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save Model
model.save("app/model/fake_logo_model.h5")
