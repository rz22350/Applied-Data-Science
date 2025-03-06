import matplotlib

# Set backend to avoid plt.show() errors in PyCharm
matplotlib.use('TkAgg')

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from sklearn.metrics import confusion_matrix, classification_report

# Set random seeds to ensure experiment reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Data folder path (modify according to your setup)
path_test = "uob_image_set"
# Only consider directories as categories
CATEGORIES = [d for d in os.listdir(path_test) if os.path.isdir(os.path.join(path_test, d))]
img_size = 256

# Define a function to remove the background of an image (assumes the target is the largest contour)
def remove_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use Otsu thresholding to obtain a binary image (assuming a bright background)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c], -1, 255, -1)
        result = cv2.bitwise_and(image, image, mask=mask)
        return result
    else:
        return image

# Generate training data (images and corresponding labels)
training = []

def createTrainingData():
    for category in CATEGORIES:
        path = os.path.join(path_test, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path)
            if img_array is None:
                continue
            # Remove background
            img_nobg = remove_background(img_array)
            new_array = cv2.resize(img_nobg, (img_size, img_size))
            training.append([new_array, class_num])

createTrainingData()
random.shuffle(training)

X = []  # Feature data
y = []  # Labels
for features, label in training:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 3)
y = np.array(y).astype('int')

# Data normalization
X = X.astype('float32') / 255.0

# Split dataset: 60% training, 20% validation, 20% testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Data augmentation layer (active only during training)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Define the model, including the data augmentation layer
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(img_size, img_size, 3)),
    data_augmentation,
    tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(CATEGORIES), activation='softmax')
])
# Output model summary
model.summary()

# Set training parameters
batch_size = 16
epochs = 200

# Define EarlyStopping and model checkpoint callbacks (saving format changed to .keras)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_loss', save_best_only=True)
callbacks = [early_stop, checkpoint]

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model training (including validation set and callbacks)
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks)

# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

# Predict on the test set and compute confusion matrix and classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Use only the unique labels that appear in the test set to generate the classification report
unique_labels = np.unique(y_test)
target_names = [CATEGORIES[i] for i in unique_labels]
print("Classification Report:")
print(classification_report(y_test, y_pred_classes, labels=unique_labels, target_names=target_names))

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(unique_labels))
plt.xticks(tick_marks, [CATEGORIES[i] for i in unique_labels], rotation=45)
plt.yticks(tick_marks, [CATEGORIES[i] for i in unique_labels])
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
