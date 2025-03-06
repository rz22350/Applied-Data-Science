import matplotlib
matplotlib.use('TkAgg')

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from sklearn.metrics import confusion_matrix, classification_report

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Data folder path (modify according to your setup)
path_test = "uob_image_set"
# Only consider directories as categories
CATEGORIES = [d for d in os.listdir(path_test) if os.path.isdir(os.path.join(path_test, d))]
img_size = 224  # MobileNetV2 默认输入尺寸为 224x224

def remove_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

# Generate data: images and corresponding labels
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
            img_nobg = remove_background(img_array)
            # Resize to match MobileNetV2 input size
            new_array = cv2.resize(img_nobg, (img_size, img_size))
            training.append([new_array, class_num])

createTrainingData()
random.shuffle(training)

X = []
y = []
for features, label in training:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 3)
y = np.array(y).astype('int')

# Normalize data (MobileNetV2 推荐预处理方法：归一化到 [-1,1])
X = X.astype('float32')
X = (X / 127.5) - 1.0

# Split dataset: 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Load MobileNetV2 base model with pre-trained ImageNet weights
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
base_model.trainable = False  # 冻结预训练层

# 添加自定义分类头
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(CATEGORIES), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

# Training parameters
batch_size = 16
epochs = 50

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_loss', save_best_only=True)
callbacks = [early_stop, checkpoint]

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)

unique_labels = np.unique(y_test)
target_names = [CATEGORIES[i] for i in unique_labels]
print("Classification Report:")
print(classification_report(y_test, y_pred_classes, labels=unique_labels, target_names=target_names))

# Plot confusion matrix
import matplotlib.pyplot as plt
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
