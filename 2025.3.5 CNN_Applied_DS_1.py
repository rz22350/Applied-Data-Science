import matplotlib

# 设置后端，防止 PyCharm 下 plt.show() 错误
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

# 设置随机种子，保证实验可复现
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# 数据文件夹路径（根据实际情况修改）
path_test = "uob_image_set"
CATEGORIES = os.listdir(path_test)
img_size = 256


# 定义移除图片背景的函数（假设目标为最大轮廓）
def remove_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用 Otsu 阈值法获得二值图像（假设背景较亮）
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


# 生成训练数据（图片及对应标签）
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
            # 移除背景
            img_nobg = remove_background(img_array)
            new_array = cv2.resize(img_nobg, (img_size, img_size))
            training.append([new_array, class_num])


createTrainingData()
random.shuffle(training)

X = []  # 特征数据
y = []  # 标签
for features, label in training:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 3)
y = np.array(y).astype('int')

# 数据归一化
X = X.astype('float32') / 255.0

# 划分数据集：60%训练，20%验证，20%测试
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 数据增强层（仅在训练时生效）
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# 定义模型，包含数据增强层
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
# 输出模型结构
model.summary()

# 设置训练参数
batch_size = 16
epochs = 50

# 定义 EarlyStopping 和模型保存回调（保存格式改为 .keras）
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_loss', save_best_only=True)
callbacks = [early_stop, checkpoint]

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 模型训练（包含验证集和回调）
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks)

# 在测试集上评估模型
score = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

# 绘制训练过程中的准确率和损失曲线
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# 准确率曲线
ax[0].plot(history.history['accuracy'], label='Train Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[0].set_title('Model Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend()
# 损失曲线
ax[1].plot(history.history['loss'], label='Train Loss')
ax[1].plot(history.history['val_loss'], label='Validation Loss')
ax[1].set_title('Model Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend()
plt.tight_layout()
plt.show()

# 对测试集进行预测，计算混淆矩阵和分类报告
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=CATEGORIES))

# 绘制混淆矩阵图
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(CATEGORIES))
plt.xticks(tick_marks, CATEGORIES, rotation=45)
plt.yticks(tick_marks, CATEGORIES)
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
