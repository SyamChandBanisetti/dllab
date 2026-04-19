import tensorflow as tf
from tensorflow.keras import layers
import cv2
import numpy as np

# Load MNIST and train
(X_train, y_train), _ = tf.keras.datasets.mnist.load_data()
X_train = X_train/255.0

model = tf.keras.Sequential([
    layers.Reshape((28,28,1), input_shape=(28,28)),
    layers.Conv2D(16,(3,3),activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=5, verbose=0)

# 🔹 Load your digit image
img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28,28))
img = img/255.0
img = img.reshape(1,28,28)

# Predict
pred = model.predict(img)
print("Predicted Digit:", np.argmax(pred))