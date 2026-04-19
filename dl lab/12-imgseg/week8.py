import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("dog.png")          # put your image path
img = cv2.resize(img, (64,64))
img = img / 255.0

X = np.array([img])

# Dummy mask (for demo)
Y = np.random.randint(0,2,(1,64,64,1))

# Model
model = tf.keras.Sequential([
    layers.Conv2D(8,(3,3),activation='relu',padding='same',input_shape=(64,64,3)),
    layers.Conv2D(1,(1,1),activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, Y, epochs=5, verbose=0)

# Predict mask
pred = model.predict(X)

# Show image + mask
plt.subplot(1,2,1)
plt.title("Input Image")
plt.imshow(img)

plt.subplot(1,2,2)
plt.title("Segmented Mask")
plt.imshow(pred[0,:,:,0], cmap='gray')

plt.show()