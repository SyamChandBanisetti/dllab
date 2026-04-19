import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2

# load image
img = cv2.imread("dog.png")
img = cv2.resize(img, (64,64))
img = img / 255.0
img = img.reshape(1, -1)   # flatten

# word mapping
words = ["<pad>","a","dog","is","running"]

# caption input
cap = np.array([[1,2,3]])  # "a dog is"

# model
i = layers.Input((64*64*3,))
c = layers.Input((3,))
x1 = layers.Dense(32, activation='relu')(i)
x2 = layers.LSTM(32)(layers.Embedding(5,32)(c))
out = layers.Dense(5, activation='softmax')(layers.add([x1,x2]))

model = tf.keras.Model([i,c], out)
model.compile('adam','sparse_categorical_crossentropy')

# dummy training
y = np.array([4])
model.fit([img,cap], y, epochs=3, verbose=0)

# predict
p = model.predict([img,cap])
print("Caption: a dog is", words[np.argmax(p)])