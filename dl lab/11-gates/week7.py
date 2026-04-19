import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Input
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

# Output → [AND, OR, XOR]
Y = np.array([
    [0,0,0],
    [0,1,1],
    [0,1,1],
    [1,1,0]
])

# Model
model = tf.keras.Sequential([
    layers.Dense(6, activation='relu', input_shape=(2,)),
    layers.Dense(6, activation='relu'),
    layers.Dense(3, activation='sigmoid')  # 3 outputs
])

model.compile(optimizer='adam', loss='binary_crossentropy')

# Train
model.fit(X, Y, epochs=200, verbose=0)

# Test
pred = model.predict(np.array([[1,0]]))
print("AND, OR, XOR:", pred.round())