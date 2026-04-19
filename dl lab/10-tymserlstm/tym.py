import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Sample time series data
data = [100,110,120,130,140,150,160,170,180,190]
data = np.array(data).reshape(-1,1)

# Normalize data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Create dataset (X, y)
X = []
y = []
for i in range(len(data)-3):
    X.append(data[i:i+3])
    y.append(data[i+3])

X = np.array(X)
y = np.array(y)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3,1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=200, verbose=0)

# Predict next value
last_input = data[-3:]
last_input = last_input.reshape((1,3,1))

prediction = model.predict(last_input)
prediction = scaler.inverse_transform(prediction)

print("Next predicted value:", prediction[0][0])