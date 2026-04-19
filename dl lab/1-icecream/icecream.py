import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset (FIXED PATH)
data = pd.read_csv(
   "IceCreamData.csv"
)

# Independent and dependent variables (FIXED COLUMN)
X = data[['Temperature']]
Y = data['Revenue']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict for 30°C
temperature = np.array([[30]])
predicted_revenue = model.predict(temperature)

print("Predicted Revenue at 30°C:", predicted_revenue[0])

# Plot regression line
X_sorted = X.sort_values(by='Temperature')
plt.scatter(X, Y)
plt.plot(X_sorted, model.predict(X_sorted))
plt.xlabel("Temperature (°C)")
plt.ylabel("Revenue")
plt.title("Ice Cream Revenue Prediction using Linear Regression")
plt.show()
