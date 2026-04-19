import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("study_hours_marks.csv")

X = data[['Hours']]
Y = data['Marks']

# Train model
model = LinearRegression()
model.fit(X, Y)

# Predict marks for 6 hours
Hours=np.array([[6]])
# prediction = model.predict(pd.DataFrame({'Hours': [6]}))
prediction = model.predict(Hours)
print("Predicted Marks for 6 hours:", prediction[0])

# Plot graph
plt.scatter(X, Y)
plt.plot(X, model.predict(X))
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks (Linear Regression)")
plt.show()
