import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load dataset
data = pd.read_csv("fuel_data.csv")

# 2. Select features and target
X = data[['weight', 'speed', 'engine_size']]
y = data['mileage']

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict mileage for test data
y_pred = model.predict(X_test)

# 6. Print actual vs predicted values
print("Actual vs Predicted Mileage:\n")
for i in range(5):
    print(f"Actual: {y_test.iloc[i]}  |  Predicted: {round(y_pred[i], 2)}")

# 7. Predict mileage for a NEW car
new_car = [[1800, 80, 2.0]]   # weight, speed, engine size
predicted_mileage = model.predict(new_car)

print("\nPredicted mileage for new car:", round(predicted_mileage[0], 2))
