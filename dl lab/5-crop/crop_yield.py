import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load dataset
data = pd.read_csv("crop_data.csv")

# 2. Input features
X = data[['rainfall', 'fertilizer', 'soil_quality']]

# 3. Target variable
y = data['yield']

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict yield
y_pred = model.predict(X_test)

# 7. Print some predictions
print("Actual vs Predicted Yield:\n")
for i in range(5):
    print("Actual:", y_test.iloc[i], 
          "Predicted:", round(y_pred[i], 2))

# 8. Predict yield for new input
new_crop = [[800, 120, 4]]  # rainfall, fertilizer, soil quality
predicted_yield = model.predict(new_crop)

print("\nPredicted Crop Yield:", round(predicted_yield[0], 2), "tons/hectare")
