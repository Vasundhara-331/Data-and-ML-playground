# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Scores": [35, 40, 50, 55, 65, 70, 75, 85, 90, 95]
}

df = pd.DataFrame(data)
print(df)

# Step 3: Plot graph
plt.scatter(df["Hours"], df["Scores"], color='blue')
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Hours vs Scores")
plt.show()

# Step 4: Split data
X = df[["Hours"]]   # input (study hours)
y = df["Scores"]    # output (marks)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:")
print(X_train, y_train)
print("\nTesting data:")
print(X_test, y_test)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")

# Step 6: Predictions
y_pred = model.predict(X_test)

print("Predicted scores:", y_pred)
print("Actual scores:", list(y_test))

# Step 7: Plot regression line
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Hours vs Scores (with Regression Line)")
plt.legend()
plt.show()

# Step 8: Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score (accuracy):", r2)

# Step 9: Predict for new value
hours = [[7.5]]  # e.g., student studies 7.5 hours
predicted_score = model.predict(hours)
print(f"Predicted Score for 7.5 hours: {predicted_score[0]:.2f}")
