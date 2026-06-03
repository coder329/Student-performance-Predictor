import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1️⃣ Create Dataset
# -------------------------------

data = {
    "study_hours": [2, 3, 5, 7, 1, 4, 6, 8],
    "sleep_hours": [7, 6, 8, 5, 9, 6, 7, 4],
    "attendance": [60, 70, 80, 90, 50, 75, 85, 95],
    "marks": [50, 55, 70, 85, 40, 65, 78, 92]
}

df = pd.DataFrame(data)

# -------------------------------
# 2️⃣ Data Visualization
# -------------------------------

print("\nDataset Preview:\n")
print(df)

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------------------
# 3️⃣ Model Training
# -------------------------------

X = df[["study_hours", "sleep_hours", "attendance"]]
y = df["marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 4️⃣ Model Evaluation
# -------------------------------

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# -------------------------------
# 5️⃣ User Input Prediction
# -------------------------------

print("\nEnter Student Details to Predict Marks")

study = float(input("Study Hours: "))
sleep = float(input("Sleep Hours: "))
attendance = float(input("Attendance (%): "))

input_df = pd.DataFrame(
    [[study, sleep, attendance]],
    columns=["study_hours", "sleep_hours", "attendance"]
)

predicted_marks = model.predict(input_df)

print("\nPredicted Marks:", round(predicted_marks[0], 2))