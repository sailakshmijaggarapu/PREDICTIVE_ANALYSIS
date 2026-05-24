# Predictive Analytics Using Historical Data
# Dataset: AirPassengers.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# make graphs look cleaner
sns.set_style("whitegrid")

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("AirPassengers.csv")

print("First 5 rows:")
print(df.head())

# rename columns
df.columns = ["date", "passengers"]

# convert date column into datetime format
df["date"] = pd.to_datetime(df["date"])

# check missing values
print("\nMissing values:")
print(df.isnull().sum())

# sort data by date
df = df.sort_values("date").reset_index(drop=True)

# -------------------------------
# Feature engineering
# -------------------------------

# extract month and year
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# time index
df["t"] = np.arange(len(df))

# previous values
df["lag_1"] = df["passengers"].shift(1)
df["lag_12"] = df["passengers"].shift(12)

# rolling average
df["rolling_3"] = df["passengers"].rolling(3).mean()

# remove null rows created because of shift/rolling
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print("\nProcessed data:")
print(df.head())

# -------------------------------
# Define input and output
# -------------------------------
features = ["t", "month", "lag_1", "lag_12", "rolling_3"]

X = df[features]
y = df["passengers"]

# use first 80% for training
split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# -------------------------------
# Train model
# -------------------------------
model = LinearRegression()

model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# -------------------------------
# Accuracy
# -------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("RMSE :", round(rmse, 2))
print("MAE  :", round(mae, 2))
print("R2   :", round(r2, 4))

# -------------------------------
# Plot 1 - Historical trend
# -------------------------------
plt.figure(figsize=(12, 4))

plt.plot(df["date"], df["passengers"])

plt.title("Historical Passenger Trend")
plt.xlabel("Year")
plt.ylabel("Passengers")

plt.tight_layout()
plt.savefig("plot1_historical.png", dpi=150)
plt.show()

# -------------------------------
# Plot 2 - Actual vs Predicted
# -------------------------------
test_dates = df["date"].iloc[split:].values
plt.figure(figsize=(12, 5))

plt.plot(test_dates, y_test.values, label="Actual")
plt.plot(test_dates, y_pred, label="Predicted", linestyle="--", color="tomato")
plt.title("Actual vs Predicted Passengers")
plt.xlabel("Date")
plt.ylabel("Passengers")

plt.legend()

plt.tight_layout()
plt.savefig("plot2_predictions.png", dpi=150)
plt.show()

# -------------------------------
# Plot 3 - Residuals
# -------------------------------
residuals = y_test.values - y_pred

plt.figure(figsize=(12, 3))

plt.plot(test_dates, residuals)
plt.axhline(0, linestyle="--")

plt.title("Residual Errors")
plt.xlabel("Date")
plt.ylabel("Error")

plt.tight_layout()
plt.savefig("plot3_residuals.png", dpi=150)
plt.show()