import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("walmart_sales.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

# Seasonality features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["WeekOfYear"] = df["Date"].dt.isocalendar().week
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Lag features (1–12 weeks)
for lag in range(1, 13):
    df[f"Sales_lag{lag}"] = df["Weekly_Sales"].shift(lag)

# Rolling features
df['Rolling_Mean_4'] = df['Sales_lag1'].rolling(4).mean()
df['Rolling_STD_4'] = df['Sales_lag1'].rolling(4).std()

# Drop rows with NaN (from lagging)
df_model = df.dropna(subset=[f"Sales_lag{lag}" for lag in range(1, 13)]).copy()

# Target: log-transform
y = np.log1p(df_model["Weekly_Sales"])

# Features
feature_cols = ["Year", "WeekOfYear", "Holiday_Flag",
                "Temperature", "Fuel_Price", "CPI", "Unemployment",
                "Sales_lag1","Sales_lag2","Sales_lag3","Sales_lag4",
                "Sales_lag5","Sales_lag6","Sales_lag7","Sales_lag8",
                "Sales_lag9","Sales_lag10","Sales_lag11","Sales_lag12",
                "Rolling_Mean_4","Rolling_STD_4",
                "Month","DayOfWeek"]

X = df_model[feature_cols]

# One-hot encode Month and DayOfWeek
X = pd.get_dummies(X, columns=["Month","DayOfWeek"], drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "walmart_model.pkl")

# Predictions for all data
y_pred_log = model.predict(X)
y_pred = np.expm1(y_pred_log)
y_true = df_model["Weekly_Sales"]

# Save scatter plot with matplotlib
plt.figure(figsize=(8,6))
plt.scatter(y_true, y_pred, color='blue', alpha=0.7, label='Predictions')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect Fit')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Walmart Weekly Sales — Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("walmart_scatter.png", dpi=300)
plt.close()

# Metrics
y_actual = y_true
r2 = r2_score(y_actual, y_pred) * 100
mae = mean_absolute_error(y_actual, y_pred)
mse = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)

print("Model successfully trained and saved as walmart_model.pkl")
print("Scatter plot saved as walmart_scatter.png")
print(f"R²: {r2:.2f}%")
print(f"MAE: {mae:,.2f}")
print(f"MSE: {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
