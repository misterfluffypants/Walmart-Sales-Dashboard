from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

# Load model and data
model = joblib.load("model/walmart_model.pkl")

df = pd.read_csv("model/walmart_sales.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

# Prepare features
df_plot = df.copy()
df_plot["Year"] = pd.to_datetime(df_plot["Date"]).dt.year
df_plot["WeekOfYear"] = pd.to_datetime(df_plot["Date"]).dt.isocalendar().week
df_plot["Month"] = pd.to_datetime(df_plot["Date"]).dt.month
df_plot["DayOfWeek"] = pd.to_datetime(df_plot["Date"]).dt.dayofweek

# Lag features
for lag in range(1, 13):
    df_plot[f"Sales_lag{lag}"] = df_plot["Weekly_Sales"].shift(lag)

# Rolling features
df_plot["Rolling_Mean_4"] = df_plot["Sales_lag1"].rolling(4).mean()
df_plot["Rolling_STD_4"] = df_plot["Sales_lag1"].rolling(4).std()

# Drop rows with any NaN (important!)
df_plot_valid = df_plot.dropna().copy()

# One-hot encode Month and DayOfWeek
df_plot_valid = pd.get_dummies(df_plot_valid, columns=["Month", "DayOfWeek"], drop_first=True)

# Add missing columns for model
for col in model.feature_names_in_:
    if col not in df_plot_valid.columns:
        df_plot_valid[col] = 0

# Reorder columns
X_plot_ready = df_plot_valid[model.feature_names_in_]

# Ensure no NaN remains
X_plot_ready = X_plot_ready.fillna(0)

# Predictions in original scale
y_pred_log = model.predict(X_plot_ready)
y_pred_plot = np.expm1(y_pred_log)
y_true = df_plot_valid["Weekly_Sales"]

# Create Plotly scatter plot
def create_plot(y_true, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        name="Predictions",
        marker=dict(color="blue", size=6, opacity=0.7)
    ))
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_true,
        mode="lines",
        name="Perfect Fit",
        line=dict(color="red", dash="dash")
    ))
    fig.update_layout(
        title="Walmart Weekly Sales — Linear Regression",
        xaxis_title="Actual Sales",
        yaxis_title="Predicted Sales",
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


plot_html = create_plot(y_true, y_pred_plot)

# Helper function for new prediction
def prepare_features(date, holiday_flag, temperature, fuel_price, cpi, unemployment):
    dt = pd.Timestamp(date)
    data = {
        "Year": dt.year,
        "WeekOfYear": dt.isocalendar().week,
        "Holiday_Flag": holiday_flag,
        "Temperature": temperature,
        "Fuel_Price": fuel_price,
        "CPI": cpi,
        "Unemployment": unemployment,
        "Month": dt.month,
        "DayOfWeek": dt.dayofweek
    }
    df_new = pd.DataFrame([data])

    # One-hot encode
    df_new = pd.get_dummies(df_new, columns=["Month", "DayOfWeek"], drop_first=True)

    # Add missing columns
    for col in model.feature_names_in_:
        if col not in df_new.columns:
            df_new[col] = 0

    df_new = df_new[model.feature_names_in_]
    df_new = df_new.fillna(0)
    return df_new

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = ""

    if request.method == "POST":
        try:
            date = request.form["date"]
            holiday_flag = float(request.form["holiday_flag"])
            temperature = float(request.form["temperature"])
            fuel_price = float(request.form["fuel_price"])
            cpi = float(request.form["cpi"])
            unemployment = float(request.form["unemployment"])

            X_new = prepare_features(date, holiday_flag, temperature, fuel_price, cpi, unemployment)
            pred_log = model.predict(X_new)[0]
            prediction = np.expm1(pred_log)
            prediction_text = f"Predicted Weekly Sales: ${prediction:,.2f}"
        except Exception as e:
            prediction_text = f"Error: {e}"

    table_data = df.to_dict(orient="records")

    return render_template(
        "index.html",
        plot_html=plot_html,
        prediction_text=prediction_text,
        table_data=table_data
    )

if __name__ == "__main__":
    app.run(debug=True)
