
# Walmart Sales Dashboard

**Walmart Sales Dashboard** is a web application for visualizing and predicting weekly sales for Walmart stores using Linear Regression.

Users can:

* View a scatter plot of actual vs. predicted sales.
* Make sales predictions based on input features.
* Browse historical sales data in a table.

---

## Features

1. **Sales Visualization**

   * Scatter plot of actual and predicted sales.
2. **Sales Prediction**

   * Input form for Date, Holiday Flag, Temperature, Fuel Price, CPI, and Unemployment.
   * Displays predicted weekly sales.
3. **Data Table**

   * Complete historical sales data with pagination.

---

## Technologies

* **Frontend**: HTML, CSS, JavaScript
* **Backend**: Python, Flask
* **Machine Learning**: Scikit-learn (Linear Regression)
* **Visualization**: Plotly, Matplotlib
* **Data**: CSV file `walmart_sales.csv`

---

## Installation and Running

1. **Clone the repository**

```bash
git clone "https://github.com/misterfluffypants/Walmart-Sales-Dashboard"
cd Walmart-Sales-Dashboard
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Train the model (optional)**
   If you have the CSV data file:

```bash
python model/train_and_plot.py
```

* The trained model will be saved as `model/walmart_model.pkl`.
* Scatter plot will be saved as `model/walmart_scatter.png`.

5. **Run the Flask application**

```bash
python app.py
```

Open in browser: `http://127.0.0.1:5000/`

---

## Project Structure

```
├── app.py                     # Main Flask application
├── model/
│   ├── train_and_plot.py       # Script to train model and generate plot
│   ├── walmart_model.pkl       # Trained model
│   └── walmart_sales.csv       # Sales data
├── templates/
│   └── index.html              # HTML template
├── static/
│   ├── css/style.css
│   └── javascript/js.js
├── README.md
```

---

## Usage

1. Go to the main page.
2. View the scatter plot of actual vs. predicted sales.
3. Use the form to make predictions and click **Predict**.
4. Browse historical sales data in the table.

---

## Model Metrics

* **R²**: Coefficient of determination
* **MAE, MSE, RMSE**: Prediction error metrics

Metrics are printed when training the model with `train_and_plot.py`.

---

If you want, I can also make a **GitHub-ready version with badges, screenshots, and example predictions** to make it look more professional.

Do you want me to do that?
