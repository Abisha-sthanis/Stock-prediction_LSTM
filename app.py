from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (required for servers)
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)

# Create static folder automatically
os.makedirs("static", exist_ok=True)

# Load dataset
df = pd.read_csv("ADANIPORTS.csv")

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date')

# Use Close Price
data = df['Close'].values.reshape(-1, 1)

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Create sequences
X = []
y = []

for i in range(60, len(data_scaled)):
    X.append(data_scaled[i-60:i])
    y.append(data_scaled[i])

X = np.array(X)
y = np.array(y)

# Train/Test Split
split = int(len(X) * 0.8)

X_train = X[:split]
X_test  = X[split:]
y_train = y[:split]
y_test  = y[split:]

# Build Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# RMSE Calculation
test_predictions = model.predict(X_test)
test_predictions  = scaler.inverse_transform(test_predictions)
actual            = scaler.inverse_transform(y_test.reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(actual, test_predictions))

# Generate Actual vs Predicted graph once at startup
plt.figure(figsize=(10, 4))
plt.plot(actual,           label='Actual Price',    color='#0090ff', linewidth=1.5)
plt.plot(test_predictions, label='Predicted Price', color='#00e5a0', linewidth=1.5)
plt.title('Actual vs Predicted Stock Price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/actual_vs_predicted.png')
plt.close()


# ---------------- ROUTES ---------------- #

@app.route('/')
def home():
    # ← was 'welcome.html', now matches stock_prediction_home.html
    return render_template('stock_prediction_home.html')


@app.route('/input')
def input_page():
    # ← was 'input.html', now matches stock_prediction.html
    return render_template('stock_prediction.html')


@app.route('/predict', methods=['POST'])
def predict():

    stock_name = request.form['stock']
    days       = int(request.form['days'])

    # New optional fields from enhanced input form
    exchange   = request.form.get('exchange', 'NSE')
    model_type = request.form.get('model', 'lstm')

    # Last 60 days
    last_60 = data_scaled[-60:]
    temp    = last_60.copy()
    predictions = []

    # Future prediction loop
    for i in range(days):
        x_input = temp[-60:].reshape(1, 60, 1)
        pred    = model.predict(x_input, verbose=0)
        predictions.append(pred[0][0])
        temp = np.vstack((temp, pred))

    # Convert back to original price scale
    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    # Save future prediction graph
    plt.figure(figsize=(10, 4))
    plt.plot(predictions, marker='o', color='#00e5a0',
             linewidth=1.5, markersize=4, label='Predicted Price')
    plt.title(f'{stock_name} — Future Price Forecast')
    plt.xlabel('Days')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/graph.png')
    plt.close()

    # Build table data
    table_data = [
        {"day": i + 1, "price": round(float(predictions[i][0]), 2)}
        for i in range(len(predictions))
    ]

    return render_template(
        # ← was 'output.html', now matches stock_prediction_result.html
        'stock_prediction_result.html',
        stock_name=stock_name,
        days=days,
        rmse=round(rmse, 2),
        table_data=table_data,
        exchange=exchange,
        model_type=model_type
    )


if __name__ == '__main__':
    app.run(debug=True)