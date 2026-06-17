from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)

# Create static folder if it doesn't exist
os.makedirs("static", exist_ok=True)

# Load dataset
df = pd.read_csv("ADANIPORTS.csv")

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date')

data = df['Close'].values.reshape(-1, 1)

# Normalize data
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
y_train = y[:split]

# Build LSTM Model
model = Sequential()

model.add(LSTM(50, return_sequences=True,
               input_shape=(X_train.shape[1], 1)))

model.add(LSTM(50))

model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    verbose=1
)

# ---------------- ROUTES ---------------- #

@app.route('/')
def home():
    return render_template('welcome.html')


@app.route('/input')
def input_page():
    return render_template('input.html')


@app.route('/predict', methods=['POST'])
def predict():

    days = int(request.form['days'])

    last_60 = data_scaled[-60:]

    temp = last_60.copy()

    predictions = []

    for i in range(days):

        x_input = temp[-60:].reshape(1, 60, 1)

        pred = model.predict(x_input, verbose=0)

        predictions.append(pred[0][0])

        temp = np.vstack((temp, pred))

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    # Create Graph
    plt.figure(figsize=(8, 4))
    plt.plot(predictions)
    plt.title("Future Stock Price Prediction")
    plt.xlabel("Days")
    plt.ylabel("Price")

    graph_path = os.path.join("static", "graph.png")

    plt.savefig(graph_path)

    plt.close()

    return render_template(
        "output.html",
        days=days
    )


if __name__ == "__main__":
    app.run(debug=True)