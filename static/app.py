# app.py

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)

# 📊 Load dataset once
df = pd.read_csv('ADANIPORTS.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date')

data = df['Close'].values.reshape(-1,1)

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Prepare sequences
X = []
y = []

for i in range(60, len(data_scaled)):
    X.append(data_scaled[i-60:i])
    y.append(data_scaled[i])

X, y = np.array(X), np.array(y)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model (only once)
model.fit(X_train, y_train, epochs=5, batch_size=32)

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

    # Take last 60 days
    last_60 = data_scaled[-60:]

    preds = []

    temp = last_60.copy()

    for i in range(days):
        temp_input = temp[-60:].reshape(1,60,1)
        pred = model.predict(temp_input)
        preds.append(pred[0][0])
        temp = np.append(temp, pred)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))

    # 📊 Plot graph
    plt.figure()
    plt.plot(preds)
    plt.title('Future Prediction')
    plt.savefig('static/graph.png')
    plt.close()

    return render_template('output.html', days=days)


if __name__ == '__main__':
    app.run(debug=True)