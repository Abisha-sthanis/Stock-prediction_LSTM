# model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load Dataset
df = pd.read_csv('ADANIPORTS.csv')

# Date Conversion
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date')

# Moving Average
df['MA_10'] = df['Close'].rolling(window=10).mean()

# Select Close Price
data = df['Close'].values.reshape(-1, 1)

# Normalization
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data)

# Create Sequences
X = []
y = []

for i in range(60, len(data_scaled)):
    X.append(data_scaled[i-60:i])
    y.append(data_scaled[i])

X = np.array(X)
y = np.array(y)

# Train Test Split
split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# Build LSTM Model
model = Sequential()

model.add(
    LSTM(
        50,
        return_sequences=True,
        input_shape=(X_train.shape[1],1)
    )
)

model.add(Dropout(0.2))

model.add(LSTM(50))

model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# Train Model
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32
)

# Prediction
predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)

actual = scaler.inverse_transform(
    y_test.reshape(-1,1)
)

# RMSE
rmse = np.sqrt(
    mean_squared_error(actual, predictions)
)

# MAE
mae = mean_absolute_error(
    actual,
    predictions
)

print("RMSE :", rmse)
print("MAE  :", mae)

# Save Results
results = pd.DataFrame({
    'Actual': actual.flatten(),
    'Predicted': predictions.flatten()
})

results.to_csv(
    'results.csv',
    index=False
)

# Save Accuracy Report
accuracy = pd.DataFrame({
    'RMSE':[rmse],
    'MAE':[mae]
})

accuracy.to_csv(
    'accuracy_report.csv',
    index=False
)

# Actual vs Predicted Graph
plt.figure(figsize=(10,5))

plt.plot(actual, label='Actual Price')
plt.plot(predictions, label='Predicted Price')

plt.title('Actual vs Predicted Stock Price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()

plt.savefig('actual_vs_predicted.png')
plt.show()

# Moving Average Graph
plt.figure(figsize=(10,5))

plt.plot(df['Close'], label='Close Price')
plt.plot(df['MA_10'], label='MA_10')

plt.title('Moving Average Analysis')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()

plt.savefig('moving_average.png')
plt.show()

print("Model Training Completed Successfully")