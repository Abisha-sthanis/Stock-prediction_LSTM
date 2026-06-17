# model.py

# 📦 Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 📊 Load dataset
df = pd.read_csv('E:/Stock prediction/ADANIPORTS.csv')

# 🧹 Data preprocessing
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date')

# Use only required columns
df = df[['Date', 'Close']]

# 🎯 Prepare data
data = df['Close'].values.reshape(-1,1)

# 📉 Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 🔢 Create sequences (60 days)
X = []
y = []

for i in range(60, len(data_scaled)):
    X.append(data_scaled[i-60:i])
    y.append(data_scaled[i])

X, y = np.array(X), np.array(y)

# ✂️ Train-test split
split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# 🧠 Build LSTM model
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(50))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 🏋️ Train model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# 🔮 Prediction
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# 📊 Convert actual values
actual = scaler.inverse_transform(y_test.reshape(-1,1))

# 💾 Save results for visualization
result = pd.DataFrame({
    'Actual': actual.flatten(),
    'Predicted': predictions.flatten()
})

result.to_csv('results.csv', index=False)

print("✅ Model training completed and results saved!")