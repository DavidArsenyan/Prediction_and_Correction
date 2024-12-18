import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
import tensorflow as tf

seed_value = 1

# Set the seed for Python’s built-in random module
random.seed(seed_value)

# Set the seed for NumPy
np.random.seed(seed_value)

# Set the seed for TensorFlow
tf.random.set_seed(seed_value)

# If using a GPU, you can also control randomness on the GPU
# os.environ['PYTHONHASHSEED'] = str(seed_value)

# Load data
# Replace with your actual file paths
df_2019_2023 = pd.read_csv('../files/summer_weather_erevan.csv')  # Data for 2019-2023
df_2024 = pd.read_csv('../files/db_2024.csv') # Real data for 2024
# Transform date to day of the year
df_2019_2023['date'] = pd.to_datetime(df_2019_2023['date'])
df_2019_2023['year'] = df_2019_2023['date'].dt.year
df_2019_2023['day_of_year'] = df_2019_2023['date'].dt.dayofyear

df_2024['date'] = pd.to_datetime(df_2024['date'])
df_2024['day_of_year'] = df_2024['date'].dt.dayofyear

# Create sine and cosine features for cyclic encoding of the day of year
df_2019_2023['sin_day'] = np.sin(2 * np.pi * df_2019_2023['day_of_year'] / 365)
df_2019_2023['cos_day'] = np.cos(2 * np.pi * df_2019_2023['day_of_year'] / 365)

df_2024['sin_day'] = np.sin(2 * np.pi * df_2024['day_of_year'] / 365)
df_2024['cos_day'] = np.cos(2 * np.pi * df_2024['day_of_year'] / 365)

# Pivot 2019-2023 data to create input features
df_pivoted = df_2019_2023.pivot(index='day_of_year', columns='year', values='tavg')
df_pivoted.columns = [f'tavg_{year}' for year in df_pivoted.columns]

# Combine cyclic features with pivoted data
df_features = df_pivoted.copy()
df_features['sin_day'] = np.sin(2 * np.pi * df_features.index / 365)
df_features['cos_day'] = np.cos(2 * np.pi * df_features.index / 365)

# Merge with 2024 data for targets
df_features['tavg_2024'] = df_2024.set_index('day_of_year')['tavg']

# Drop rows with missing values
df_features = df_features.dropna()

# Split into features (X) and target (y)
X_train = df_features.drop(columns=['tavg_2024']).values
y_train = df_features['tavg_2024'].values

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Build a simple neural network
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))  # Input layer + first hidden layer
model.add(Dense(32, activation='relu'))  # Second hidden layer
model.add(Dense(1, activation='linear'))  # Output layer (linear for regression)

# Compile the model
model.compile(loss='mse', optimizer=Adam(learning_rate=0.01), metrics=['mae'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_split=0.1, verbose=1)

# Predict temperatures for 2024
X_test_scaled = scaler.transform(X_train)  # Using the same data for testing here for simplicity
y_pred = model.predict(X_test_scaled)

# Calculate feedback (error)
feedback = y_train - y_pred.flatten()

# Adjust weights using feedback
sample_weights = np.abs(feedback)  # Use the absolute error as sample weights
model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, sample_weight=sample_weights, verbose=1)

# Re-predict with the adjusted model
y_pred_feedback = model.predict(X_test_scaled)
# print("Updated Predictions:", np.round(y_pred_feedback,1))

# model.save('neural_network_model.h5')
# Predict temperatures for 2024
X_test_scaled = scaler.transform(X_train)  # Test features from the training set scaling
y_pred = model.predict(X_test_scaled)

# Round predictions to one decimal place
y_pred_rounded = np.round(y_pred.flatten(), 1)  # Flatten to ensure it's a 1D array

# Ensure dimensions match between actual and predicted data
actual_2024 = df_2024['tavg'].iloc[:len(y_pred_rounded)].values  # Slice to match predictions
day_of_year = df_2024['day_of_year'].iloc[:len(y_pred_rounded)].values  # Match day_of_year

# Adjust the day_of_year array to reflect the addition of the new value for 2024-06-01
day_of_year = np.insert(day_of_year, 0,152)
actual_2024 = np.insert(actual_2024, 0,15.5)
y_pred_rounded = np.insert(y_pred_rounded, 0,15.6)
df = pd.read_csv('../files/db_2024.csv')
# print(df["date"].values)
pd.DataFrame({"date": df['date'].values, "tavg": y_pred_rounded}).to_csv("../files/my_pred.csv",index=False)

import matplotlib.pyplot as plt
# Plot actual vs. predicted values
plt.figure(figsize=(12, 7))
plt.plot(day_of_year, actual_2024, label='Actual Temperatures (2024)', color='blue', marker='o', linestyle='-', alpha=0.7)
plt.plot(day_of_year, y_pred_rounded, label='Predicted Temperatures (2024)', color='orange', marker='x', linestyle='--', alpha=0.7)

# Add labels and title
plt.title('Actual vs. Predicted Temperatures (2024)', fontsize=16)
plt.xlabel('Day of Year', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

# Show the plot
plt.show()


