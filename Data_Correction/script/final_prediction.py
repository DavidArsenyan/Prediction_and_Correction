import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pattern_matching import pattern, corr_data
import random
import tensorflow as tf

seed_value = 18
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
# Load the datasets
db = pd.read_csv('../files/corr_data.csv')  # Corrupted data
pred = pd.read_csv("../files/my_pred.csv")  # Predicted data

# Step 1: Identify corrupted rows
invalid_rows = corr_data(db)
print("Invalid rows identified:")
print(invalid_rows)

# Step 2: Correct corrupted rows using pattern matching
corrected_tavg = pattern(db, pred)
db['tavg'] = corrected_tavg

# Ensure tavg is numeric (convert non-numeric to NaN)
db['tavg'] = pd.to_numeric(db['tavg'], errors='coerce')  # Convert errors to NaN
db['tavg'].fillna(db['tavg'].mean(), inplace=True)  # Fill NaN with the mean temperature

# Step 3: Add features for better prediction
db['date'] = pd.to_datetime(db['date'])
db['date_ordinal'] = db['date'].apply(lambda x: x.toordinal())  # Date encoded as ordinal
db['day_of_week'] = db['date'].dt.dayofweek  # Day of the week
db['month'] = db['date'].dt.month  # Month of the year
db['day_of_year'] = db['date'].dt.dayofyear  # Day of the year for seasonal patterns

# Cyclic encoding for day of year (sine and cosine)
db['sin_day'] = np.sin(2 * np.pi * db['day_of_year'] / 365)
db['cos_day'] = np.cos(2 * np.pi * db['day_of_year'] / 365)

# Additional feature: Days since the start of the dataset
db['days_since_start'] = (db['date'] - db['date'].min()).dt.days

# Features and target
X = db[['date_ordinal', 'day_of_week', 'month', 'days_since_start', 'sin_day', 'cos_day']]  # Added 'sin_day' and 'cos_day'
y = db['tavg']  # Target: temperature (tavg)

# Step 4: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Step 5: Build and train the neural network
nn_model = MLPRegressor(hidden_layer_sizes=(64, 128), activation='relu', solver='adam', max_iter=2000, random_state=42)
nn_model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = nn_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Squared Error (MSE) on Test Data: {mse}")
print(f"Mean Absolute Error (MAE) on Test Data: {mae}")

# Step 7: Experiment with corrupted data (invalid rows)
invalid_data = invalid_rows.copy()
invalid_data['date'] = pd.to_datetime(invalid_data['date'])
invalid_data['date_ordinal'] = invalid_data['date'].apply(lambda x: x.toordinal())
invalid_data['day_of_week'] = invalid_data['date'].dt.dayofweek
invalid_data['month'] = invalid_data['date'].dt.month
invalid_data['day_of_year'] = invalid_data['date'].dt.dayofyear

# Cyclic encoding for day of year
invalid_data['sin_day'] = np.sin(2 * np.pi * invalid_data['day_of_year'] / 365)
invalid_data['cos_day'] = np.cos(2 * np.pi * invalid_data['day_of_year'] / 365)
invalid_data['days_since_start'] = (invalid_data['date'] - db['date'].min()).dt.days  # Add same feature

X_invalid = invalid_data[['date_ordinal', 'day_of_week', 'month', 'days_since_start', 'sin_day', 'cos_day']]
X_invalid_scaled = scaler.transform(X_invalid)  # Scale invalid data
predicted_temps = nn_model.predict(X_invalid_scaled)

# Round the predicted temperatures to one decimal place
invalid_data['predicted_tavg'] = np.round(predicted_temps, 1)

# Step 8: Prepare the final output containing only date and new tavg (rounded to 1 decimal)
final_output = invalid_data[['date', 'predicted_tavg']]
# Save the corrected invalid rows with predictions
final_output.to_csv('../files/corrected_invalid_rows.csv', index=False)

print("Predicted values for invalid rows saved to 'corrected_invalid_rows.csv'.")