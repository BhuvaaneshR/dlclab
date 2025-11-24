# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import r2_score

# Configuration
np.random.seed(0)
seq_length = 10     # The length of the sequence (time steps)
num_samples = 1000  # Total number of data points

# Step 2: Generate Synthetic Data
# Input: Random numbers. Shape: (Samples, Time Steps, Features)
X = np.random.randn(num_samples, seq_length, 1)

# Target: The sum of the sequence + some noise
# This is a "Regression" task (predicting a continuous number)
y = X.sum(axis=1) + 0.1 * np.random.randn(num_samples, 1)

# Step 3: Split Data into Train and Test sets
split_ratio = 0.8
split_index = int(split_ratio * num_samples)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Step 4: Build the Recurrent Neural Network (RNN)
model = Sequential()

# SimpleRNN Layer
# units=50: The internal memory size
# input_shape=(10, 1): 10 time steps, 1 feature per step
model.add(SimpleRNN(units=50, activation='relu', input_shape=(seq_length, 1)))

# Dense Output Layer
# units=1: We are predicting a single continuous value (the sum)
model.add(Dense(units=1))

# Step 5: Compile the Model
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Step 6: Train the Model
batch_size = 30
epochs = 50

print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    verbose=1
)

# Step 7: Evaluate the Model
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'\nTest Loss (MSE): {test_loss:.4f}')

# Make predictions
y_pred = model.predict(X_test)

# Calculate R^2 Score (Standard metric for regression)
r2 = r2_score(y_test, y_pred)
print(f'Test R^2 Score: {r2:.4f}')

# Step 8: Visualizing the Results
# Let's verify the first 5 predictions against the actual values
print("\nSample Predictions:")
for i in range(5):
    print(f"Actual Sum: {y_test[i][0]:.2f}, Predicted Sum: {y_pred[i][0]:.2f}")

# Plotting Actual vs Predicted for the first 50 test samples
plt.figure(figsize=(10, 5))
plt.plot(y_test[:50], label='Actual Value', marker='o')
plt.plot(y_pred[:50], label='Predicted Value', marker='x')
plt.title('SimpleRNN Regression: Actual Sum vs Predicted Sum')
plt.xlabel('Sample Index')
plt.ylabel('Sum Value')
plt.legend()
plt.show()