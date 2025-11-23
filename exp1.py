# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Configuration
feature_vector_length = 784  # 28 x 28 pixels flattened
num_classes = 10             # Digits 0 through 9

# Step 2: Load the Data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Step 3: Preprocessing
# Reshape the data - Flattening the images from 28x28 to a 1D vector of 784
X_train = X_train.reshape(X_train.shape[0], feature_vector_length)
X_test = X_test.reshape(X_test.shape[0], feature_vector_length)

# Normalize pixel values (0-255 -> 0-1)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# One-hot encode the labels
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

# Define input shape as a tuple
input_shape = (feature_vector_length,) 
print(f'Feature shape: {input_shape}')

# Step 4: Build the Model (Multi-Layer Perceptron)
model = Sequential()
# First Hidden Layer
model.add(Dense(350, input_shape=input_shape, activation='relu'))
# Second Hidden Layer
model.add(Dense(50, activation='relu'))
# Output Layer
model.add(Dense(num_classes, activation='softmax'))

# Step 5: Compile and Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=250, verbose=1, validation_split=0.2)

# Step 6: Evaluate the Model
test_results = model.evaluate(X_test, Y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}')

# Step 7: Visualize Predictions
# Pick the first 5 images from the test set
predictions = model.predict(X_test[:5])
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(Y_test[:5], axis=1)

# Indentation fixed here
for i in range(5):
    # We must reshape the flat vector back to 28x28 to plot it as an image
    plt.figure(figsize=(2, 2))
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {predicted_classes[i]}, Actual: {true_classes[i]}")
    plt.axis('off')
    plt.show()