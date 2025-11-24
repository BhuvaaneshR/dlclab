#Build GAN with Keras/TensorFlow
# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

# Step 2: Load the Data
# Note: We don't need the labels (y_train, y_test) because this is Unsupervised Learning.
# The Autoencoder tries to predict the *input* from the input.
(x_train, _), (x_test, _) = mnist.load_data()

# Step 3: Preprocessing
# Normalize pixel values (0-255 -> 0-1)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Flatten the images: 28x28 -> 784
# np.prod(x_train.shape[1:]) calculates 28*28 = 784 automatically
feature_vector_length = np.prod(x_train.shape[1:]) 

x_train = x_train.reshape((len(x_train), feature_vector_length))
x_test = x_test.reshape((len(x_test), feature_vector_length))

# Step 4: Build the Autoencoder Architecture

# Input Placeholder
input_img = Input(shape=(feature_vector_length,))

# ENCODER: Compresses the 784 pixels down to 32 floats
# This is the "Bottleneck" layer
encoded = Dense(32, activation='relu')(input_img)

# DECODER: Reconstructs the 784 pixels from the 32 compressed floats
# We use 'sigmoid' because the input data is normalized between 0 and 1
decoded = Dense(feature_vector_length, activation='sigmoid')(encoded)

# Create the Model mapping Input -> Output
autoencoder = Model(input_img, decoded)

# Step 5: Compile and Train
# We use 'binary_crossentropy' because we are treating pixel intensity like a probability (0 to 1)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Notice that x_train is both the input AND the target (x_train, x_train)
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose=1)

# Step 6: Evaluate
test_loss = autoencoder.evaluate(x_test, x_test, verbose=0)
decoded_imgs = autoencoder.predict(x_test)

# Calculate Pixel-wise Accuracy
# We threshold the image: if pixel > 0.5 make it 1, else 0
threshold = 0.5
correct_predictions = np.sum(
    np.where(x_test >= threshold, 1, 0) == np.where(decoded_imgs >= threshold, 1, 0)
)
total_pixels = x_test.shape[0] * x_test.shape[1]
test_accuracy = correct_predictions / total_pixels

print(f"Test Loss: {test_loss:.4f}")
print(f"Pixel-wise Accuracy: {test_accuracy:.4f}")

# Step 7: Visualize Results
n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))

for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n // 2:
        ax.set_title("Original Images")

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n // 2:
        ax.set_title("Reconstructed Images")

# IMPORTANT: This must be outside the loop
plt.show()