import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Hyperparameters
max_words = 5000
max_len = 200

# Load and prepare data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
X_train = pad_sequences(x_train, maxlen=max_len)
X_test = pad_sequences(x_test, maxlen=max_len)

# Build model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=32, input_length=max_len))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
print("Training...")
model.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.2)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

# Prepare decoding utilities
word_index = imdb.get_word_index()
reverse_word_index = {v: k for (k, v) in word_index.items()}

def decode_review(review_indices):
    # imdb reserves indices 0,1,2 for padding and special tokens; dataset words start at index 3
    return " ".join([reverse_word_index.get(i - 3, "?") for i in review_indices])

# Predict a sample review
sample_review = X_test[0]
prediction = model.predict(np.expand_dims(sample_review, axis=0))[0, 0]

# Print original (raw) review tokens and predicted sentiment
print("\nReview text:", decode_review(x_test[0]))
print("Predicted Sentiment:", "Positive" if prediction > 0.5 else "Negative")
