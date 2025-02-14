import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import TextVectorization
from keras.src.layers import Embedding, LSTM, Dense, Dropout
from data import training_labels,validation_labels
import pickle

from data import max_length_95

# Load preprocessed texts
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    training_texts = data['custom_preprocessed_training_texts']
    validation_texts = data['custom_preprocessed_validation_texts']

## VECTORIZATION

# Vectorization layer - Sequential approach (Word order matters - for RNNs and Transformers)
sequential_vectorizer = TextVectorization(
    max_tokens = 20000,  # Maximum number of unique words to consider
    output_mode = 'int',  # Output integers (token indices)
    output_sequence_length = max_length_95,  # For Padding/truncating all sequences to this length
# Important parameter than we need to think hard about how to set
# Too long isn't good because it will be wasteful
# Too short isn't good either because it will truncate longer tweets
# Set it to 95th percentile for now!
# I put standardization outside of the model itself, so it only needs to be done once
  #standardize = 'lower'  # Custom
# Reminder that standardization isn't done yet!
)

# Adapt the vectorizer to the training data
sequential_vectorizer.adapt(training_texts)

# Vectorize training data
vectorized_training_texts = sequential_vectorizer(training_texts)

# Vectorize validation data
vectorized_validation_texts = sequential_vectorizer(validation_texts)




## 1: DEFINING THE MODEL

# Define the model
model = Sequential()

# Add an Embedding layer
# Input_dim is the size of the vocabulary (20000 in your case)
# Output_dim is the size of the embedding vectors (e.g., 128)
# Input_length is the length of the input sequences (max_length_95)
model.add(Embedding(input_dim=20000, output_dim=128, input_length=max_length_95))

# Add an LSTM layer (or GRU layer)
# You can adjust the number of units (e.g., 64, 128, etc.)
model.add(LSTM(64, return_sequences=False))  # Set return_sequences=True if stacking RNN layers

# Add a Dense layer for classification
# Assuming binary classification (e.g., sentiment analysis)
model.add(Dense(1, activation='sigmoid'))

# Add Dropout for regularization (optional)
model.add(Dropout(0.5))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

## 3: TRAINING THE MODEL

# Train the model
history = model.fit(
    vectorized_training_texts,  # Vectorized training texts
    training_labels,           # Training labels
    validation_data=(vectorized_validation_texts, validation_labels),  # Validation data
    epochs=10,                  # Number of epochs
    batch_size=32               # Batch size
)

## 4: EVALUATING THE MODEL
# Evaluate the model on the validation data
loss, accuracy = model.evaluate(vectorized_validation_texts, validation_labels)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

## 5:

# Vectorization layer - Sequential approach (Word order matters - for RNNs and Transformers)
##sequential_vectorizer = TextVectorization(
#    max_tokens = 20000,  # Maximum number of unique words to consider
#    output_mode = 'int',  # Output integers (token indices)
#    output_sequence_length = max_length_95,  # For Padding/truncating all sequences to this length
    # Important parameter than we need to think hard about how to set
    # Too long isn't good because it will be wasteful
    # Too short isn't good either because it will truncate longer tweets
    # Set it to 95th percentile for now!
    # I put standardization outside of the model itself, so it only needs to be done once
    # standardize = 'lower'  # Custom
    # Reminder that standardization isn't done yet!
#)

# Adapting the TextVectorization layer to the training data
# Remember - we only want the vectorizer adapted to training data!
##sequential_vectorizer.adapt(custom_preprocessed_training_texts1)

# Checking the vocabulary size and sample tokens (training)
# Remember - first two entries in the vocabulary are:
# the mask token (index 0)
# the OOV token (index 1)
##vocabulary = sequential_vectorizer.get_vocabulary()
##print(f"Vocabulary size: {len(vocabulary)}")
# print("Top 10 tokens:", vocabulary[:10])

# Some testing
##sample_texts = ["I love programming.", "This is irrelevant.", "Hello Microsoft!"]
##vectorized_texts = sequential_vectorizer(sample_texts)
# print("Original text:", sample_texts)
# print("Vectorized text:", vectorized_texts.numpy())

# Training data vectorization
##vectorized_training_texts = sequential_vectorizer(custom_preprocessed_training_texts1)
# Validation data vectorization
##vectorized_validation_texts = sequential_vectorizer(custom_preprocessed_validation_texts1)