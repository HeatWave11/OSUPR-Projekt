import numpy as np
import os
import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import TextVectorization, BatchNormalization
from keras.src.layers import Embedding, LSTM, Dense, Dropout
from keras.src.callbacks import EarlyStopping
import json

from data import training_labels, validation_labels, max_length_95


# Load preprocessed texts from the JSON file
with open('preprocessed_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    training_texts = data['custom_preprocessed_training_texts']
    validation_texts = data['custom_preprocessed_validation_texts']

print("Data loaded successfully from JSON!")

## VECTORIZATION
sequential_vectorizer = TextVectorization(
    max_tokens=20000,
    output_mode='int',
    output_sequence_length=max_length_95,
)
sequential_vectorizer.adapt(training_texts)
vectorized_training_texts = sequential_vectorizer(training_texts)
vectorized_validation_texts = sequential_vectorizer(validation_texts)

## 1: DEFINING THE MODEL
model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=max_length_95),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    LSTM(32, return_sequences=False),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

## 2: TRAINING THE MODEL
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    vectorized_training_texts,
    training_labels,
    validation_data=(vectorized_validation_texts, validation_labels),
    epochs=10,
    batch_size=32,
    callbacks=[early_stopping]
)

## 3: EVALUATING THE MODEL
loss, accuracy = model.evaluate(vectorized_validation_texts, validation_labels)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

## 4: SAVING THE MODEL AND VECTORIZER
model.save("SavedModels/rnn_seq_model.keras")
print("\nModel saved successfully to 'SavedModels/rnn_seq_model.keras'")


print("Saving vectorizer state...")

# Get the configuration and the learned vocabulary directly
config = sequential_vectorizer.get_config()
vocabulary = sequential_vectorizer.get_vocabulary()

# Combine them into a single dictionary
vectorizer_data_to_save = {
    "config": config,
    "vocabulary": vocabulary # Save the vocabulary directly under its own key
}

# Save to the .json file, which will be named "rnn_vectorizer.json"
with open("SavedVectorizers/rnn_vectorizer.json", "w", encoding='utf-8') as f:
    json.dump(vectorizer_data_to_save, f, indent=4)

print("Vectorizer state saved successfully to 'SavedVectorizers/rnn_vectorizer.json'")
