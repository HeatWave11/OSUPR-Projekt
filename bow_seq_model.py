import os
import json  # <-- Import JSON
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras.src.layers import TextVectorization, Dense, Dropout
from keras.src.models import Sequential
import tensorflow as tf
import numpy as np
import data  # Assuming data.py has your labels

bow_training_labels = data.training_labels
bow_validation_labels = data.validation_labels

# --- SECTION 1: LOAD PREPROCESSED DATA USING JSON (REPLACES PICKLE) ---
# Load preprocessed texts from the JSON file
with open('preprocessed_data.json', 'r', encoding='utf-8') as f:
    loaded_data = json.load(f)
    bow_training_texts = loaded_data['custom_preprocessed_training_texts']
    bow_validation_texts = loaded_data['custom_preprocessed_validation_texts']
# --- END OF SECTION 1 ---

print("Data loaded successfully from JSON!")
print("First 5 training texts:", bow_training_texts[:5])

# 1: Vectorization layer - BoW approach
bow_vectorizer = TextVectorization(
    max_tokens=10000,
    output_mode='multi_hot',
)

# 2: Adapt the vectorizer to the training data
bow_vectorizer.adapt(bow_training_texts)

# Vectorize training and validation data
vectorized_bow_training_texts = bow_vectorizer(bow_training_texts)
vectorized_bow_validation_texts = bow_vectorizer(bow_validation_texts)

# 3: Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(vectorized_bow_training_texts.shape[1],)), # Good practice to define input_shape
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# 4: Compiling the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5: Print model summary
model.summary()

# 6: Fit the model
history = model.fit(
    x=vectorized_bow_training_texts,
    y=bow_training_labels,
    validation_data=(vectorized_bow_validation_texts, bow_validation_labels),
    epochs=10,
    batch_size=32
)

# 7: Evaluate the model
loss, accuracy = model.evaluate(vectorized_bow_validation_texts, bow_validation_labels)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# Save the trained model (this part is already safe)
model.save("SavedModels/bow_seq_model_dropout.keras")
print("\nModel saved successfully to 'SavedModels/bow_seq_model_dropout.keras'")


# --- SECTION 2: SAVE THE VECTORIZER USING JSON (REPLACES PICKLE) ---
# Get the configuration and weights (vocabulary) from the adapted vectorizer
config = bow_vectorizer.get_config()
weights = bow_vectorizer.get_weights() # This is a list with one numpy array containing the vocabulary

# The weights need to be converted to a standard Python list to be JSON-serializable
# This is a safe and standard way to handle it.
serializable_weights = [w.tolist() for w in weights]

# Combine config and weights into a single dictionary
vectorizer_data_to_save = {
    "config": config,
    "weights": serializable_weights
}

# Save to a new .json file
with open("SavedVectorizers/bow_vectorizer.json", "w", encoding='utf-8') as f:
    json.dump(vectorizer_data_to_save, f, indent=4)

print("Vectorizer state saved successfully to 'SavedVectorizers/bow_vectorizer.json'")
# --- END OF SECTION 2 ---