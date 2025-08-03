import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import TextVectorization, BatchNormalization
from keras.src.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from data import training_labels, validation_labels
from keras.src.callbacks import EarlyStopping
import json
from data import max_length_95

# --- SECTION 1: LOAD PREPROCESSED DATA USING JSON ---
# Load preprocessed texts from the JSON file
with open('preprocessed_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    training_texts = data['custom_preprocessed_training_texts']
    validation_texts = data['custom_preprocessed_validation_texts']
# --- END OF SECTION 1 ---

print("Data loaded successfully from JSON!")

## VECTORIZATION
# Vectorization layer - Sequential approach
sequential_vectorizer = TextVectorization(
    max_tokens=20000,
    output_mode='int',
    output_sequence_length=max_length_95,
)

# Adapt the vectorizer to the training data
sequential_vectorizer.adapt(training_texts)

# Vectorize training and validation data
vectorized_training_texts = sequential_vectorizer(training_texts)
vectorized_validation_texts = sequential_vectorizer(validation_texts)

## 1: DEFINING THE CNN MODEL
model = Sequential([
    # Embedding Layer
    Embedding(input_dim=20000, output_dim=128, input_length=max_length_95),

    # 1D Convolutional Layers
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    BatchNormalization(),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),

    # Fully Connected Layer
    Dense(64, activation='relu'),
    Dropout(0.5),

    # Output Layer
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

## 3: TRAINING THE MODEL
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    vectorized_training_texts,
    training_labels,
    validation_data=(vectorized_validation_texts, validation_labels),
    epochs=10,
    batch_size=32,
    callbacks=[early_stopping]
)

## 4: EVALUATING THE MODEL
loss, accuracy = model.evaluate(vectorized_validation_texts, validation_labels)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

#

## 5: SAVING THE MODEL AND VECTORIZER
model.save("SavedModels/cnn_seq_model.keras")
print("\nModel saved successfully to 'SavedModels/cnn_seq_model.keras'")

# --- THIS IS THE CORRECT SAVING METHOD ---
print("Saving vectorizer state...")

config = sequential_vectorizer.get_config()
vocabulary = sequential_vectorizer.get_vocabulary()

vectorizer_data_to_save = {
    "config": config,
    "vocabulary": vocabulary # Save the vocabulary directly under its own key
}

with open("SavedVectorizers/cnn_vectorizer.json", "w", encoding='utf-8') as f:
    json.dump(vectorizer_data_to_save, f, indent=4)

print("Vectorizer state saved successfully to 'SavedVectorizers/cnn_vectorizer.json'")

