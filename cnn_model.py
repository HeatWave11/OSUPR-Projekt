import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import TextVectorization, BatchNormalization
from keras.src.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from data import training_labels, validation_labels
from keras.src.callbacks import EarlyStopping
import pickle

from data import max_length_95

# Load preprocessed texts
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    training_texts = data['custom_preprocessed_training_texts']
    validation_texts = data['custom_preprocessed_validation_texts']




## VECTORIZATION

# Vectorization layer - Sequential approach (Word order matters - for CNNs too)
sequential_vectorizer = TextVectorization(
    max_tokens=20000,
    output_mode='int',
    output_sequence_length=max_length_95,
)

vocab = sequential_vectorizer.get_vocabulary()

# Print the specific problem area
print("Problematic characters:", vocab[84921:84925])

# Adapt the vectorizer to the training data
sequential_vectorizer.adapt(training_texts)

# Vectorize training and validation data
vectorized_training_texts = sequential_vectorizer(training_texts)
vectorized_validation_texts = sequential_vectorizer(validation_texts)

## 1: DEFINING THE CNN MODEL
model = Sequential([
    # Embedding Layer
    Embedding(input_dim=20000, output_dim=128, input_length=max_length_95),

    # 1D Convolutional Layer
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    BatchNormalization(),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),  # This layer reduces the output dimensions

    # Fully Connected Layer
    Dense(64, activation='relu'),
    Dropout(0.5),

    # Output Layer (4 classes)
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

## 3: TRAINING THE MODEL
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
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

## 5: SAVING THE MODEL AND VECTORIZER
model.save("SavedModels/cnn_seq_model.keras")

with open("SavedVectorizers/cnn_vectorizer.pkl", "wb") as f:
    pickle.dump(sequential_vectorizer, f)
