import numpy as np
import os
import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import TextVectorization, BatchNormalization
from keras.src.layers import Embedding, LSTM, Dense, Dropout
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import pickle

from data import training_labels, validation_labels, max_length_95

# Load preprocessed texts
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    all_texts = np.array(data['custom_preprocessed_training_texts'])  # Convert to NumPy array
    all_labels = np.array(training_labels)  # Convert labels to NumPy array

# Define N-Fold Cross Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Create directories to store models and vectorizers
os.makedirs("SavedModels", exist_ok=True)
os.makedirs("SavedVectorizers", exist_ok=True)

# Initialize list to store results
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(all_texts)):
    print(f"\n🔹 Training Fold {fold+1}/{n_splits} 🔹")

    # Split data
    train_texts, val_texts = all_texts[train_idx], all_texts[val_idx]
    train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

    # Text Vectorization
    vectorizer = TextVectorization(
        max_tokens=20000,
        output_mode='int',
        output_sequence_length=max_length_95
    )
    vectorizer.adapt(train_texts)

    vectorized_train_texts = vectorizer(train_texts)
    vectorized_val_texts = vectorizer(val_texts)

    # Define model
    model = Sequential([
        Embedding(input_dim=20000, output_dim=128, input_length=max_length_95),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        LSTM(32, return_sequences=False)    ,
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train model
    history = model.fit(
        vectorized_train_texts, train_labels,
        validation_data=(vectorized_val_texts, val_labels),
        epochs=10,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate model
    loss, accuracy = model.evaluate(vectorized_val_texts, val_labels)
    fold_accuracies.append(accuracy)
    print(f"✅ Fold {fold+1} Accuracy: {accuracy:.4f}")

    # Save the model
    model_save_path = f"SavedModels/rnn_seq_model_fold{fold+1}.keras"
    model.save(model_save_path)
    print(f"📌 Model saved at: {model_save_path}")

    # Save the vectorizer
    vectorizer_save_path = f"SavedVectorizers/rnn_vectorizer_fold{fold+1}.pkl"
    with open(vectorizer_save_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"📌 Vectorizer saved at: {vectorizer_save_path}")

# Print final cross-validation results
print(f"\n📊 Cross-Validation Results:")
print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"Standard Deviation: {np.std(fold_accuracies):.4f}")
print(f"Some other metric that could actually turn out to be quite useful: {np.std()}")
