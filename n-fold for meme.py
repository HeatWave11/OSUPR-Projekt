import numpy as np
import os
import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import TextVectorization, BatchNormalization
from keras.src.layers import Embedding, LSTM, Dense, Dropout
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import json

# Just to test if my approach to the problem is correct.

# Import BOTH sets of labels from your data file
from data import training_labels, validation_labels, max_length_95

# Load preprocessed texts from the JSON file
with open('preprocessed_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    # Combine training and validation texts
    all_texts = np.array(data['custom_preprocessed_training_texts'] + data['custom_preprocessed_validation_texts'])

    # Combine training and validation labels to match the texts
    all_labels = np.concatenate((training_labels, validation_labels))

# Check that the lengths now match
assert len(all_texts) == len(all_labels), "Mismatch between number of texts and labels!"
print(f"Data loaded successfully! Total samples: {len(all_texts)}")



# Define K-Fold Cross Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Create directories if they don't exist
os.makedirs("SavedModels", exist_ok=True)
os.makedirs("SavedVectorizers", exist_ok=True)

fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(all_texts)):
    print(f"\n🔹 Training Fold {fold+1}/{n_splits} 🔹")


    train_texts, val_texts = all_texts[train_idx], all_texts[val_idx]
    train_labels, val_labels = all_labels[train_idx], all_labels[val_idx]

    # Text Vectorization for the current fold
    vectorizer = TextVectorization(
        max_tokens=20000,
        output_mode='int',
        output_sequence_length=max_length_95
    )
    vectorizer.adapt(train_texts)

    vectorized_train_texts = vectorizer(train_texts)
    vectorized_val_texts = vectorizer(val_texts)

    # Define the LSTM model
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

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    model.fit(
        vectorized_train_texts, train_labels,
        validation_data=(vectorized_val_texts, val_labels),
        epochs=10,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(vectorized_val_texts, val_labels)
    fold_accuracies.append(accuracy)
    print(f"✅ Fold {fold+1} Accuracy: {accuracy:.4f}")

    # Save the model for this fold
    model_save_path = f"SavedModels/rnn_seq_model_fold{fold+1}.keras"
    model.save(model_save_path)
    print(f"📌 Model saved at: {model_save_path}")


    config = vectorizer.get_config()
    vocabulary = vectorizer.get_vocabulary()
    vectorizer_data_to_save = {
        "config": config,
        "vocabulary": vocabulary
    }

    vectorizer_save_path = f"SavedVectorizers/rnn_vectorizer_fold{fold+1}.json"
    with open(vectorizer_save_path, "w", encoding='utf-8') as f:
        json.dump(vectorizer_data_to_save, f)
    print(f"📌 Vectorizer saved at: {vectorizer_save_path}")



# Print final results
print(f"\n📊 Final Cross-Validation Results ({n_splits} folds):")
print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(fold_accuracies):.4f}")