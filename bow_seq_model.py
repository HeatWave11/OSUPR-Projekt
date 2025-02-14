import os

import keras.src
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# for hiding those tensorflow warnings/info
# reminder that if I don't manage to use my Vega 56 GPU, I should try build tensorflow with my CPU optimizations enabled

from keras.src.layers import TextVectorization, Dense
from keras.src.models import Sequential
import tensorflow as tf
import numpy as np
# import standardization
import data
import pickle

# bow_training_texts = standardization.custom_preprocessed_training_texts
# bow_validation_texts = standardization.custom_preprocessed_validation_texts

bow_training_labels = data.training_labels
bow_validation_labels = data.validation_labels

# Load preprocessed texts
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    bow_training_texts = data['custom_preprocessed_training_texts']
    bow_validation_texts = data['custom_preprocessed_validation_texts']

bow_training_texts = np.array(bow_training_texts).astype(str)
bow_validation_texts = np.array(bow_validation_texts).astype(str)

# Convert training labels if necessary (already in NumPy array)
bow_training_labels = np.array(bow_training_labels, dtype=np.int32)
bow_validation_labels = np.array(bow_validation_labels, dtype=np.int32)  # Convert validation labels as well

#bow_training_texts_list = bow_training_texts.tolist()
print("First 5 training texts:", bow_training_texts[:5])
print("First 5 training labels:", bow_training_labels[:5])
print("Training texts dtype:", type(bow_training_texts))
print("Training labels dtype:", type(bow_training_labels))

# Instead of using .shape, use len() for lists
print("Length of training texts:", len(bow_training_texts))

print("Type of training texts:", type(bow_training_texts[0]))

# Convert training texts to standard Python strings
# bow_training_texts = bow_training_texts.astype(str).tolist()
# bow_validation_texts = bow_validation_texts.astype(str).tolist()


# FIRST ROUGH EXAMPLE: USING BOW WITH A KERAS SEQUENTIAL MODEL (simple feedforward neural network)

# 1: Vectorization layer - BoW approach (Word order doesn't matter - for others)
bow_vectorizer = TextVectorization(
    max_tokens=10000,  # Limit the vocabulary size to the top 10,000 words
    output_mode='tf_idf',  # Output multi-hot encoded vectors
    standardize=None,  # Convert to lowercase and remove punctuation
    # 'lower_and_strip_punctuation'
)

# 2: Adapt the vectorizer to the training data
bow_vectorizer.adapt(bow_training_texts)
# Inspect the vocabulary size and tokens
# vocabulary = bow_vectorizer.get_vocabulary()
#print("Vocabulary size:", len(vocabulary))
#print("First 10 tokens in the vocabulary:", vocabulary[:10])
#print("Last 10 tokens in the vocabulary:", vocabulary[-10:])

# Vectorize the training texts again
vectorized_training_texts = bow_vectorizer(np.array(bow_training_texts))

# Inspect the shape and first vectorized text
print("Shape of vectorized training texts:", vectorized_training_texts.shape)
print("First vectorized training text:", vectorized_training_texts[0].numpy())

# 3: Build the model
# Remember: Not all Sequential models care about word order!
# Look at your Obsidian notes if you're confused about this in the future
# Here we have BoW + Dense which is NOT sequnce based
# In order for sequential models to be sequence based they'd need to have layers like LSTM, CNN or Transformers
model = Sequential([
    keras.src.layers.InputLayer(shape=(), dtype= tf.string),
    bow_vectorizer,  # TextVectorization layer (outputs multi-hot encoded vectors)
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),  # Hidden layer
    Dense(4, activation='softmax')  # Output layer for 4 classes (Positive, Negative, Neutral, Irrelevant)
])

# Step 4: Compiling the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("nya")
# 5: Print model summary
model.summary()

# Convert training and validation texts to a standard Python list
# bow_training_texts_list = bow_training_texts.tolist()
# bow_validation_texts_list = bow_validation_texts.tolist()

# 6: Fit the model
history = model.fit(
    x=bow_training_texts,  # Use converted NumPy array
    y=bow_training_labels,
    validation_data=(bow_validation_texts, bow_validation_labels),
    epochs=10,
    batch_size=32
)



# 7: Evaluate the model
loss, accuracy = model.evaluate(bow_validation_texts, bow_validation_labels)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
# I got Validation Loss: 0.2136, Validation Accuracy: 0.9630 which isn't terrible