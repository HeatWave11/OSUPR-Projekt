import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# for hiding those tensorflow warnings/info
# reminder that if I don't manage to use my Vega 56 GPU, I should try build tensorflow with my CPU optimizations enabled

import pandas as pd
import tensorflow as tf
import keras
import numpy as np
from keras.src.layers import TextVectorization, Dense
from keras.src.models import Sequential

# Klemen Požlep
# Code/Comments will be made in English so I'll have an easier time with documentation, textbooks and help from LLMs

# Relevant notes
# do we need stemming
# are we going to use sequential or bag-of-words approach
# random shuffle?

# BOW for logistic regression, SVM, Naive Bayes?
# Sequential approach for Deep learning models (?)

# Data import
column_names = ['Text ID', 'Entity', 'Sentiment', 'Tweet']

training_dataset = pd.read_csv('E:/School 2425/OSUPR/Projekt/archive/twitter_training.csv',
                               header=None, names=column_names)
validation_dataset = pd.read_csv('E:/School 2425/OSUPR/Projekt/archive/twitter_validation.csv',
                                 header=None, names=column_names)

print("Column names:", training_dataset.columns)
print("Column names:", validation_dataset.columns)

# Extracting the "Message/Tweet' column
# we also made sure all entries are strings
training_texts = training_dataset['Tweet'].astype(str).values
validation_texts = validation_dataset['Tweet'].astype(str).values

# Mapping sentiment labels to numeric values
sentiment_mapping = {'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3}
training_labels = training_dataset['Sentiment'].map(sentiment_mapping).values
validation_labels = validation_dataset['Sentiment'].map(sentiment_mapping).values

# Verifying the data (training)
print("Sample texts:", training_texts[:5])
print("Sample labels:", training_labels[:5])

# Verifying the data (validation)
print("Sample texts:", validation_texts[:5])
print("Sample labels:", validation_labels[:5])

# For finding out what our output_sequence_length should realistically be
# We'll tune this hyperparameter more a bit later (maybe)
# Calculating tweet lengths
tweet_lengths = [len(text.split()) for text in training_texts]

# Distribution Plot
import matplotlib.pyplot as plt
plt.hist(tweet_lengths, bins=50)
plt.title('Distribution of Tweet Lengths')
plt.xlabel('Tweet Length (words)')
plt.ylabel('Frequency')
plt.show()

# Finding the 95th percentile
max_length_95 = int(np.percentile(tweet_lengths, 95))
print(f"95th Percentile tweet length is: {max_length_95}")

# Vectorization layer - Sequential approach (Word order matters - for RNNs and Transformers)
sequential_vectorizer = TextVectorization(
    max_tokens = 20000,  # Maximum number of unique words to consider
    output_mode = 'int',  # Output integers (token indices)
    output_sequence_length = max_length_95,  # For Padding/truncating all sequences to this length
    # Important parameter than we need to think hard about how to set
    # Too long isn't good because it will be wasteful
    # Too short isn't good either because it will truncate longer tweets
    # Set it to 95th percentile for now!
    standardize = 'lower_and_strip_punctuation'  # Convert to lowercase and remove punctuation
    # Reminder that standardization isn't done yet!
)



# Adapting the TextVectorization layer to the training data
# Remember - we only want the vectorizer adapted to training data!
sequential_vectorizer.adapt(training_texts)

# Checking the vocabulary size and sample tokens (training)
# Remember - first two entries in the vocabulary are:
# the mask token (index 0)
# the OOV token (index 1)
vocabulary = sequential_vectorizer.get_vocabulary()
print(f"Vocabulary size: {len(vocabulary)}")
print("Top 10 tokens:", vocabulary[:10])

# Some testing
sample_texts = ["I love programming.", "This is irrelevant.", "Hello Microsoft!"]
vectorized_texts = sequential_vectorizer(sample_texts)
print("Original text:", sample_texts)
print("Vectorized text:", vectorized_texts.numpy())

# Training data vectorization
vectorized_training_texts = sequential_vectorizer(training_texts)
# Validation data vectorization
vectorized_validation_texts = sequential_vectorizer(validation_texts)






# FIRST ROUGH EXAMPLE: USING BOW WITH A KERAS SEQUENTIAL MODEL (simple feedforward neural network)

# 1: Vectorization layer - BoW approach (Word order doesn't matter - for others)
bow_vectorizer = TextVectorization(
    max_tokens=10000,  # Limit the vocabulary size to the top 10,000 words
    output_mode='multi_hot',  # Output multi-hot encoded vectors
    standardize='lower_and_strip_punctuation',  # Convert to lowercase and remove punctuation
)

# 2: Adapt the vectorizer to the training data
bow_vectorizer.adapt(training_texts)

# 3: Build the model
# Remember: Not all Sequential models
# Look at your obsidian notes if you're confused about this in the future
# Here we have BoW + Dense which is NOT sequnce based
# In order for sequential models to be sequence based they'd need to have layers like LSTM, CNN or Transformers
model = Sequential([
    bow_vectorizer,  # TextVectorization layer (outputs multi-hot encoded vectors)
    Dense(64, activation='relu'),  # Hidden layer
    Dense(4, activation='softmax')  # Output layer for 4 classes (Positive, Negative, Neutral, Irrelevant)
])

# Step 4: Compiling the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5: Print model summary
model.summary()

# 6: Train the model
history = model.fit(
    x=training_texts,  # Raw text inputs!
    y=training_labels,
    validation_data=(validation_texts, validation_labels),
    epochs=10,
    batch_size=32
)

# 7: Evaluate the model
loss, accuracy = model.evaluate(validation_texts, validation_labels)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
# I got Validation Loss: 0.2136, Validation Accuracy: 0.9630 which isn't terrible


