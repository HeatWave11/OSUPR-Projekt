import pandas as pd
import tensorflow as tf
import keras
from keras.src.layers import TextVectorization

# Klemen Požlep
# Code/Comments will be made in English so I'll have an easier time with documentation

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

# Vectorization layer tweaking
vectorizer = TextVectorization(
    max_tokens = 20000,  # Maximum number of unique words to consider
    output_mode = 'int',  # Output integers (token indices)
    output_sequence_length = 20,  # For Padding/truncating all sequences to this length
    standardize = 'lower_and_strip_punctuation'  # Convert to lowercase and remove punctuation
)

# Adapting the TextVectorization layer to the training data
# Remember - we only want the vectorizer adapted to training data!
vectorizer.adapt(training_texts)

# Checking the vocabulary size and sample tokens (training)
# Remember - we only want the vectorizer adapted to training data!!
vocabulary = vectorizer.get_vocabulary()
print(f"Vocabulary size: {len(vocabulary)}")
print("Top 10 tokens:", vocabulary[:10])

# Some testing
sample_texts = ["I love programming.", "This is irrelevant.", "Hello Microsoft!"]
vectorized_texts = vectorizer(sample_texts)
print("Original text:", sample_texts)
print("Vectorized text:", vectorized_texts.numpy())





