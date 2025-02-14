import tensorflow as tf
from tensorflow.keras.models import Sequential



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
    # standardize = 'lower'  # Custom
    # Reminder that standardization isn't done yet!
)

# Adapting the TextVectorization layer to the training data
# Remember - we only want the vectorizer adapted to training data!
sequential_vectorizer.adapt(custom_preprocessed_training_texts1)

# Checking the vocabulary size and sample tokens (training)
# Remember - first two entries in the vocabulary are:
# the mask token (index 0)
# the OOV token (index 1)
vocabulary = sequential_vectorizer.get_vocabulary()
print(f"Vocabulary size: {len(vocabulary)}")
# print("Top 10 tokens:", vocabulary[:10])

# Some testing
sample_texts = ["I love programming.", "This is irrelevant.", "Hello Microsoft!"]
vectorized_texts = sequential_vectorizer(sample_texts)
# print("Original text:", sample_texts)
# print("Vectorized text:", vectorized_texts.numpy())

# Training data vectorization
vectorized_training_texts = sequential_vectorizer(custom_preprocessed_training_texts1)
# Validation data vectorization
vectorized_validation_texts = sequential_vectorizer(custom_preprocessed_validation_texts1)