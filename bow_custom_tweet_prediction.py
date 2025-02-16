import keras
import pickle
import pandas as pd
from standardization import custom_standardization

with open("SavedVectorizers/bow_vectorizer.pkl", "rb") as f:
    bow_vectorizer = pickle.load(f)

model = keras.models.load_model("SavedModels/bow_seq_model_dropout.keras")

new_tweets = [
    "I absolutely love this movie, it was fantastic!",
    "This is the worst experience I've ever had.",
    "Meh, it was okay. Nothing special.",
    "I'm not sure how I feel about this..."
]
# is a list!!
# Standardize the new tweets using the imported function
# Ensure new_tweets is a list of strings
custom_new_tweets = custom_standardization([str(tweet) for tweet in new_tweets])
# had to add the list line in standardization for this to work

# Vectorize the new tweets
vectorized_custom_new_tweets = bow_vectorizer(custom_new_tweets)

predictions = model.predict(vectorized_custom_new_tweets)

import numpy as np

# Get the index of the highest probability (most likely class)
predicted_classes = np.argmax(predictions, axis=1)

# Define the class labels (make sure they match your dataset's labels)
class_labels = ["Positive", "Negative", "Neutral", "Irrelevant"]

# Convert numerical predictions to class labels
predicted_labels = [class_labels[i] for i in predicted_classes]

# Print results
for tweet, label in zip(new_tweets, predicted_labels):
    print(f"Tweet: {tweet}\nPredicted Sentiment: {label}\n")

import numpy as np

# Get prediction probabilities
predictions = model.predict(vectorized_custom_new_tweets)

# Print tweet, predicted class, and probabilities
for tweet, prob in zip(custom_new_tweets, predictions):
    predicted_class = np.argmax(prob)
    predicted_label = class_labels[predicted_class]
    print(f"Tweet: {tweet}\nPredicted Sentiment: {predicted_label}\nProbabilities: {prob}\n")
