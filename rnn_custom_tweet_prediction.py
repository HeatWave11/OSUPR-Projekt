import keras
import json  # <-- Import JSON
import numpy as np
from sklearn.metrics import classification_report
from keras.src.layers import TextVectorization #<-- Import TextVectorization

from standardization import custom_standardization

# --- REPLACEMENT FOR PICKLE ---
# This block replaces 'with open(...) as f: pickle.load(f)'

# 1. Load the saved configuration and weights from the file
with open("SavedVectorizers/bow_vectorizer.json", "r", encoding='utf-8') as f:
    vectorizer_data = json.load(f)

# 2. Recreate the TextVectorization layer from the loaded configuration
bow_vectorizer = TextVectorization.from_config(vectorizer_data['config'])

# 3. Get the weights (vocabulary), convert them back to a NumPy array, and set them
# The vocabulary must be a NumPy array of string type (dtype=np.str_)
weights_as_np = [np.array(w, dtype=np.str_) for w in vectorizer_data['weights']]
bow_vectorizer.set_weights(weights_as_np)

print("Vectorizer reloaded successfully from JSON!")
# --- END OF REPLACEMENT BLOCK ---


# Load the pre-trained Keras model (this is already safe)
model = keras.models.load_model("SavedModels/bow_seq_model_dropout.keras") # Make sure the model name matches what you saved
print("Model loaded successfully!")

# New tweets to classify
new_tweets = [
    "I absolutely love this movie, it was fantastic! I really love love love it so much",
    "This is the worst experience I've ever had. Absolutly horrible and shit.",
    "Meh, it was okay. Nothing special. I don't care about it at all.",
    "I'm not sure how I feel about this...",
    "Now the President is slapping Americans in the face that he really did commit an unlawful act after his acquittal! From Discover on Google vanityfair.com/news/2020/02/t…"
]

# Standardize the new tweets using your custom function
# Your standardization function should accept a single string at a time.
# We will loop through the list and apply it to each tweet.
custom_new_tweets = [custom_standardization(tweet) for tweet in new_tweets]
print("\nStandardized Tweets:", custom_new_tweets)


# Vectorize the standardized tweets
vectorized_custom_new_tweets = bow_vectorizer(custom_new_tweets)

# Make predictions
predictions = model.predict(vectorized_custom_new_tweets)

# Get the index of the highest probability for each prediction
predicted_classes = np.argmax(predictions, axis=1)

# Define the class labels
class_labels = ["Positive", "Negative", "Neutral", "Irrelevant"]

# Print the results for each tweet
print("\n--- PREDICTION RESULTS ---")
for tweet, pred_class_index, probs in zip(new_tweets, predicted_classes, predictions):
    predicted_label = class_labels[pred_class_index]
    print(f"Tweet: {tweet}")
    print(f"Predicted Sentiment: {predicted_label}")
    # Formatting the probabilities for cleaner output
    prob_str = ", ".join([f"{label}: {p:.4f}" for label, p in zip(class_labels, probs)])
    print(f"Probabilities: [{prob_str}]\n")


# Optional: Evaluate with a classification report if you have ground truth labels
# val_labels_new_tweets = [0, 1, 2, 3, 3] # Example labels for the tweets above
# print("\n--- CLASSIFICATION REPORT ---")
# print(classification_report(val_labels_new_tweets, predicted_classes, target_names=class_labels))