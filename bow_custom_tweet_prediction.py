import keras
import json
import numpy as np
from keras.src.layers import TextVectorization
from standardization import custom_standardization

# 1. Load the saved configuration and weights from the file
with open("SavedVectorizers/bow_vectorizer.json", "r", encoding='utf-8') as f:
    vectorizer_data = json.load(f)

# 2. Recreate the TextVectorization layer from the loaded configuration
bow_vectorizer = TextVectorization.from_config(vectorizer_data['config'])

# 3. Get the weights (vocabulary), convert them back to a NumPy array
weights_as_np = [np.array(w, dtype=np.str_) for w in vectorizer_data['weights']]


# 4. Explicitly adapt the layer to the loaded vocabulary.
# Forces the layer to build its internal lookup table (resolves an error that I had)
# The vocabulary is the first (and only) element in the weights list.
bow_vectorizer.adapt(weights_as_np[0])


print("Vectorizer reloaded and adapted successfully from JSON!")


# Load the pre-trained Keras model
model = keras.models.load_model("SavedModels/bow_seq_model_dropout.keras")
print("Model loaded successfully!\n")


# New tweets for classification
new_tweets = [
    "I absolutely love this movie, it was fantastic!",
    "This is the worst experience I've ever had.",
    "Meh, it was okay. Nothing special.",
    "I'm not sure how I feel about this..."
]

# Standardize the new tweets
custom_new_tweets = [custom_standardization(tweet) for tweet in new_tweets]

# Vectorize the standardized tweets (this will now work)
vectorized_custom_new_tweets = bow_vectorizer(custom_new_tweets)

# Make predictions
predictions = model.predict(vectorized_custom_new_tweets)

# Define the class labels
class_labels = ["Positive", "Negative", "Neutral", "Irrelevant"]


# Print the results
print("--- PREDICTION RESULTS ---")
for original_tweet, probs in zip(new_tweets, predictions):
    predicted_class_index = np.argmax(probs)
    predicted_label = class_labels[predicted_class_index]

    print(f"Tweet: {original_tweet}")
    print(f"Predicted Sentiment: {predicted_label}")

    prob_str = ", ".join([f"{label}: {p:.4f}" for label, p in zip(class_labels, probs)])
    print(f"Probabilities: [{prob_str}]\n")