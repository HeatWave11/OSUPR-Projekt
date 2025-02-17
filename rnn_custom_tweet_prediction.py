import keras
import pickle
from sklearn.metrics import f1_score, classification_report

from standardization import custom_standardization

with open("SavedVectorizers/rnn_vectorizer.pkl", "rb") as f:
    bow_vectorizer = pickle.load(f)

model = keras.models.load_model("SavedModels/rnn_seq_model.keras")

new_tweets = [
    "I absolutely love this movie, it was fantastic! I really love love love it so much",
    "This is the worst experience I've ever had. Absolutly horrible and shit.",
    "Meh, it was okay. Nothing special. I don't care about it at all.",
    "I'm not sure how I feel about this..."
    "Now the President is slapping Americans in the face that he really did commit an unlawful act after his  acquittal! From Discover on Google vanityfair.com/news/2020/02/t…"
]

val_labels_new_tweets = [0,1,2,3]
# is a list!!
# Standardize the new tweets using the imported function
# Ensure new_tweets is a list of strings
custom_new_tweets = custom_standardization([str(tweet) for tweet in new_tweets])
print(custom_new_tweets)

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

import numpy as np

# Get prediction probabilities
predictions = model.predict(vectorized_custom_new_tweets)

# Print tweet, predicted class, and probabilities
for tweet, prob in zip(new_tweets, predictions):
    predicted_class = np.argmax(prob)
    predicted_label = class_labels[predicted_class]
    print(f"Tweet: {tweet}\nPredicted Sentiment: {predicted_label}\nProbabilities: {prob}\n")

# Ensure your predicted classes are correctly formatted
print(classification_report(val_labels_new_tweets, predicted_classes, target_names=class_labels))
