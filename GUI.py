import dearpygui.dearpygui as dpg
import pickle
import keras
import os
import numpy as np

# Get the absolute path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the correct relative path
model_path = os.path.join(script_dir, "SavedModels", "rnn_seq_model_fold1.keras")

# Construct the direct path
#model_loaded = keras.models.load_model(model_path)

# Load the model
model = keras.models.load_model(model_path)


# Load your trained model and vectorizer
# model = keras.models.load_model("SavedModels/rnn_seq_model.keras")
#nya

vectorizer_path = os.path.join(script_dir, "SavedVectorizers", "rnn_vectorizer_fold1.pkl")

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)


# Define a function to analyze sentiment
def analyze_tweet(sender, app_data):
    tweet = dpg.get_value("tweet_input")

    # Vectorize the tweet
    vectorized_tweet = vectorizer([tweet])

    # Predict sentiment
    prediction = model.predict(vectorized_tweet)
    sentiment_index = np.argmax(prediction)

    sentiments = ['Positive', 'Negative', 'Neutral', 'Irrelevant']
    predicted_sentiment = sentiments[sentiment_index]

    # Update the result in the GUI
    dpg.set_value("result_output", f"Predicted Sentiment: {predicted_sentiment}")
dpg.create_context()

with dpg.window(label="Sentiment Analysis", width=500, height=400):
    dpg.add_text("Enter a tweet:")
    dpg.add_input_text(label="Tweet", tag="tweet_input", width=300)
    dpg.add_button(label="Analyze", callback=analyze_tweet)
    dpg.add_text("", tag="result_output")

dpg.create_viewport(title="Sentiment Analysis", width=500, height=400)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

# Create GUI elements
with dpg.window(label="Sentiment Analysis", width=400, height=300):
    dpg.add_text("Enter a tweet:")
    dpg.add_input_text(label="Tweet", tag="tweet_input", width=300)
    dpg.add_button(label="Analyze", callback=analyze_tweet)
    dpg.add_text("", tag="result_output")


# Start the GUI
dpg.start_dearpygui()

