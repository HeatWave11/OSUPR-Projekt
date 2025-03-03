import dearpygui.dearpygui as dpg
import pickle
import os
import numpy as np
import joblib # why exactly did I import this again??
import tensorflow as tf

# Paths to saved models and vectorizers
MODEL_DIR = "SavedModels"
VECTORIZER_DIR = "SavedVectorizers"

# Get lists of models and vectorizers
available_models = [f for f in os.listdir(MODEL_DIR) if f.endswith((".pkl", ".h5", ".keras"))]
available_vectorizers = [f for f in os.listdir(VECTORIZER_DIR) if f.endswith(".pkl")]

# Global variables for the selected model and vectorizer
loaded_model = None
loaded_vectorizer = None

def load_model(sender, app_data):
    """Load the selected model."""
    global loaded_model
    model_name = app_data
    model_path = os.path.join(MODEL_DIR, model_name)

    # Load model based on type
    if model_name.endswith(".pkl"):
        loaded_model = joblib.load(model_path)
    elif model_name.endswith((".h5", ".keras")):
        loaded_model = tf.keras.models.load_model(model_path)

    # Update GUI
    dpg.set_value("selected_model_display", f"Loaded: {model_name}")
    print(f"Loaded model: {model_name}")

def load_vectorizer(sender, app_data):
    """Load the selected vectorizer."""
    global loaded_vectorizer
    vectorizer_name = app_data
    vectorizer_path = os.path.join(VECTORIZER_DIR, vectorizer_name)

    # Load vectorizer
    with open(vectorizer_path, "rb") as f:
        loaded_vectorizer = pickle.load(f)

    # Update GUI
    dpg.set_value("vectorizer_display", f"Loaded: {vectorizer_name}")
    print(f"Loaded vectorizer: {vectorizer_name}")

def analyze_tweet(sender, app_data):
    """Predict sentiment of the input tweet."""
    if not loaded_model:
        dpg.set_value("result_output", "No model selected!")
        return
    if not loaded_vectorizer:
        dpg.set_value("result_output", "No vectorizer selected!")
        return

    tweet = dpg.get_value("tweet_input")
    vectorized_tweet = loaded_vectorizer([tweet])  # Transform input text

    # Predict sentiment
    prediction = loaded_model.predict(vectorized_tweet)
    sentiment_index = np.argmax(prediction)
    sentiments = ['Positive', 'Negative', 'Neutral', 'Irrelevant']
    predicted_sentiment = sentiments[sentiment_index]

    # Update result in GUI
    dpg.set_value("result_output", f"Predicted Sentiment: {predicted_sentiment}")

def exit_callback():
    """Exit GUI."""
    dpg.stop_dearpygui()

# GUI Setup
dpg.create_context()
dpg.create_viewport(title="Sentiment Analysis", width=500, height=450)

with dpg.window(label="Sentiment Analysis", width=500, height=450):
    dpg.add_text("Select a model:")
    dpg.add_combo(available_models, callback=load_model, default_value=available_models[0] if available_models else "No Models Found", tag="model_selector")
    dpg.add_text("Selected Model: ")
    dpg.add_text("", tag="selected_model_display")

    dpg.add_text("\nSelect a vectorizer:")
    dpg.add_combo(available_vectorizers, callback=load_vectorizer, default_value=available_vectorizers[0] if available_vectorizers else "No Vectorizers Found", tag="vectorizer_selector")
    dpg.add_text("Selected Vectorizer: ")
    dpg.add_text("", tag="vectorizer_display")

    dpg.add_text("\nEnter a tweet:")
    dpg.add_input_text(label="Tweet", tag="tweet_input", width=300, on_enter=True, callback=analyze_tweet)
    dpg.add_button(label="Analyze", callback=analyze_tweet)
    dpg.add_button(label="Exit", callback=exit_callback, pos=(100, 120))
    dpg.add_text("", tag="result_output")

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

