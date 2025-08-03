import dearpygui.dearpygui as dpg
import os
import json
import keras
import joblib
import numpy as np
from sklearn.metrics import classification_report
from keras.src.layers import TextVectorization

# --- IMPORTANT: Make sure these files are in the same directory or accessible ---
from standardization import custom_standardization
from data import validation_texts, validation_labels

# --- 1. MODEL AND VECTORIZER REGISTRY ---
MODEL_REGISTRY = {
    "SVM (SVC)": {"model_path": os.path.join("SavedModels", "svm_model.joblib"), "vectorizer_path": os.path.join("SavedVectorizers", "svm_tfidf_vectorizer.joblib"),"type": "sklearn"},
    "CNN (Keras)": {"model_path": os.path.join("SavedModels", "cnn_seq_model.keras"),"vectorizer_path": os.path.join("SavedVectorizers", "cnn_vectorizer.json"),"type": "keras"},
    "Bag-of-Words (Keras)": {"model_path": os.path.join("SavedModels", "bow_seq_model_dropout.keras"),"vectorizer_path": os.path.join("SavedVectorizers", "bow_vectorizer.json"),"type": "keras"},
    "RNN (Fold 1)": {"model_path": os.path.join("SavedModels", "rnn_seq_model_fold1.keras"),"vectorizer_path": os.path.join("SavedVectorizers", "rnn_vectorizer_fold1.json"),"type": "keras"},
    "RNN (Fold 2)": {"model_path": os.path.join("SavedModels", "rnn_seq_model_fold2.keras"),"vectorizer_path": os.path.join("SavedVectorizers", "rnn_vectorizer_fold2.json"),"type": "keras"},
    "RNN (Fold 3)": {"model_path": os.path.join("SavedModels", "rnn_seq_model_fold3.keras"),"vectorizer_path": os.path.join("SavedVectorizers", "rnn_vectorizer_fold3.json"),"type": "keras"},
    "RNN (Fold 4)": {"model_path": os.path.join("SavedModels", "rnn_seq_model_fold4.keras"),"vectorizer_path": os.path.join("SavedVectorizers", "rnn_vectorizer_fold4.json"),"type": "keras"},
    "RNN (Single)": {
        "model_path": os.path.join("SavedModels", "rnn_seq_model.keras"),
        "vectorizer_path": os.path.join("SavedVectorizers", "rnn_vectorizer.json"),
        "type": "keras"
    },
    "RNN (Fold 5)": {"model_path": os.path.join("SavedModels", "rnn_seq_model_fold5.keras"),"vectorizer_path": os.path.join("SavedVectorizers", "rnn_vectorizer_fold5.json"),"type": "keras"}
}

# --- 2. GLOBAL VARIABLES FOR STATE MANAGEMENT ---
current_model = None
current_vectorizer = None
current_model_type = None
CLASS_LABELS = ['Positive', 'Negative', 'Neutral', 'Irrelevant']

print("Preprocessing validation data for evaluation...")
processed_validation_texts = [custom_standardization(text) for text in validation_texts]
print("Preprocessing complete.")

# --- 3. GUI CALLBACK FUNCTIONS ---
# (This entire section remains exactly the same, no changes needed here)
def load_model_callback(sender, app_data):
    """Called when a new model is selected from the dropdown."""
    global current_model, current_vectorizer, current_model_type
    model_name = app_data
    dpg.set_value("status_text", f"Loading {model_name}...")
    if model_name not in MODEL_REGISTRY:
        dpg.set_value("status_text", f"Error: Model '{model_name}' not found in registry.")
        return
    model_info = MODEL_REGISTRY[model_name]
    model_path = model_info["model_path"]
    vectorizer_path = model_info["vectorizer_path"]
    current_model_type = model_info["type"]
    try:
        if current_model_type == "sklearn":
            current_model = joblib.load(model_path)
            current_vectorizer = joblib.load(vectorizer_path)
        elif current_model_type == "keras":
            current_model = keras.models.load_model(model_path)
            with open(vectorizer_path, "r", encoding='utf-8') as f:
                vectorizer_data = json.load(f)
            current_vectorizer = TextVectorization.from_config(vectorizer_data['config'])
            current_vectorizer.adapt(vectorizer_data['vocabulary'])
        dpg.set_value("status_text", f"✅ {model_name} loaded successfully!")
        evaluate_and_update_gui()
    except FileNotFoundError:
        dpg.set_value("status_text", f"❌ Error: File not found for {model_name}. Did you train it?")
        dpg.set_value("stats_output", "Model files not found. Please train the selected model first.")
    except Exception as e:
        dpg.set_value("status_text", f"❌ An error occurred: {e}")
        dpg.set_value("stats_output", f"An unexpected error occurred during loading:\n{e}")

def evaluate_and_update_gui():
    """Evaluates the currently loaded model and updates the stats display."""
    if not current_model or not current_vectorizer: return
    dpg.set_value("stats_output", "Evaluating model on validation set...")
    vectorized_val_texts = current_vectorizer.transform(processed_validation_texts) if current_model_type == "sklearn" else current_vectorizer(processed_validation_texts)
    if current_model_type == "sklearn":
        predictions = current_model.predict(vectorized_val_texts)
    else:
        predictions_proba = current_model.predict(vectorized_val_texts)
        predictions = np.argmax(predictions_proba, axis=1)
    report = classification_report(validation_labels, predictions, target_names=CLASS_LABELS, zero_division=0)
    dpg.set_value("stats_output", report)

def analyze_tweet_callback(sender, app_data):
    """Called when the 'Analyze' button is clicked for a single tweet."""
    if not current_model:
        dpg.set_value("result_output", "Please select and load a model first.")
        return
    tweet = dpg.get_value("tweet_input")
    if not tweet:
        dpg.set_value("result_output", "Please enter a tweet.")
        return
    processed_tweet = custom_standardization(tweet)
    vectorized_tweet = current_vectorizer([processed_tweet])
    if current_model_type == "sklearn":
        prediction_index = current_model.predict(vectorized_tweet)[0]
    else:
        prediction_proba = current_model.predict(vectorized_tweet)
        prediction_index = np.argmax(prediction_proba)
    predicted_sentiment = CLASS_LABELS[prediction_index]
    dpg.set_value("result_output", f"Predicted Sentiment: {predicted_sentiment}")

# --- 4. GUI SETUP ---
dpg.create_context()

# --- NEW, ROBUST FONT LOADER ---
with dpg.font_registry():
    # First, try to load a good monospace font from a standard Windows location
    # This is perfect for the classification report
    font_path = "C:/Windows/Fonts/consola.ttf"
    if os.path.exists(font_path):
        print(f"Loading font: {font_path}")
        default_font = dpg.add_font(font_path, 16)
    else:
        # If the preferred font isn't found, load the default built-in font
        # This is the correct way to get the default font handle
        print("Consolas font not found. Using default application font.")
        default_font = dpg.add_font()
# --- END OF FONT LOADER ---


with dpg.window(label="Sentiment Analysis Dashboard", width=800, height=700, tag="main_window"):
    dpg.add_text("Model Selection and Evaluation")
    dpg.add_separator()
    dpg.add_combo(list(MODEL_REGISTRY.keys()), label="Select Model", callback=load_model_callback, width=300)
    dpg.add_text("", tag="status_text")
    dpg.add_separator()
    dpg.add_text("Validation Set Performance:")
    with dpg.child_window(height=300):
        dpg.add_text("Select a model to see its performance statistics.", tag="stats_output")
    dpg.add_separator()
    dpg.add_text("\nLive Tweet Prediction")
    dpg.add_separator()
    dpg.add_input_text(label="Enter Tweet Here", tag="tweet_input", width=-1)
    dpg.add_button(label="Analyze", callback=analyze_tweet_callback)
    dpg.add_text("", tag="result_output")

# Bind the loaded font to the main window so all text uses it
dpg.bind_font(default_font)

dpg.create_viewport(title="Sentiment Analysis Dashboard", width=800, height=700)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()