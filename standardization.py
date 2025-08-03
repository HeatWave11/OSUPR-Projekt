import re
import nltk
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Assuming 'data.py' contains your training and validation sets
from data import training_texts, training_labels, validation_texts, validation_labels


# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


# --- MODIFIED STOP WORD HANDLING (THE KEY FIX) ---

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# 1. Get the original list of English stop words
original_stop_words = set(stopwords.words('english'))

# 2. Define a set of negation and contrast words that are important for sentiment.
#    We will REMOVE these from the stop word list so they are KEPT in the final text.
negation_words = {
    'not', 'no', 'nor', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
    "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
    'but', 'against'
}

# 3. Create the final stop words list by removing the negations from the original list.
stop_words = original_stop_words - negation_words

# --- END OF MODIFICATION ---


def custom_standardization(text):
    """Processes a single string for text cleaning and standardization."""
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize the text
    words = word_tokenize(text)

    # Get POS tags
    tagged_words = nltk.pos_tag(words)

    processed_words = []
    for word, tag in tagged_words:
        # Use the NEW, corrected stop_words list here
        if word not in stop_words:
            if tag.startswith('J'):
                pos = 'a'
            elif tag.startswith('V'):
                pos = 'v'
            elif tag.startswith('N'):
                pos = 'n'
            elif tag.startswith('R'):
                pos = 'r'
            else:
                pos = 'n'

            processed_words.append(lemmatizer.lemmatize(word, pos))

    return ' '.join(processed_words)

# Main execution block
if __name__ == "__main__":
    print("Standardization script running...")
    print(f"Using a custom stop word list of size: {len(stop_words)}")
    # A quick check to prove 'not' is no longer a stop word
    if 'not' not in stop_words:
        print("Check successful: 'not' is correctly being kept for sentiment analysis.")

    print("\nPreprocessing training data...")
    custom_preprocessed_training_texts = [
        custom_standardization(text) for text in tqdm(training_texts, desc="Processing training texts")
    ]

    print("Preprocessing validation data...")
    custom_preprocessed_validation_texts = [
        custom_standardization(text) for text in tqdm(validation_texts, desc="Processing validation texts")
    ]

    # Create a dictionary to hold your data
    data_to_save = {
        'custom_preprocessed_training_texts': custom_preprocessed_training_texts,
        'custom_preprocessed_validation_texts': custom_preprocessed_validation_texts
    }

    # Save the dictionary to the JSON file
    with open('preprocessed_data.json', 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4)

    print("\nSuccessfully saved NEW preprocessed data to 'preprocessed_data.json'")