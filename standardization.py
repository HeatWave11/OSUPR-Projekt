import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from data import training_texts, training_labels, validation_texts, validation_labels


# check certificates in the python version directory if you get ssl error
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def custom_standardization(text):
    # If 'texts' is a list, apply the standardization to each element
    if isinstance(text, list):
        return [custom_standardization(text) for text in text]
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
    # tokenizes the string into a list of words!

    # Remove stopwords and lemmatize
    # processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Get POS tags
    tagged_words = nltk.pos_tag(words)

    processed_words = []
    for word, tag in tagged_words:
        if word not in stop_words:
            # Convert NLTK POS tags to WordNet POS tags
            if tag.startswith('J'):
                pos = 'a'
            elif tag.startswith('V'):
                pos = 'v'
            elif tag.startswith('N'):
                pos = 'n'
            elif tag.startswith('R'):
                pos = 'r'
            else:
                pos = 'n'  # Default to noun

            processed_words.append(lemmatizer.lemmatize(word, pos))

    # Reconstruct the sentence
    return ' '.join(processed_words)

# testing
print(custom_standardization("Nyaaa www.google.com 5555 22!! running"))

if __name__ == "__main__":

    custom_preprocessed_training_texts = [custom_standardization(text) for text in training_texts]
    custom_preprocessed_validation_texts = [custom_standardization(text) for text in validation_texts]
    #   both are still NumPy arrays of strings BUT now they have been preprocessed
    # shape is the same
    # strings were modified by preprocessing

    print("First 10 texts:")
    for i, text in enumerate(custom_preprocessed_training_texts[:10], start=1):
        print(f"{i}. {text}")


    with open('preprocessed_data.pkl', 'wb') as f:
        pickle.dump({
            'custom_preprocessed_training_texts': custom_preprocessed_training_texts,
            'custom_preprocessed_validation_texts': custom_preprocessed_validation_texts
        }, f)

    # replace this as well