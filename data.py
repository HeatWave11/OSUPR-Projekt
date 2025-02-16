# made this so I don't run into circular import dependencies ...
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Data import
column_names = ['Text ID', 'Entity', 'Sentiment', 'Tweet']

# Define the relative path to the 'Data' folder inside your project
data_folder = os.path.join(os.getcwd(), "Data")

# Load the datasets using the new path
training_dataset = pd.read_csv(os.path.join(data_folder, "twitter_training.csv"),
                               header=None, names=column_names)

validation_dataset = pd.read_csv(os.path.join(data_folder, "twitter_validation.csv"),
                                 header=None, names=column_names)

print("Datasets loaded successfully!")

# Datatype: Pandas DataFrames
# Structure: Tabular data, like a spreadsheet.

print("Column names:", training_dataset.columns)
print("Column names:", validation_dataset.columns)

# Extracting the "Message/Tweet' column
training_texts = training_dataset['Tweet'].astype(str).values
validation_texts = validation_dataset['Tweet'].astype(str).values
# Data Type: NumPy array of strings
# Shape: 1D array
# Content: Each element is a Python string that represents a single tweet
# .astype(str) will make sure everything is a string!
# .values will convert from Pandas to NumPy array

# Mapping sentiment labels to numeric values
sentiment_mapping = {'Positive': 0, 'Negative': 1, 'Neutral': 2, 'Irrelevant': 3}
training_labels = training_dataset['Sentiment'].map(sentiment_mapping).values
validation_labels = validation_dataset['Sentiment'].map(sentiment_mapping).values
# Data Type: NumPy array of integers
# Shape: 1D array
# Content: Each element is a Python string that represents a single tweet
# .astype(str) will make sure everything is a string!
# .values will convert from Pandas to NumPy array

# Verifying the data (training)
print("Sample texts:", training_texts[:5])
print("Sample labels:", training_labels[:5])

# Verifying the data (validation)
print("Sample texts:", validation_texts[:5])
print("Sample labels:", validation_labels[:5])

# For finding out what our output_sequence_length should realistically be
# We'll tune this hyperparameter more a bit later (maybe)
# Calculating tweet lengths
tweet_lengths = [len(text.split()) for text in training_texts]

# Distribution Plot
plt.hist(tweet_lengths, bins=50)
plt.title('Distribution of Tweet Lengths')
plt.xlabel('Tweet Length (words)')
plt.ylabel('Frequency')
plt.show()

# Finding the 95th percentile
max_length_95 = int(np.percentile(tweet_lengths, 95))
print(f"95th Percentile tweet length is: {max_length_95}")