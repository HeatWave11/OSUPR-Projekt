import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# for hiding those tensorflow warnings/info
# reminder that if I don't manage to use my Vega 56 GPU, I should try build tensorflow with my CPU optimizations enabled
import keras
import pandas as pd
import numpy as np
from keras.src.layers import TextVectorization, Dense
from keras.src.models import Sequential
import nltk
nltk.download("punkt")
import data

# Klemen Požlep
# Code/Comments will be made in English so I'll have an easier time with documentation, textbooks and help from LLMs
# BOW for logistic regression, SVM, Naive Bayes?
# Sequential approach for Deep learning models (?)

bow_model = keras.models.load_model("SavedModels/bow_seq_model.keras")







