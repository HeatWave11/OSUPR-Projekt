import pandas as pd
import tensorflow as tf
import keras
from keras.src.layers import TextVectorization


# test

# Uvoz naših twitter podatkov
training_dataset = pd.read_csv('E:/School 2425/OSUPR/Projekt/archive/twitter_training.csv')
validation_dataset = pd.read_csv('E:/School 2425/OSUPR/Projekt/archive/twitter_validation.csv')
print(training_dataset.head())
print(validation_dataset.info())