import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV # Import GridSearchCV

print("--- SVM Model Training with Hyperparameter Tuning ---")

# 1. LOAD PREPROCESSED DATA
# This part is very fast, so a progress bar isn't needed here.
with open('preprocessed_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    training_texts = data['custom_preprocessed_training_texts']
    validation_texts = data['custom_preprocessed_validation_texts']

# Load corresponding labels (make sure these are correct)
from data import training_labels, validation_labels
# training_labels = ...
# validation_labels = ...

print(f"Loaded {len(training_texts)} training texts and {len(validation_texts)} validation texts.")


# 2. FEATURE EXTRACTION (VECTORIZATION)
# Adding some recommended parameters for better performance
print("\nCreating and fitting the TF-IDF Vectorizer...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=3, # Ignore words that are too rare
    max_df=0.9 # Ignore words that are too common
)

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(training_texts)
# Only transform the validation data
X_val_tfidf = tfidf_vectorizer.transform(validation_texts)

print("Vectorization complete.")
print("Shape of training data:", X_train_tfidf.shape)


# 3. TRAIN THE SVM MODEL USING GRIDSEARCHCV FOR BEST RESULTS
# Define the base model
svm = SVC()

# Define the grid of parameters to search through
# This is a focused grid. You can expand it for a more exhaustive search.
param_grid = {
    'C': [1, 10],            # Regularization parameter
    'kernel': ['linear', 'rbf'], # Kernel type
}

# --- ADDING PROGRESS INDICATOR HERE ---
# Create the GridSearchCV object.
# verbose=2 will print detailed progress for each combination.
# cv=3 means 3-fold cross-validation will be used for tuning.
# n_jobs=-1 uses all available CPU cores to speed up the search.
grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=3,
    verbose=2, # <-- THIS IS YOUR PROGRESS BAR!
    n_jobs=-1
)

print("\nRunning GridSearchCV to find the best SVM hyperparameters...")
print("This may take several minutes depending on your data size and CPU.")

# Fit the grid search to the data. You will see progress updates in the console.
grid_search.fit(X_train_tfidf, training_labels)

# Get the best model found by the search
print("\nGrid search complete.")
print(f"Best parameters found: {grid_search.best_params_}")
best_svm_model = grid_search.best_estimator_


# 4. EVALUATE THE BEST MODEL
print("\nEvaluating best model's performance on the unseen validation set...")
predictions = best_svm_model.predict(X_val_tfidf)

class_labels = ["Positive", "Negative", "Neutral", "Irrelevant"]
# Adjust target_names based on the actual number of unique labels in your validation set
unique_labels_count = len(set(validation_labels))
report = classification_report(validation_labels, predictions, target_names=class_labels[:unique_labels_count])

print("--- Final Classification Report ---")
print(report)


# 5. SAVE THE BEST MODEL AND THE VECTORIZER
model_filename = "SavedModels/svm_model.joblib"
vectorizer_filename = "SavedVectorizers/svm_tfidf_vectorizer.joblib"

joblib.dump(best_svm_model, model_filename)
joblib.dump(tfidf_vectorizer, vectorizer_filename)

print(f"\n✅ Best SVM model saved successfully to '{model_filename}'")
print(f"✅ Vectorizer saved successfully to '{vectorizer_filename}'")