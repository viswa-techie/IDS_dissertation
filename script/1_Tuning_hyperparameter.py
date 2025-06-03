import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

# First, load the preprocessed data with selected features
print("Loading preprocessed data with selected features...")
try:
    # Load feature mask
    selected_features_mask = np.load('selected_features_mask.npy')
    
    # Load preprocessed data
    X_train = np.load('X_train_preprocessed.npy')[:, selected_features_mask]
    X_test = np.load('X_test_preprocessed.npy')[:, selected_features_mask]
    y_train = np.load('y_train_binary.npy')
    y_test = np.load('y_test_binary.npy')
    
    print(f"Data loaded successfully!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Class distribution in training set: {np.unique(y_train, return_counts=True)}")
    
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please make sure all required files exist in the current directory.")
    exit(1)

# Now perform hyperparameter tuning for MLP
print("\nPerforming hyperparameter tuning for MLP...")

# Define parameter grid
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (150, 75)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [200, 300]
}

# Use RandomizedSearchCV for efficiency
mlp_clf = MLPClassifier(random_state=42)
random_search = RandomizedSearchCV(
    mlp_clf, param_grid, n_iter=15, cv=3, 
    scoring='f1_weighted', random_state=42, n_jobs=-1,
    verbose=2  # Add verbosity to see progress
)

# Fit the model
print("Training models with various hyperparameters (this may take some time)...")
random_search.fit(X_train, y_train)

# Print best parameters and score
print("\nBest parameters:", random_search.best_params_)
print("Best cross-validation score: {:.4f}".format(random_search.best_score_))

# Evaluate best model on test set
best_mlp = random_search.best_estimator_
print("\nEvaluating best model on test set...")
y_pred = best_mlp.predict(X_test)
print("Best MLP Test Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
print("Best MLP Test F1 Score: {:.4f}".format(f1_score(y_test, y_pred, average='weighted')))

# Print detailed classification report
print("\nClassification Report for Best Model:")
print(classification_report(y_test, y_pred))

# Save the best model
import joblib
joblib.dump(best_mlp, 'best_mlp_model.joblib')
print("Best model saved as 'best_mlp_model.joblib'")
