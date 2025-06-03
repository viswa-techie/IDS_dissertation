import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Load your preprocessed data
X_train = np.load('X_train_preprocessed.npy')
X_test = np.load('X_test_preprocessed.npy')
y_train = np.load('y_train_binary.npy')  # or y_train_multi.npy depending on your task
y_test = np.load('y_test_binary.npy')    # or y_test_multi.npy

# Load the feature names if available
try:
    with open('feature_names.txt', 'r') as f:
        feature_names = f.read().splitlines()
except:
    # Create generic feature names if file doesn't exist
    feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

# Define classifiers for feature importance analysis
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'MLP': joblib.load('best_mlp_model.joblib')
}

# Feature importance analysis
print("Performing feature importance analysis...")

if 'Random Forest' in classifiers:
    # Train Random Forest for feature importance
    rf = classifiers['Random Forest']
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature ranking
    print("Feature ranking:")
    for f in range(min(20, X_train.shape[1])):  # Print top 20 features
        print(f"{f+1}. Feature {feature_names[indices[f]]} ({importances[indices[f]]})")
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(min(20, X_train.shape[1])), importances[indices[:20]], align="center")
    plt.xticks(range(min(20, X_train.shape[1])), [feature_names[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Select most important features
    selector = SelectFromModel(rf, prefit=True, threshold='mean')
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Save the selected feature mask for future use
    selected_features_mask = selector.get_support()
    np.save('selected_features_mask.npy', selected_features_mask)
    
    # Save selected feature names
    # Fix for the index out of bounds issue
selected_feature_names = []
for i in range(len(selected_features_mask)):
    if i < len(feature_names) and selected_features_mask[i]:
        selected_feature_names.append(feature_names[i])
    
    print(f"Selected {X_train_selected.shape[1]} important features out of {X_train.shape[1]}")
    print(f"Selected features saved to 'selected_features.txt'")
