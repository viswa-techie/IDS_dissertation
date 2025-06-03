import numpy as np
import pandas as pd
import joblib

# Load the model and data
best_mlp = joblib.load('best_mlp_model.joblib')
X_test = np.load('X_test_preprocessed.npy')
y_test = np.load('y_test_binary.npy')  # Load binary labels

# Check for feature selection mask and apply if it exists
# Load the specific feature indices used during training
try:
    selected_indices = np.load('selected_feature_indices.npy')
    X_test = X_test[:, selected_indices]
    print(f"Using the exact {len(selected_indices)} features from training")
except FileNotFoundError:
    print("Could not find selected feature indices")

# Check for feature selection mask and apply if it exists
try:
    selected_features_mask = np.load('selected_features_mask.npy')
    print(f"Applying feature selection: reducing from {X_test.shape[1]} to {sum(selected_features_mask)} features")
    X_test = X_test[:, selected_features_mask]
except FileNotFoundError:
    print("No feature selection mask found. This might cause a feature count mismatch.")

# Check feature count
print(f"Model expects {best_mlp.n_features_in_} features, X_test has {X_test.shape[1]} features")

# If still mismatched, try loading alternative feature-selected data
if hasattr(best_mlp, 'n_features_in_') and best_mlp.n_features_in_ != X_test.shape[1]:
    print("Feature count mismatch detected. Looking for pre-selected data...")
    try:
        X_test = np.load('X_test_selected.npy')  # Try to load pre-selected data if available
        print(f"Loaded pre-selected test data with {X_test.shape[1]} features")
    except FileNotFoundError:
        # If all else fails, just use the first n_features from X_test
        print(f"WARNING: Subsetting features from {X_test.shape[1]} to {best_mlp.n_features_in_}")
        X_test = X_test[:, :best_mlp.n_features_in_]

# After feature selection but before prediction - show what we're actually using
print(f"Using these features (first 3 shown):")
for i in range(min(3, X_test.shape[1])):
    print(f"Feature {i}: First few values: {X_test[:3, i]}")

# Analyze misclassifications
y_pred = best_mlp.predict(X_test)
misclassified = X_test[y_pred != y_test]
true_labels = y_test[y_pred != y_test]
pred_labels = y_pred[y_pred != y_test]

print(f"Total misclassified instances: {len(misclassified)}")
print(f"Misclassification rate: {len(misclassified)/len(y_test):.4f}")

# Analyze which types of attacks are most commonly misclassified
if len(np.unique(y_test)) > 2:  # If you have multi-class labels
    # Load multi-class labels if available
    y_test_multi = np.load('y_test_multi.npy')
    
    # Map back to attack types
    attack_types = {0: 'normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}
    y_test_attack_types = np.array([attack_types[y] for y in y_test_multi])
    
    # Get attack types of misclassified samples
    misclassified_attack_types = y_test_attack_types[y_pred != y_test]
    
    # Count by attack type
    attack_counts = pd.Series(misclassified_attack_types).value_counts()
    print("\nMisclassified samples by attack type:")
    print(attack_counts)
    
    # Normalize by total count of each type
    total_by_type = pd.Series(y_test_attack_types).value_counts()
    
    error_rates = {}
    for attack_type in total_by_type.index:
        if attack_type in attack_counts:
            error_rates[attack_type] = attack_counts[attack_type] / total_by_type[attack_type]
        else:
            error_rates[attack_type] = 0
    
    print("\nError rates by attack type:")
    for attack_type, rate in error_rates.items():
        print(f"{attack_type}: {rate:.4f}")
