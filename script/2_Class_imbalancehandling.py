from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load the best MLP model saved from the previous script
best_mlp = joblib.load('best_mlp_model.joblib')

print("Implementing class balancing techniques...")

# Load your preprocessed data files
X_train = np.load('X_train_preprocessed.npy')
X_test = np.load('X_test_preprocessed.npy')
# Choose one of these depending on your classification task:
y_train = np.load('y_train_binary.npy')  # For binary classification
y_test = np.load('y_test_binary.npy')    # For binary classification
# OR
# y_train = np.load('y_train_multi.npy')  # For multi-class classification
# y_test = np.load('y_test_multi.npy')    # For multi-class classification

# Create a pipeline with SMOTE oversampling and your best classifier
smote_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', best_mlp)
])

# Train and evaluate the balanced model
smote_pipeline.fit(X_train, y_train)
y_pred_balanced = smote_pipeline.predict(X_test)

print("Balanced Model Results:")
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_balanced)))
print("F1 Score: {:.4f}".format(f1_score(y_test, y_pred_balanced, average='weighted')))
print(classification_report(y_test, y_pred_balanced))
