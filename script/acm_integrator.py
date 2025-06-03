import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data with selected features
print("Loading preprocessed data with selected features...")
selected_features_mask = np.load('selected_features_mask.npy')

X_train = np.load('X_train_preprocessed.npy')[:, selected_features_mask]
X_test = np.load('X_test_preprocessed.npy')[:, selected_features_mask]
y_train = np.load('y_train_binary.npy')  # Using binary classification
y_test = np.load('y_test_binary.npy')

# Load feature names
selected_features = []
with open('selected_features.txt', 'r') as f:
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split(',')
        if len(parts) > 1:
            selected_features.append(parts[1])

print(f"Dataset shape after feature selection: {X_train.shape}")
print(f"Selected features: {selected_features}")

# Define classifiers to evaluate
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

# Train and evaluate each classifier
results = {}
training_times = {}
prediction_times = {}

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    
    # Training time
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    training_times[name] = train_time
    
    # Prediction time
    start_time = time.time()
    y_pred = clf.predict(X_test)
    pred_time = time.time() - start_time
    prediction_times[name] = pred_time
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'train_time': train_time,
        'pred_time': pred_time
    }
    
    print(f"{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Prediction Time: {pred_time:.2f} seconds")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'cm_{name.replace(" ", "_")}.png')
    
    # Save detailed classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)

# Compare classifiers
# Convert results to DataFrame for easier comparison
results_df = pd.DataFrame({
    clf_name: {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1 Score': metrics['f1'],
        'Training Time (s)': metrics['train_time'],
        'Prediction Time (s)': metrics['pred_time']
    }
    for clf_name, metrics in results.items()
})

print("\nClassifier Comparison:")
print(results_df)

# Save comparison results
results_df.to_csv('classifier_comparison.csv')

# Plot comparison charts
plt.figure(figsize=(12, 8))

# Plot accuracy, precision, recall, F1
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    plt.bar(results_df.columns, results_df.loc[metric])
    plt.title(metric)
    plt.ylim(0, 1)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('performance_comparison.png')

# Plot time comparison
plt.figure(figsize=(10, 6))
time_data = {
    'Training Time': [results[clf]['train_time'] for clf in classifiers],
    'Prediction Time': [results[clf]['pred_time'] for clf in classifiers]
}
df = pd.DataFrame(time_data, index=classifiers.keys())
df.plot(kind='bar')
plt.title('Classifier Time Comparison')
plt.ylabel('Time (seconds)')
plt.tight_layout()
plt.savefig('time_comparison.png')

print("Classifier evaluation complete!")
