import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Step 1: Load your NSL-KDD dataset from .txt files
print("Loading dataset...")
try:
    # Use the .txt file in your readme directory
    train_path = '/home/viswa/dissertation/nsl_kdd/readme/KDDTrain+.txt'
    
    # NSL-KDD dataset typically has these columns
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class'
    ]
    
    # Load data with column names, using comma as separator
    data = pd.read_csv(train_path, header=None, names=column_names)
    print(f"Successfully loaded data from {train_path}")
    print(f"Dataset shape: {data.shape}")

except Exception as e:
    print(f"Error loading data: {e}")
    user_path = input("Please enter the full path to your NSL-KDD dataset: ")
    data = pd.read_csv(user_path, header=None, names=column_names)
    print(f"Successfully loaded data from {user_path}")

# Step 2: Identify and handle categorical data
categorical_cols = ['protocol_type', 'service', 'flag']

# Check for any other non-numeric columns
non_numeric_cols = data.select_dtypes(include=['object']).columns.tolist()
print(f"Non-numeric columns detected: {non_numeric_cols}")

# One-hot encode all categorical features
print("One-hot encoding categorical features...")
data = pd.get_dummies(data, columns=categorical_cols)
print(f"Dataset shape after encoding: {data.shape}")

# Check if all columns are now numeric
if data.select_dtypes(include=['object']).shape[1] > 0:
    print("Warning: Still have non-numeric columns:")
    print(data.select_dtypes(include=['object']).columns.tolist())
    
    # Handle any remaining object columns - try to convert to category
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = pd.Categorical(data[col]).codes

# Prepare your data
# Using all features except the target class
if 'class' in data.columns:
    # If class is still a column after encoding
    X = data.drop(['class'], axis=1)
    y = data['class']
    
    # If class is categorical, convert to numeric
    if y.dtype == 'object' or y.dtype.name == 'category':
        print("Converting target class to numeric...")
        y = pd.Categorical(y).codes
else:
    # If class got encoded as part of dummies
    class_cols = [col for col in data.columns if col.startswith('class_')]
    if class_cols:
        X = data.drop(class_cols, axis=1)
        y = data[class_cols].idxmax(axis=1).str.replace('class_', '')
    else:
        raise ValueError("Cannot find target class column")

print(f"Final X shape: {X.shape}")
print(f"Target y shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Create and train individual models
print("Training individual models...")

# Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Results:")
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_rf)))
print("F1 Score: {:.4f}".format(f1_score(y_test, y_pred_rf, average='weighted')))

# MLP (Neural Network)
print("Training MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print("\nMLP Results:")
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_mlp)))
print("F1 Score: {:.4f}".format(f1_score(y_test, y_pred_mlp, average='weighted')))

# Store models in a dictionary
classifiers = {
    'Random Forest': rf,
    'MLP': mlp
}

# Step 4: Create and evaluate the ensemble
print("\nTraining ensemble model...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', classifiers['Random Forest']),
        ('mlp', classifiers['MLP'])
    ],
    voting='soft'  # Use probability estimates for voting
)

# Train and evaluate
ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)

print("\nEnsemble Model Results:")
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred_ensemble)))
print("F1 Score: {:.4f}".format(f1_score(y_test, y_pred_ensemble, average='weighted')))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_ensemble))

# Step 5: ROC Curve Analysis
print("\nGenerating ROC curves...")

# Check if we're dealing with binary or multiclass classification
unique_classes = np.unique(y)
num_classes = len(unique_classes)
print(f"Number of unique classes: {num_classes}")

plt.figure(figsize=(10, 8))

classifiers_to_evaluate = {
    'Random Forest': classifiers['Random Forest'],
    'MLP': classifiers['MLP'],
    'Ensemble': ensemble
}

# For multiclass ROC curve, we use One-vs-Rest approach
if num_classes > 2:
    # Create binary labels for each class
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    
    for name, clf in classifiers_to_evaluate.items():
        # Get probability predictions
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)
            
            # Compute micro-average ROC curve and ROC area
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            plt.plot(fpr["micro"], tpr["micro"], lw=2, 
                     label=f'{name} (micro-avg AUC = {roc_auc["micro"]:.3f})')
else:
    # Binary classification case
    for name, clf in classifiers_to_evaluate.items():
        if hasattr(clf, "predict_proba"):
            probas = clf.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, probas[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

# Plot diagonal line for random classifier
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for NSL-KDD Classification')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('nsl_kdd_roc_curves.png')
print("ROC curve saved as 'nsl_kdd_roc_curves.png'")

plt.show()
