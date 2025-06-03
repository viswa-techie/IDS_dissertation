import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os

# File paths
TRAIN_PATH = 'KDDTrain+.txt'  # Adjust paths to your files
TEST_PATH = 'KDDTest+.txt'

# Column names for the NSL-KDD dataset
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
             "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
             "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
             "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
             "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", 
             "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
             "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]

# Load data
print("Loading dataset...")
train_df = pd.read_csv(TRAIN_PATH, header=None, names=col_names)
test_df = pd.read_csv(TEST_PATH, header=None, names=col_names)

# Remove 'difficulty' column which is not needed
train_df = train_df.drop('difficulty', axis=1)
test_df = test_df.drop('difficulty', axis=1)

# Check for class distribution before preprocessing
print("Train class distribution before preprocessing:")
print(train_df['label'].value_counts())
print("Test class distribution before preprocessing:")
print(test_df['label'].value_counts())

# Convert attack labels to binary classification (normal=0, attack=1)
# But let's preserve a multi-class version for potential multi-class classification
train_df['attack_class'] = train_df['label'].copy()
test_df['attack_class'] = test_df['label'].copy()

# Map attack types to binary classification (normal vs attack)
def process_labels(df):
    # Preserve original labels
    df['original_label'] = df['label']
    
    # Create binary labels (normal=0, attack=1)
    df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Create multi-class labels (normal=0, DoS=1, Probe=2, R2L=3, U2R=4)
    dos_attacks = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable', 'worm']
    probe_attacks = ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']
    r2l_attacks = ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy', 'xlock', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named']
    u2r_attacks = ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']
    
    def get_attack_class(label):
        if label == 'normal':
            return 0
        elif label in dos_attacks:
            return 1
        elif label in probe_attacks:
            return 2
        elif label in r2l_attacks:
            return 3
        elif label in u2r_attacks:
            return 4
        else:
            return 1  # Unknown attack types are classified as attacks
    
    df['multi_label'] = df['label'].apply(get_attack_class)
    return df

train_df = process_labels(train_df)
test_df = process_labels(test_df)

# Print class distributions after transformation
print("Binary class distribution in training data:")
print(train_df['binary_label'].value_counts())
print("\nMulti-class distribution in training data:")
print(train_df['multi_label'].value_counts())

# Identify categorical and numerical columns
categorical_cols = ['protocol_type', 'service', 'flag']
numeric_cols = [col for col in train_df.columns if col not in categorical_cols + 
               ['label', 'binary_label', 'multi_label', 'attack_class', 'original_label']]

# Check for and remove columns with zero variance
for col in numeric_cols:
    if train_df[col].nunique() <= 1:
        print(f"Removing zero variance column: {col}")
        numeric_cols.remove(col)

# Print feature types
print(f"\nCategorical features: {len(categorical_cols)}")
print(f"Numerical features: {len(numeric_cols)}")

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# Separate features and target
X_train = train_df.drop(['label', 'binary_label', 'multi_label', 'attack_class', 'original_label'], axis=1)
X_test = test_df.drop(['label', 'binary_label', 'multi_label', 'attack_class', 'original_label'], axis=1)

# Choose which label type to use (binary or multi-class)
# For now, let's use binary classification
y_train_binary = train_df['binary_label'] 
y_test_binary = test_df['binary_label']
y_train_multi = train_df['multi_label']
y_test_multi = test_df['multi_label']

# Fit and transform the training data
print("Preprocessing data...")
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
ohe_feature_names = []
for name, transform, features in preprocessor.transformers_:
    if name == 'cat':
        for i, feature in enumerate(features):
            categories = preprocessor.named_transformers_['cat'].categories_[i]
            for category in categories:
                ohe_feature_names.append(f"{feature}_{category}")
    else:
        ohe_feature_names.extend(features)

print(f"Number of features after preprocessing: {X_train_preprocessed.shape[1]}")

# Save preprocessed data
print("Saving preprocessed data...")
np.save('X_train_preprocessed.npy', X_train_preprocessed)
np.save('X_test_preprocessed.npy', X_test_preprocessed)
np.save('y_train_binary.npy', y_train_binary)
np.save('y_test_binary.npy', y_test_binary) 
np.save('y_train_multi.npy', y_train_multi)
np.save('y_test_multi.npy', y_test_multi)

# Save feature names
with open('feature_names.txt', 'w') as f:
    for feature in ohe_feature_names:
        f.write(f"{feature}\n")

print("Preprocessing complete!")
