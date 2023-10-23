import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Prepare your dataset
# Replace 'path_to_your_data.csv' with the actual path to your CSV file containing review texts and their corresponding labels.
data = pd.read_csv('/Users/zhengbaoqin/Desktop/fz/ColabteslaAnalysis.csv')

# Step 2: Data Split
X = data['Review']  # Assuming 'review_text' is the column name for review texts.
y = data['Polarity']    # Assuming 'sentiment' is the column name for sentiment labels.

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 3: Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000)  # You can experiment with different settings for max_features.
X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)
X_test_vectorized = vectorizer.transform(X_test)

# Step 4: Model Selection and Training
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_vectorized, y_train)

# Step 5: Evaluation
y_val_pred = svm_model.predict(X_val_vectorized)
y_test_pred = svm_model.predict(X_test_vectorized)

# Evaluation Metrics
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, average='weighted')
val_recall = recall_score(y_val, y_val_pred, average='weighted')
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print("Validation Set Metrics:")
print(f"Accuracy: {val_accuracy:.2f}")
print(f"Precision: {val_precision:.2f}")
print(f"Recall: {val_recall:.2f}")
print(f"F1-Score: {val_f1:.2f}")

print("\nTest Set Metrics:")
print(f"Accuracy: {test_accuracy:.2f}")
print(f"Precision: {test_precision:.2f}")
print(f"Recall: {test_recall:.2f}")
print(f"F1-Score: {test_f1:.2f}")