import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import time

print("Loading test data...")
X_test = joblib.load('data/processed/test.pkl')
y_test = joblib.load('data/processed/test_labels.pkl')

# Use only 10% of test data for quick evaluation
test_size = int(len(X_test) * 0.1)
X_test_sample = X_test.iloc[:test_size]
y_test_sample = y_test.iloc[:test_size]

print(f"Using {test_size} test samples (10% of full test set)")

# Load the smallest model
print("\nLoading Random Forest model...")
model = joblib.load('models/random_forest.pkl')

print("Making predictions...")
start = time.time()
y_pred = model.predict(X_test_sample)
pred_time = time.time() - start

# Calculate metrics
f1 = f1_score(y_test_sample, y_pred)
precision = precision_score(y_test_sample, y_pred)
recall = recall_score(y_test_sample, y_pred)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test_sample, y_pred).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print("\n" + "="*50)
print("RANDOM FOREST EVALUATION RESULTS (10% sample)")
print("="*50)
print(f"F1-Score:    {f1:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"FPR:         {fpr:.4f}")
print(f"Pred. Time:  {pred_time:.2f}s")
print("\nConfusion Matrix:")
print(f"  TN: {tn:6d}  FP: {fp:6d}")
print(f"  FN: {fn:6d}  TP: {tp:6d}")

print("\n[SUCCESS] Quick evaluation complete!")