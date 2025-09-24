import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time

# Load test data
print("Loading test data...")
X_test = joblib.load('data/processed/test.pkl')
y_test = joblib.load('data/processed/test_labels.pkl')

# Load available models
print("Loading models...")
models = {}

if os.path.exists('models/rf_model.pkl'):
    models['Random Forest'] = joblib.load('models/rf_model.pkl')
    print("  [OK] Random Forest loaded")

if os.path.exists('models/svm_model.pkl'):
    models['SVM'] = joblib.load('models/svm_model.pkl')
    print("  [OK] SVM loaded")

if os.path.exists('models/isolation_model.pkl'):
    models['Isolation Forest'] = joblib.load('models/isolation_model.pkl')
    print("  [OK] Isolation Forest loaded")

if os.path.exists('models/random_forest_old.pkl') and 'Random Forest' not in models:
    models['Random Forest'] = joblib.load('models/random_forest_old.pkl')
    print("  [OK] Random Forest (old) loaded")

def evaluate_model_fast(model, X, y, name):
    """Fast evaluation without bootstrap."""
    print(f"\nEvaluating {name}...")
    start = time.time()

    # Predictions
    if name == 'Isolation Forest':
        y_pred = model.predict(X)
        y_pred = (y_pred == -1).astype(int)  # Convert to binary
    else:
        y_pred = model.predict(X)

    # Calculate metrics
    metrics = {
        'Model': name,
        'F1-Score': f1_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'Time (s)': time.time() - start
    }

    # AUC if model has predict_proba
    if hasattr(model, 'predict_proba') and name != 'Isolation Forest':
        try:
            y_proba = model.predict_proba(X)[:, 1]
            metrics['AUC'] = roc_auc_score(y, y_proba)
        except:
            metrics['AUC'] = 'N/A'
    else:
        metrics['AUC'] = 'N/A'

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    metrics['TN'], metrics['FP'], metrics['FN'], metrics['TP'] = cm.ravel()
    metrics['FPR'] = metrics['FP'] / (metrics['FP'] + metrics['TN']) if (metrics['FP'] + metrics['TN']) > 0 else 0

    return metrics

def plot_results(results_df):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Metrics comparison
    metrics_to_plot = ['F1-Score', 'Precision', 'Recall']
    metrics_data = results_df[metrics_to_plot].values.T
    x = np.arange(len(metrics_to_plot))
    width = 0.2

    for i, model in enumerate(results_df['Model']):
        offset = (i - len(results_df) / 2) * width + width / 2
        axes[0, 0].bar(x + offset, metrics_data[:, i], width, label=model)

    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics_to_plot)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1.05])

    # 2. F1-Score bar plot
    axes[0, 1].barh(results_df['Model'], results_df['F1-Score'])
    axes[0, 1].set_xlabel('F1-Score')
    axes[0, 1].set_title('F1-Score by Model')
    axes[0, 1].set_xlim([0, 1])

    # Add value labels
    for i, v in enumerate(results_df['F1-Score']):
        axes[0, 1].text(v + 0.01, i, f'{v:.4f}', va='center')

    # 3. FPR comparison
    axes[1, 0].bar(results_df['Model'], results_df['FPR'])
    axes[1, 0].set_ylabel('False Positive Rate')
    axes[1, 0].set_title('False Positive Rate by Model')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. Confusion matrix for best model
    best_model_idx = results_df['F1-Score'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    cm_data = [[results_df.loc[best_model_idx, 'TN'], results_df.loc[best_model_idx, 'FP']],
               [results_df.loc[best_model_idx, 'FN'], results_df.loc[best_model_idx, 'TP']]]

    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title(f'Confusion Matrix - {best_model_name}')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    axes[1, 1].set_yticklabels(['BENIGN', 'ATTACK'])
    axes[1, 1].set_xticklabels(['BENIGN', 'ATTACK'])

    plt.suptitle('Model Evaluation Results', fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Plots saved to figures/model_comparison.png")

def main():
    if not models:
        print("[ERROR] No models found in models/ directory!")
        print("   Please run model_training_all.py first.")
        return

    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print('='*60)

    results = []

    # Evaluate each model
    for name, model in models.items():
        metrics = evaluate_model_fast(model, X_test, y_test, name)
        results.append(metrics)

        # Print results
        print(f"  F1-Score: {metrics['F1-Score']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall: {metrics['Recall']:.4f}")
        if metrics['AUC'] != 'N/A':
            print(f"  AUC: {metrics['AUC']:.4f}")
        print(f"  FPR: {metrics['FPR']:.4f}")

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print('='*60)
    print(results_df[['Model', 'F1-Score', 'Precision', 'Recall', 'FPR']].to_string(index=False))

    # Save results
    results_df.to_csv('figures/evaluation_results.csv', index=False)
    print("\n[OK] Results saved to figures/evaluation_results.csv")

    # Generate plots
    plot_results(results_df)

    # Find best model
    best_idx = results_df['F1-Score'].idxmax()
    print(f"\n[BEST MODEL] {results_df.loc[best_idx, 'Model']}")
    print(f"   F1-Score: {results_df.loc[best_idx, 'F1-Score']:.4f}")

    print("\n[SUCCESS] Evaluation complete!")

if __name__ == "__main__":
    main()