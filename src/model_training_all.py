#Only skip svm if too slow to run
import argparse
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import SGDClassifier  # Faster alternative to SVM
from sklearn.model_selection import cross_val_score
import joblib
import mlflow
import mlflow.sklearn
import time
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
X_train = joblib.load('data/processed/train.pkl')
y_train = joblib.load('data/processed/train_labels.pkl')
X_val = joblib.load('data/processed/val.pkl')
y_val = joblib.load('data/processed/val_labels.pkl')

# Sample for optimization
sample_size = int(len(X_train) * 0.05)  # 5% for very fast optimization
print(f"Using {sample_size} samples for hyperparameter optimization...")
X_train_sample = X_train.iloc[:sample_size]
y_train_sample = y_train.iloc[:sample_size]

def rf_objective(trial):
    """Random Forest optimization."""
    n_est = trial.suggest_int('n_estimators', 50, 150)
    depth = trial.suggest_int('max_depth', 5, 15)

    model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=depth,
        random_state=42,
        n_jobs=-1
    )

    score = cross_val_score(model, X_train_sample, y_train_sample, cv=2, scoring='f1').mean()
    return score

def sgd_objective(trial):
    """SGD Classifier (fast alternative to SVM)."""
    alpha = trial.suggest_float('alpha', 0.0001, 0.01, log=True)
    max_iter = trial.suggest_int('max_iter', 100, 1000)

    model = SGDClassifier(
        loss='hinge',  # SVM-like
        alpha=alpha,
        max_iter=max_iter,
        random_state=42,
        n_jobs=-1
    )

    score = cross_val_score(model, X_train_sample, y_train_sample, cv=2, scoring='f1').mean()
    return score

def isolation_forest_objective(trial):
    """Isolation Forest optimization."""
    n_est = trial.suggest_int('n_estimators', 50, 150)
    contamination = trial.suggest_float('contamination', 0.05, 0.15)

    model = IsolationForest(
        n_estimators=n_est,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_sample)
    preds = model.predict(X_train_sample)
    preds_binary = (preds == -1).astype(int)

    score = f1_score(y_train_sample, preds_binary)
    return score

def train_and_evaluate_models():
    """Train all models and evaluate."""
    models = {}
    results = {}

    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)

    # 1. Random Forest (Full dataset)
    print("\n1. Random Forest")
    print("-" * 40)
    start = time.time()

    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(rf_objective, n_trials=3, show_progress_bar=False)

    rf_model = RandomForestClassifier(
        **study_rf.best_params,
        random_state=42,
        n_jobs=-1
    )

    print(f"   Optimization done. Training on full dataset...")
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_val)
    rf_f1 = f1_score(y_val, y_pred_rf)

    models['random_forest'] = rf_model
    results['random_forest'] = {
        'f1_score': rf_f1,
        'time': time.time() - start,
        'params': study_rf.best_params
    }

    print(f"   ✓ F1-Score: {rf_f1:.4f}")
    print(f"   ✓ Time: {results['random_forest']['time']:.1f}s")

    # 2. SGD Classifier (Fast SVM alternative)
    print("\n2. SGD Classifier (SVM-like)")
    print("-" * 40)
    start = time.time()

    study_sgd = optuna.create_study(direction='maximize')
    study_sgd.optimize(sgd_objective, n_trials=3, show_progress_bar=False)

    sgd_model = SGDClassifier(
        loss='hinge',
        **study_sgd.best_params,
        random_state=42,
        n_jobs=-1
    )

    print(f"   Optimization done. Training on subset (100k samples)...")
    train_size = min(100000, len(X_train))
    sgd_model.fit(X_train.iloc[:train_size], y_train.iloc[:train_size])

    y_pred_sgd = sgd_model.predict(X_val)
    sgd_f1 = f1_score(y_val, y_pred_sgd)

    models['sgd_svm'] = sgd_model
    results['sgd_svm'] = {
        'f1_score': sgd_f1,
        'time': time.time() - start,
        'params': study_sgd.best_params
    }

    print(f"   ✓ F1-Score: {sgd_f1:.4f}")
    print(f"   ✓ Time: {results['sgd_svm']['time']:.1f}s")

    # 3. Isolation Forest (Unsupervised)
    print("\n3. Isolation Forest (Unsupervised)")
    print("-" * 40)
    start = time.time()

    study_if = optuna.create_study(direction='maximize')
    study_if.optimize(isolation_forest_objective, n_trials=3, show_progress_bar=False)

    if_model = IsolationForest(
        **study_if.best_params,
        random_state=42,
        n_jobs=-1
    )

    print(f"   Optimization done. Training on full dataset...")
    if_model.fit(X_train)

    y_pred_if = if_model.predict(X_val)
    y_pred_if_binary = (y_pred_if == -1).astype(int)
    if_f1 = f1_score(y_val, y_pred_if_binary)

    models['isolation_forest'] = if_model
    results['isolation_forest'] = {
        'f1_score': if_f1,
        'time': time.time() - start,
        'params': study_if.best_params
    }

    print(f"   ✓ F1-Score: {if_f1:.4f}")
    print(f"   ✓ Time: {results['isolation_forest']['time']:.1f}s")

    return models, results

def save_models_and_results(models, results):
    """Save all models and print summary."""
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)

    for name, model in models.items():
        filename = f"models/{name}.pkl"
        joblib.dump(model, filename)
        print(f"✓ Saved {name} to {filename}")

    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    print(f"\n{'Model':<20} {'F1-Score':<12} {'Time (s)':<10}")
    print("-" * 42)

    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['f1_score']:.4f}      {metrics['time']:.1f}")

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\n🏆 Best Model: {best_model[0]} with F1-Score: {best_model[1]['f1_score']:.4f}")

    # Save results to file
    import json
    with open('models/results_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\n✓ Results saved to models/results_summary.json")

def main(args):
    mlflow.set_experiment("tcc_all_models")

    with mlflow.start_run():
        # Train and evaluate all models
        models, results = train_and_evaluate_models()

        # Log to MLflow
        for model_name, metrics in results.items():
            mlflow.log_metric(f"{model_name}_f1", metrics['f1_score'])
            mlflow.log_metric(f"{model_name}_time", metrics['time'])
            for param, value in metrics['params'].items():
                mlflow.log_param(f"{model_name}_{param}", value)

        # Save models and results
        save_models_and_results(models, results)

        print("\n✅ All models trained successfully!")
        print("📊 Check 'models/' directory for saved models")
        print("📈 Run 'python src/evaluation.py' for detailed evaluation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all models efficiently")
    args = parser.parse_args()
    main(args)