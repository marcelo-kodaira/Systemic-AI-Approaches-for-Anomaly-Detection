import argparse
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import joblib
import mlflow
import mlflow.sklearn
import time
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Load data
print("Loading data...")
X_train = joblib.load('data/processed/train.pkl')
y_train = joblib.load('data/processed/train_labels.pkl')
X_val = joblib.load('data/processed/val.pkl')
y_val = joblib.load('data/processed/val_labels.pkl')

# Sample for faster optimization
sample_size = int(len(X_train) * 0.1)
print(f"Using {sample_size} samples for optimization...")
X_train_sample = X_train.iloc[:sample_size]
y_train_sample = y_train.iloc[:sample_size]

def rf_objective(trial):
    """Random Forest optimization."""
    n_est = trial.suggest_int('n_estimators', 50, 200)
    depth = trial.suggest_int('max_depth', 5, 20)
    min_samples = trial.suggest_int('min_samples_split', 2, 10)

    model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=depth,
        min_samples_split=min_samples,
        random_state=42,
        n_jobs=-1
    )

    score = cross_val_score(model, X_train_sample, y_train_sample, cv=3, scoring='f1').mean()
    return score

def svm_objective(trial):
    """SVM optimization."""
    C = trial.suggest_float('C', 0.1, 10.0, log=True)
    gamma = trial.suggest_float('gamma', 0.001, 1.0, log=True)

    model = SVC(
        C=C,
        gamma=gamma,
        kernel='rbf',
        random_state=42
    )

    # Use smaller sample for SVM (computationally expensive)
    svm_sample = min(5000, len(X_train_sample))
    X_svm = X_train_sample.iloc[:svm_sample]
    y_svm = y_train_sample.iloc[:svm_sample]

    score = cross_val_score(model, X_svm, y_svm, cv=2, scoring='f1').mean()
    return score

def isolation_forest_objective(trial):
    """Isolation Forest optimization (unsupervised)."""
    n_est = trial.suggest_int('n_estimators', 50, 200)
    contamination = trial.suggest_float('contamination', 0.01, 0.2)
    max_samples = trial.suggest_float('max_samples', 0.5, 1.0)

    model = IsolationForest(
        n_estimators=n_est,
        contamination=contamination,
        max_samples=max_samples,
        random_state=42,
        n_jobs=-1
    )

    # Fit and predict
    model.fit(X_train_sample)
    preds = model.predict(X_train_sample)
    # Convert: -1 (anomaly) to 1 (attack), 1 (normal) to 0 (benign)
    preds_binary = (preds == -1).astype(int)

    score = f1_score(y_train_sample, preds_binary)
    return score

def train_best_models(studies):
    """Train final models with best parameters on full dataset."""
    models = {}

    # Random Forest
    print("\n1. Training Random Forest with best parameters...")
    rf_params = studies['rf'].best_params
    rf_model = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    models['rf'] = rf_model
    print(f"   Best params: {rf_params}")

    # SVM (train on subset due to computational cost)
    print("\n2. Training SVM with best parameters...")
    svm_params = studies['svm'].best_params
    svm_model = SVC(**svm_params, kernel='rbf', random_state=42, probability=True, cache_size=1000)
    # Train on MUCH smaller subset for SVM (it's too slow)
    svm_size = min(10000, len(X_train))  # Reduced from 50000 to 10000
    print(f"   Training on {svm_size} samples (SVM is computationally expensive)...")
    start_time = time.time()
    svm_model.fit(X_train.iloc[:svm_size], y_train.iloc[:svm_size])
    print(f"   Training completed in {time.time() - start_time:.1f} seconds")
    models['svm'] = svm_model
    print(f"   Best params: {svm_params}")

    # Isolation Forest
    print("\n3. Training Isolation Forest with best parameters...")
    if_params = studies['isolation'].best_params
    if_model = IsolationForest(**if_params, random_state=42, n_jobs=-1)
    if_model.fit(X_train)
    models['isolation'] = if_model
    print(f"   Best params: {if_params}")

    return models

def evaluate_models(models):
    """Evaluate all models on validation set."""
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)

    for name, model in models.items():
        print(f"\n{name.upper()} Model:")

        if name == 'isolation':
            # Special handling for Isolation Forest
            preds = model.predict(X_val)
            preds_binary = (preds == -1).astype(int)
        else:
            preds_binary = model.predict(X_val)

        print(classification_report(y_val, preds_binary,
                                   target_names=['BENIGN', 'ATTACK'],
                                   digits=4))

        cm = confusion_matrix(y_val, preds_binary)
        print(f"Confusion Matrix:\n{cm}")

def main(args):
    mlflow.set_experiment("tcc_models_complete")

    with mlflow.start_run():
        studies = {}

        # 1. Random Forest optimization
        print("\n" + "="*60)
        print("OPTIMIZING RANDOM FOREST")
        print("="*60)
        study_rf = optuna.create_study(direction='maximize', study_name='rf')
        study_rf.optimize(rf_objective, n_trials=args.trials)
        studies['rf'] = study_rf
        print(f"Best F1: {study_rf.best_value:.4f}")

        # 2. SVM optimization
        print("\n" + "="*60)
        print("OPTIMIZING SVM")
        print("="*60)
        study_svm = optuna.create_study(direction='maximize', study_name='svm')
        study_svm.optimize(svm_objective, n_trials=args.trials)
        studies['svm'] = study_svm
        print(f"Best F1: {study_svm.best_value:.4f}")

        # 3. Isolation Forest optimization
        print("\n" + "="*60)
        print("OPTIMIZING ISOLATION FOREST")
        print("="*60)
        study_if = optuna.create_study(direction='maximize', study_name='isolation')
        study_if.optimize(isolation_forest_objective, n_trials=args.trials)
        studies['isolation'] = study_if
        print(f"Best F1: {study_if.best_value:.4f}")

        # Train best models
        models = train_best_models(studies)

        # Evaluate
        evaluate_models(models)

        # Save models
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        for name, model in models.items():
            filename = f"models/{name}_model.pkl"
            joblib.dump(model, filename)
            print(f"Saved {name} to {filename}")
            mlflow.sklearn.log_model(model, name)

        # Log parameters
        for name, study in studies.items():
            mlflow.log_params({f"{name}_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric(f"{name}_best_f1", study.best_value)

        print("\n✅ Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all models")
    parser.add_argument("--trials", type=int, default=5, help="Number of Optuna trials")
    args = parser.parse_args()
    main(args)