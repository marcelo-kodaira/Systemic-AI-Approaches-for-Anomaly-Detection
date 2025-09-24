import argparse
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
import mlflow
import mlflow.sklearn
import time

# Load data (from preprocessing)
print("Loading data...")
X_train = joblib.load('data/processed/train.pkl')
y_train = joblib.load('data/processed/train_labels.pkl')
X_val = joblib.load('data/processed/val.pkl')
y_val = joblib.load('data/processed/val_labels.pkl')

# Sample data for faster training (10% of original)
sample_size = int(len(X_train) * 0.1)
print(f"Sampling {sample_size} from {len(X_train)} training samples for faster testing...")
X_train_sample = X_train.iloc[:sample_size]
y_train_sample = y_train.iloc[:sample_size]

def rf_objective(trial):
    """Optuna objective for Random Forest."""
    n_est = trial.suggest_int('n_estimators', 50, 150)  # Reduced range
    depth = trial.suggest_int('max_depth', 5, 15)  # Reduced range
    model = RandomForestClassifier(
        n_estimators=n_est, 
        max_depth=depth, 
        random_state=42, 
        n_jobs=-1)

    print(f"Trial {trial.number}: Testing n_estimators={n_est}, max_depth={depth}")
    start = time.time()

    # Use only 2-fold CV for speed
    score = cross_val_score(model, X_train_sample, y_train_sample, cv=2, scoring='f1').mean()

    print(f"  Score: {score:.4f} (took {time.time()-start:.1f}s)")
    return score

def main(args):
    mlflow.set_experiment("tcc_models_fast")
    with mlflow.start_run():
        # Run Optuna with just 3 trials for quick testing
        print("\nStarting optimization with 3 trials...")
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(rf_objective, n_trials=3)

        print(f"\nBest parameters: {study_rf.best_params}")
        print(f"Best F1 score: {study_rf.best_value:.4f}")

        # Train final model with best params on full data
        print("\nTraining final model on full dataset...")
        best_rf = RandomForestClassifier(**study_rf.best_params, random_state=42)
        best_rf.fit(X_train, y_train)

        # Evaluate on validation set
        from sklearn.metrics import classification_report, confusion_matrix
        y_pred = best_rf.predict(X_val)

        print("\nValidation Results:")
        print(classification_report(y_val, y_pred, target_names=['BENIGN', 'ATTACK']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))

        mlflow.sklearn.log_model(best_rf, "random_forest")
        joblib.dump(best_rf, "models/random_forest.pkl")

        mlflow.log_params(study_rf.best_params)
        print("\nModel saved to models/random_forest.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast model training for testing")
    args = parser.parse_args()
    main(args)