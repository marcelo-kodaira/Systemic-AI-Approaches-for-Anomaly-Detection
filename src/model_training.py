import argparse
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.optimizers import Adam
import joblib
import mlflow
import mlflow.sklearn
from tensorflow.keras.callbacks import EarlyStopping

# Load data (from preprocessing)
X_train = joblib.load('data/processed/train.pkl')  # Assume loaded
y_train = joblib.load('data/processed/train_labels.pkl')  # Adjust
X_val = joblib.load('data/processed/val.pkl')
y_val = joblib.load('data/processed/val_labels.pkl')

def rf_objective(trial):
    """Optuna objective for Random Forest. Why? Tunes trees/depth for best F1 on validation."""
    n_est = trial.suggest_int('n_estimators', 100, 500)
    depth = trial.suggest_int('max_depth', 5, 20)
    model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='f1').mean()
    return score

def svm_objective(trial):
    """For SVM. Why? Kernel/C tune for non-linear separation in high-dim logs."""
    C = trial.suggest_float('C', 1e-2, 1e2, log=True)
    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
    model = SVC(C=C, kernel=kernel, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='f1').mean()
    return score

def isolation_objective(trial):
    """Unsupervised Isolation Forest. Why? No labels needed; tunes contamination for anomaly fraction."""
    n_est = trial.suggest_int('n_estimators', 100, 300)
    cont = trial.suggest_float('contamination', 0.01, 0.1)
    model = IsolationForest(n_estimators=n_est, contamination=cont, random_state=42)
    preds = model.fit_predict(X_train)  # -1=anomaly
    from sklearn.metrics import f1_score  # Pseudo-F1 if labels available
    # Convert predictions: -1 (anomaly) to 1 (attack), 1 (normal) to 0 (benign)
    preds_binary = (preds == -1).astype(int)
    return f1_score(y_train, preds_binary, average='binary')  # For eval; ignore if pure unsup

def build_lstm(trial, input_shape):
    """Build bi-LSTM. Why? Bidirectional captures forward/backward seq patterns in timed logs."""
    units = trial.suggest_int('units', 64, 128)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    model = Sequential([
        Bidirectional(LSTM(units, return_sequences=True, input_shape=input_shape)),
        Dropout(dropout),
        Bidirectional(LSTM(units)),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=trial.suggest_float('lr', 1e-4, 1e-2, log=True)),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def lstm_objective(trial):
    """Optuna for bi-LSTM. Why? Seq data like request deltas need RNNs; early stop prevents overfit."""
    # Reshape for seq: (samples, timesteps=1, features) - adjust if multi-step
    X_train_seq = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
    X_val_seq = np.reshape(X_val.values, (X_val.shape[0], 1, X_val.shape[1]))
    
    model = build_lstm(trial, (1, X_train.shape[1]))
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train_seq, y_train, epochs=50, batch_size=32, validation_data=(X_val_seq, y_val),
              callbacks=[es], verbose=0)
    preds = (model.predict(X_val_seq) > 0.5).flatten()
    from sklearn.metrics import f1_score
    return f1_score(y_val, preds, average='binary')

def cnn_lstm_objective(trial):
    """Hybrid CNN-LSTM. Why? CNN extracts local patterns (e.g., URI spikes), LSTM seq; ensemble boosts F1."""
    filters = trial.suggest_int('filters', 32, 64)
    kernel = trial.suggest_int('kernel_size', 3, 5)
    X_train_seq = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))  # Timesteps=1 for 1D
    X_val_seq = np.reshape(X_val.values, (X_val.shape[0], 1, X_val.shape[1]))

    model = Sequential([
        Conv1D(filters, kernel_size=kernel, activation='relu', input_shape=(1, X_train.shape[1])),
        GlobalMaxPooling1D(),
        LSTM(128, dropout=0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    # Fit similar to LSTM
    model.fit(X_train_seq, y_train, epochs=10, batch_size=32, validation_data=(X_val_seq, y_val), verbose=0)
    preds = (model.predict(X_val_seq) > 0.5).flatten()
    from sklearn.metrics import f1_score
    return f1_score(y_val, preds, average='binary')

def train_hybrid(trial):
    """Stacking hybrid bi-LSTM + RF. Why? Combines deep seq learning with tree robustness (TCC 4.3)."""
    # Train base models first (simplified; use tuned from above)
    rf_base = RandomForestClassifier(n_estimators=400, random_state=42)
    # lstm_base = build_lstm(...) - wrap in KerasClassifier for stacking
    from sklearn.ensemble import StackingClassifier
    from scikeras.wrappers import KerasClassifier  # pip install scikeras if needed, but assume in reqs
    lstm_base = KerasClassifier(build_fn=build_lstm, epochs=10, batch_size=32, verbose=0)
    stack = StackingClassifier(estimators=[('rf', rf_base), ('lstm', lstm_base)], final_estimator=SVC())
    score = cross_val_score(stack, X_train, y_train, cv=3, scoring='f1_binary').mean()
    return score

def main(args):
    mlflow.set_experiment("tcc_models")
    with mlflow.start_run():
        # Run Optuna for each (reduced trials for testing)
        study_rf = optuna.create_study(direction='maximize')
        study_rf.optimize(rf_objective, n_trials=5)  # Reduced from 30 for faster testing
        best_rf = RandomForestClassifier(**study_rf.best_params, random_state=42)
        best_rf.fit(X_train, y_train)
        mlflow.sklearn.log_model(best_rf, "random_forest")
        joblib.dump(best_rf, "models/random_forest.pkl")
        
        # Repeat for SVM, Isolation (adapt objectives)
        study_svm = optuna.create_study(direction='maximize')
        study_svm.optimize(svm_objective, n_trials=5)  # Reduced from 30 for faster testing
        # ... similarly for others
        
        # For DL: study_lstm.optimize(lstm_objective, n_trials=30)
        # Save: model.save("models/bi_lstm.h5")
        
        # Hybrid (commented out for initial testing)
        # study_hybrid = optuna.create_study(direction='maximize')
        # study_hybrid.optimize(train_hybrid, n_trials=5)
        
        mlflow.log_params(study_rf.best_params)  # Log bests
        print("Training complete. Models saved to models/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and optimize models")
    args = parser.parse_args()
    main(args)