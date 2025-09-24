import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_data_temporal():
    """Load data and simulate temporal splits for drift detection."""
    X_train = joblib.load('data/processed/train.pkl')
    X_val = joblib.load('data/processed/val.pkl')
    X_test = joblib.load('data/processed/test.pkl')

    # Simulate temporal data (divide test set into time windows)
    window_size = len(X_test) // 5  # 5 time windows
    windows = []

    for i in range(5):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, len(X_test))
        window_data = X_test.iloc[start_idx:end_idx]
        windows.append(window_data)

    return X_train, windows

def kolmogorov_smirnov_test(reference_data, current_data, feature_names):
    """KS test for distribution drift detection."""
    drift_scores = {}

    for feature in feature_names:
        ref_values = reference_data[feature].values
        curr_values = current_data[feature].values

        # KS test
        ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)

        drift_scores[feature] = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'drift_detected': p_value < 0.05  # Significance level
        }

    return drift_scores

def population_stability_index(reference_data, current_data, feature_names, n_bins=10):
    """Calculate PSI for feature drift."""
    psi_scores = {}

    for feature in feature_names:
        ref_values = reference_data[feature].values
        curr_values = current_data[feature].values

        # Create bins based on reference data
        _, bin_edges = np.histogram(ref_values, bins=n_bins)

        # Calculate distributions
        ref_hist, _ = np.histogram(ref_values, bins=bin_edges)
        curr_hist, _ = np.histogram(curr_values, bins=bin_edges)

        # Normalize
        ref_hist = ref_hist / len(ref_values) + 1e-10
        curr_hist = curr_hist / len(curr_values) + 1e-10

        # Calculate PSI
        psi = np.sum((curr_hist - ref_hist) * np.log(curr_hist / ref_hist))

        psi_scores[feature] = {
            'psi': psi,
            'drift_level': 'No drift' if psi < 0.1 else 'Moderate' if psi < 0.25 else 'Significant'
        }

    return psi_scores

def visualize_drift(drift_scores, psi_scores):
    """Create drift visualization plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. KS test p-values
    features = list(drift_scores.keys())[:10]  # Top 10 features
    p_values = [drift_scores[f]['p_value'] for f in features]

    axes[0].bar(range(len(features)), p_values)
    axes[0].axhline(y=0.05, color='r', linestyle='--', label='Significance threshold')
    axes[0].set_xlabel('Feature Index')
    axes[0].set_ylabel('P-value')
    axes[0].set_title('Kolmogorov-Smirnov Test Results')
    axes[0].legend()

    # 2. PSI scores
    features_psi = list(psi_scores.keys())[:10]
    psi_values = [psi_scores[f]['psi'] for f in features_psi]

    colors = ['green' if x < 0.1 else 'yellow' if x < 0.25 else 'red' for x in psi_values]
    axes[1].bar(range(len(features_psi)), psi_values, color=colors)
    axes[1].axhline(y=0.1, color='g', linestyle='--', label='No drift')
    axes[1].axhline(y=0.25, color='r', linestyle='--', label='Significant drift')
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('PSI Score')
    axes[1].set_title('Population Stability Index')
    axes[1].legend()

    plt.suptitle('Drift Detection Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/drift_analysis.png', dpi=300)
    print("Drift analysis plot saved to figures/drift_analysis.png")

def generate_drift_report(drift_scores, psi_scores):
    """Generate drift detection report."""
    report = []
    report.append("="*60)
    report.append("DRIFT DETECTION REPORT")
    report.append("="*60)

    # KS Test summary
    drift_count = sum([1 for d in drift_scores.values() if d['drift_detected']])
    report.append(f"\n1. Kolmogorov-Smirnov Test:")
    report.append(f"   - Features with drift: {drift_count}/{len(drift_scores)}")
    report.append(f"   - Drift percentage: {(drift_count/len(drift_scores)*100):.1f}%")

    # PSI summary
    report.append(f"\n2. Population Stability Index:")
    no_drift = sum([1 for p in psi_scores.values() if p['drift_level'] == 'No drift'])
    moderate = sum([1 for p in psi_scores.values() if p['drift_level'] == 'Moderate'])
    significant = sum([1 for p in psi_scores.values() if p['drift_level'] == 'Significant'])

    report.append(f"   - No drift: {no_drift} features")
    report.append(f"   - Moderate drift: {moderate} features")
    report.append(f"   - Significant drift: {significant} features")

    # Recommendations
    report.append("\n3. Recommendations:")
    if drift_count > len(drift_scores) * 0.3:
        report.append("   [WARNING] Significant drift detected - consider model retraining")
    elif drift_count > len(drift_scores) * 0.1:
        report.append("   [CAUTION] Moderate drift detected - monitor closely")
    else:
        report.append("   [OK] Model is stable - no immediate action required")

    return "\n".join(report)

def main(args):
    print("Loading data...")
    X_train, windows = load_data_temporal()

    print("Running drift detection analysis...")

    # Use first window as reference
    reference_data = windows[0]
    feature_names = reference_data.columns[:20]  # Top 20 features

    # Analyze last window for drift
    current_window = windows[-1]

    # KS Test
    drift_scores = kolmogorov_smirnov_test(reference_data, current_window, feature_names)

    # PSI
    psi_scores = population_stability_index(reference_data, current_window, feature_names)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_drift(drift_scores, psi_scores)

    # Generate report
    report = generate_drift_report(drift_scores, psi_scores)
    print("\n" + report)

    # Save report
    with open('figures/drift_report.txt', 'w') as f:
        f.write(report)

    print("\n[SUCCESS] Drift detection complete! Check figures/ for visualizations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect concept drift")
    args = parser.parse_args()
    main(args)