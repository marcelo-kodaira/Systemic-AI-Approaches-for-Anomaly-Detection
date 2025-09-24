import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

def load_dataset(dataset_name):
    """Load raw CSV from data/raw/. Why? Datasets are public CSVs; this abstracts loading."""
    if dataset_name == "cic":
        # For CIC-IDS dataset, load all CSV files from the subdirectory
        cic_path = "data/raw/cic-ids/MachineLearningCSV"
        if not os.path.exists(cic_path):
            raise FileNotFoundError(f"CIC-IDS dataset not found in {cic_path}")

        csv_files = [f for f in os.listdir(cic_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {cic_path}")

        dfs = []
        for csv_file in csv_files:
            file_path = os.path.join(cic_path, csv_file)
            print(f"Loading {csv_file}...")
            df_temp = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
            dfs.append(df_temp)

        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df)} total records from CIC-IDS dataset")
        return df
    else:
        path = f"data/raw/{dataset_name}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Download {dataset_name} to data/raw/")
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} records from {dataset_name}")
        return df

def clean_data(df):
    """Clean: Drop corrupt rows, fill missing values. Why? Real logs have noise; this ensures model stability."""
    df = df.dropna(thresh=len(df.columns)*0.8)  # Drop rows with >20% missing
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    return df

def normalize_and_encode(df):
    """Normalize numerics [0,1], one-hot categoricals. Why? Models like SVM/LSTM sensitive to scales; one-hot for non-numeric."""
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    scaler = MinMaxScaler()
    # CIC-IDS numeric columns (key flow features)
    num_cols = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
                'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean']

    # Check which columns actually exist in the dataframe
    existing_num_cols = [col for col in num_cols if col in df.columns]
    if existing_num_cols:
        # Replace infinite values with NaN, then fill with 0
        df[existing_num_cols] = df[existing_num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        df[existing_num_cols] = scaler.fit_transform(df[existing_num_cols])

    # CIC-IDS doesn't have categorical columns that need encoding
    # Label is the target variable (BENIGN or attack type)
    return df

def engineer_features(df):
    """Add temporal/semantic features. Why? Captures patterns like bursty attacks (TCC 3.3)."""
    # Clean column names
    df.columns = df.columns.str.strip()

    # CIC-IDS specific features
    # Calculate packet ratio features
    df['fwd_bwd_packet_ratio'] = df['Total Fwd Packets'] / (df['Total Backward Packets'] + 1)

    # Calculate byte ratio features
    df['fwd_bwd_bytes_ratio'] = df['Total Length of Fwd Packets'] / (df['Total Length of Bwd Packets'] + 1)

    # Flow duration features (avoid division by zero)
    df['packets_per_second'] = (df['Total Fwd Packets'] + df['Total Backward Packets']) / (df['Flow Duration'] + 1)
    df['bytes_per_second'] = (df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']) / (df['Flow Duration'] + 1)

    # Flag-based features
    flag_cols = ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count']
    existing_flag_cols = [col for col in flag_cols if col in df.columns]
    if existing_flag_cols:
        df['total_flags'] = df[existing_flag_cols].sum(axis=1)

    # Replace infinite values with 0
    df = df.replace([np.inf, -np.inf], 0)

    return df

def split_data(df, target_col='Label'):  # CIC-IDS uses 'Label' column
    """Stratified split 70/15/15. Why? Balanced classes for fair training."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Convert labels to binary (0 for BENIGN, 1 for any attack)
    y_binary = (y != 'BENIGN').astype(int)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y_binary, test_size=0.15, stratify=y_binary, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def main(args):
    df = load_dataset(args.dataset)
    df = clean_data(df)
    df = normalize_and_encode(df)
    df = engineer_features(df)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(df)

    # Save processed data with both features and labels
    X_train.to_pickle('data/processed/train.pkl')
    y_train.to_pickle('data/processed/train_labels.pkl')

    X_val.to_pickle('data/processed/val.pkl')
    y_val.to_pickle('data/processed/val_labels.pkl')

    X_test.to_pickle('data/processed/test.pkl')
    y_test.to_pickle('data/processed/test_labels.pkl')

    print("Preprocessing complete. Files saved to data/processed/")
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess web log datasets")
    parser.add_argument("--dataset", choices=["nyu", "cic"], required=True, help="Dataset name")
    args = parser.parse_args()
    main(args)