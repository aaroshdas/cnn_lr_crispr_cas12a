"""

python model/scripts/train_regression.py 
python model/scripts/train_regression.py --no-embed --include-onehot = hand-crafted features only
python model/scripts/train_regression.py --no-handcrafted = embeddings only
python model/scripts/train_regression.py --layer mean = mean pooling instead of CLS
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


import create_embeddings
import feature_engineering

# PATHS
DATASET_PATH = os.path.join("data", "training_sets", "raw_data")
TRAIN_FILE = "Kim_2018_Train.csv"
TEST_FILE = "Kim_2018_Test.csv"

WEIGHTS_DIR = os.path.join("model", "weights")
OUTPUT_DIR = os.path.join("model", "results")

TARGET_COL = "Indel frequency"
INP_COL = "Context Sequence"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET_COL])

    return df.reset_index(drop=True)


def get_sequences(df: pd.DataFrame) -> list:
    return df[INP_COL].tolist()
  


def normalize_target(train_df, test_df):
    """Z-score normalise target + reuse saved stats if available."""
    mean_path = os.path.join(WEIGHTS_DIR, "target_mean.npy")
    std_path  = os.path.join(WEIGHTS_DIR, "target_std.npy")

    if os.path.exists(mean_path) and os.path.exists(std_path):
        mean = float(np.load(mean_path))
        std  = float(np.load(std_path))
    else:
        mean = train_df[TARGET_COL].mean()
        std  = train_df[TARGET_COL].std()
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        np.save(mean_path, mean)
        np.save(std_path, std)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[TARGET_COL] = (train_df[TARGET_COL] - mean) / std
    test_df[TARGET_COL]  = (test_df[TARGET_COL]  - mean) / std
    return train_df, test_df, mean, std


def evaluate(y_true, y_pred, prefix=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    print(f" {prefix}RMSE={rmse:.4f}  MAE={mae:.4f}  " f"Pearson r={pr:.4f}  Spearman ρ={sr:.4f}")
    return {"rmse": rmse, "mae": mae, "pearson": pr, "spearman": sr}


def assemble_features(sequences, args):
    parts = []

    if not args.no_handcrafted:
        print("[features] Building generated features")
        print(f"[features] Include one hot: {args.include_onehot}")
        hc = feature_engineering.build_features(sequences, include_one_hot=args.include_onehot).values.astype(np.float32)
        hc = np.nan_to_num(hc)
        parts.append(hc)

    if not args.no_embed:
        emb = create_embeddings.get_embeddings(sequences, layer=args.layer, method=args.embedding_method)
        parts.append(emb.astype(np.float32))


    x = np.hstack(parts)
    print(f"[features] Final feature matrix = {x.shape}")
    return x


def cross_validate(x, y, model_class, param_grid, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    print(f"\n[cv] {n_splits}-fold cross-validation ...")
    for fold, (train_i, val_i) in enumerate(kf.split(x), 1):
        x_train, x_val = x[train_i], x[val_i]
        y_train, y_val = y[train_i], y[val_i]

        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model_class())])

        gs = GridSearchCV(pipeline, {f"model__{k}": v for k, v in param_grid.items()}, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1,)
        gs.fit(x_train, y_train)
        preds = gs.predict(x_val)

        m = evaluate(y_val, preds, prefix=f"Fold {fold}: ")
        m["best_params"] = gs.best_params_
        fold_metrics.append(m)

    print("[cv] Aggregate CV results:")
    for key in ["rmse", "mae", "pearson", "spearman"]:
        vals=[]
        for m in fold_metrics:
            vals.append(m[key])
        print(f"{key}: {np.mean(vals):.4f} {np.std(vals):.4f}")

    return fold_metrics


def train_final_model(x_train, y_train, model_class, param_grid):
    pipe = Pipeline([("scaler", StandardScaler()), ("model",  model_class())])
    gs = GridSearchCV(pipe, {f"model__{k}": v for k, v in param_grid.items()}, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1,)
    gs.fit(x_train, y_train)
    print(f"[train] Best params: {gs.best_params_}")
    return gs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="cls", choices=["cls", "mean", "last_mean"])
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--no-handcrafted", action="store_true")
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--embedding-method", default="kmer", choices=["kmer", "dnabert2"])
    parser.add_argument("--include-onehot", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = load_data(os.path.join(DATASET_PATH, TRAIN_FILE))
    test_df  = load_data(os.path.join(DATASET_PATH, TEST_FILE))
    print(f"train: {len(train_df)} | test: {len(test_df)}")

    train_df, test_df, t_mean, t_std = normalize_target(train_df, test_df)

    train_seqs = get_sequences(train_df)
    test_seqs = get_sequences(test_df)
    y_train = train_df[TARGET_COL].values.astype(np.float32)
    y_test = test_df[TARGET_COL].values.astype(np.float32)

    x_train = assemble_features(train_seqs, args)
    x_test  = assemble_features(test_seqs,  args)

    model_class = Ridge
    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}

    if not args.no_cv:
        fold_metrics = cross_validate(x_train, y_train, model_class, param_grid)
        pd.DataFrame(fold_metrics).to_csv(os.path.join(OUTPUT_DIR, "cv_metrics.csv"), index=False)


    final_model = train_final_model(x_train, y_train, model_class, param_grid)

    model_path = os.path.join(WEIGHTS_DIR, f"K18_ridge_regression_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    y_pred = final_model.predict(x_test)
    test_metrics = evaluate(y_test, y_pred, prefix="Test ")

    y_pred_raw = y_pred * t_std + t_mean
    y_test_raw = y_test * t_std + t_mean
    mae_raw = mean_absolute_error(y_test_raw, y_pred_raw)
    print(f'Test MAE = {mae_raw:.4f}')

    pred_df = pd.DataFrame({
        "context_seq": test_seqs,
        "guide_seq": test_df[INP_COL].tolist(),
        "y_true_norm": y_test,
        "y_pred_norm": y_pred,
        "y_true_pct": y_test_raw,
        "y_pred_pct": y_pred_raw,
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "K18_predictions.csv"), index=False)

    summary = {**vars(args), **{f"test_{k}": v for k, v in test_metrics.items()}}
    pd.DataFrame([summary]).to_csv(os.path.join(OUTPUT_DIR, "K18_run_summary.csv"), index=False)


if __name__ == "__main__":
    main()