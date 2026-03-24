"""
* had to scale model down cause im travelling so running this on CPU, but I had Claude
reform model to run on google collab with GPU so use that for now

Architecture:
    One-hot
    Multi-scale conv towers
    Residual connections
    Attention pooling
    Handcrafted features at DNN stage

Usage:
    (seq only) python cnn_model/scripts/train_model.py 
    (w/ handcrafted) python cnn_model/scripts/train_model.py --handcrafted
    (checkpoint) python cnn_model/scripts/train_model.py --checkpoint cnn_model/weights/cnn_best.pt
    (eval)  python cnn_model/scripts/train_model.py --eval-only --checkpoint cnn_model/weights/cnn_best.pt

    python cnn_model/scripts/train_model.py --handcrafted --no-cv
"""

import argparse
import sys, os


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../model/scripts'))
import feature_engineering # type: ignore


DATASET_PATH = os.path.join("data", "training_sets", "raw_data")
TRAIN_FILE = "Kim_2018_Train.csv"
TEST_FILE  = "Kim_2018_Test.csv"
WEIGHTS_DIR = os.path.join("cnn_model", "weights")
OUTPUT_DIR = os.path.join("cnn_model", "results")

TARGET_COL = "Indel frequency"
INP_COL = "Context Sequence"


def one_hot_encode(seq: str) -> np.ndarray:
    base_i = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq = seq.upper().strip()
    arr = np.zeros((4, 34), dtype=np.float32)
    for i, base in enumerate(seq[:34]):
        index = base_i.get(base)
        if index is not None:
            arr[index, i] = 1.0
    return arr

class GRNADataset(Dataset):
    def __init__(self, sequences, targets, handcrafted_features=None):
        self.seqs = torch.tensor(np.stack([one_hot_encode(s) for s in sequences])) 
        self.y = torch.tensor(targets.astype(np.float32))
        self.hc = None
        if handcrafted_features is not None:
            self.hc = torch.tensor(handcrafted_features)
    
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        if self.hc is not None:
            return self.seqs[i], self.hc[i], self.y[i]
        return self.seqs[i], self.y[i]

class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AttentionPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Conv1d(channels, 1, 1)

    def forward(self, x):                    
        w = torch.softmax(self.attn(x), dim=-1)
        return (x * w).sum(dim=-1)

class CNN(nn.Module):
    def __init__(self, hc_dim: int = 0, dropout: float = 0.3):
        super().__init__()

        self.tower3 = nn.Sequential(ResConvBlock(4, 64, 3), ResConvBlock(64, 64, 3))
        self.tower5 = nn.Sequential(ResConvBlock(4, 64, 5), ResConvBlock(64, 64, 5))
        self.tower7 = nn.Sequential(ResConvBlock(4, 64, 7), ResConvBlock(64, 64, 7))

        self.merge = ResConvBlock(192, 128, 3)
        self.pool = AttentionPool(128)

        fc_in = 128 + hc_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq, x_hc=None):
        t3 = self.tower3(x_seq)
        t5 = self.tower5(x_seq)
        t7 = self.tower7(x_seq)
        x = torch.cat([t3, t5, t7], dim=1) 
        x = self.merge(x)
        x = self.pool(x)
        if x_hc is not None:
            x = torch.cat([x, x_hc], dim=1)
        return self.fc(x).squeeze(1)

def load_data(path):
    df = pd.read_csv(path).dropna(subset=[TARGET_COL])
    return df.reset_index(drop=True)


def normalize_target(train_df, test_df):
    mean_path = os.path.join(WEIGHTS_DIR, "target_mean.npy")
    std_path = os.path.join(WEIGHTS_DIR, "target_std.npy")
    if os.path.exists(mean_path) and os.path.exists(std_path):
        mean = float(np.load(mean_path))
        std = float(np.load(std_path))
    else:
        mean = train_df[TARGET_COL].mean()
        std = train_df[TARGET_COL].std()
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        np.save(mean_path, mean)
        np.save(std_path,  std)
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[TARGET_COL] = (train_df[TARGET_COL] - mean) / std
    test_df[TARGET_COL] = (test_df[TARGET_COL]  - mean) / std
    return train_df, test_df, mean, std


def build_hc_features(sequences, mean=None, std=None):
    """
    re-use existing feature_engineering pipeline w/o one-hot
    """
    SCALAR_FEATURES = [
        'gc_content', 
        'tm', 
        'nn_dg', 
        'self_comp', 
        'homopolymer_count', 
        'mono_A', 'mono_C', 'mono_G', 'mono_T', 
        'di_repeats', 
        'pos_gc', 
        'pam_t_count', 'pam_is_tttv', 
        'spacer_gc', 
        'seed_gc', 
        'seed_a_count', 
        'cleavage_gc',
        'spacer_tm', 
        'spacer_nn_dg', 
        'full_self_comp', 
        'upstream_A', 'upstream_C', 'upstream_G', 'upstream_T',
    ]
    df = feature_engineering.build_features(sequences)
    cols = [c for c in SCALAR_FEATURES if c in df.columns]
    arr = df[cols].values.astype(np.float32)
    arr = np.nan_to_num(arr)

    if mean is None:
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0,  keepdims=True) + 1e-6

    return (arr - mean) / std, len(cols), mean, std


def evaluate(y_true, y_pred, prefix=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pr,_ = pearsonr(y_true, y_pred)
    sr,_ = spearmanr(y_true, y_pred)
    print(f" {prefix}RMSE={rmse:.4f}  MAE={mae:.4f}  Pearson r={pr:.4f}  Spearman ρ={sr:.4f}")
    return {"rmse": rmse, "mae": mae, "pearson": pr, "spearman": sr}

import time
def train_epoch(model, loader, optimiser, device, use_hc):
    model.train()
    total_loss = 0.0
    t_load, t_forward, t_backward = 0.0, 0.0, 0.0
    t0 = time.time()
    for batch in loader:
        t_load += time.time() - t0

        t1 = time.time()
        if use_hc:
            x_seq, x_hc, y = [b.to(device) for b in batch]
            pred = model(x_seq, x_hc)
        else:
            x_seq, y = [b.to(device) for b in batch]
            pred = model(x_seq)
        t_forward += time.time() - t1

        t2 = time.time()
        loss = F.mse_loss(pred, y)
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        t_backward += time.time() - t2

        total_loss += loss.item() * len(y)
        t0 = time.time()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device, use_hc):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        if use_hc:
            x_seq, x_hc, y = [b.to(device) for b in batch]
            pred = model(x_seq, x_hc)
        else:
            x_seq, y = [b.to(device) for b in batch]
            pred = model(x_seq)
        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--handcrafted", action="store_true", help="Use scalar hand-crafted features at the FC layer")
    p.add_argument("--no-cv", action="store_true")
    p.add_argument("--cv-folds", type=int, default=2)
    p.add_argument("--patience", type=int, default=15, help="early-stopping patience")
    p.add_argument("--checkpoint", type=str, default=None, help="path to .pt checkpoint to resume from")
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR,  exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")


    train_df = load_data(os.path.join(DATASET_PATH, TRAIN_FILE))
    test_df  = load_data(os.path.join(DATASET_PATH, TEST_FILE))
    print(f"[data] train={len(train_df)}  test={len(test_df)}")

    train_df, test_df, t_mean, t_std = normalize_target(train_df, test_df)

    train_seqs = train_df[INP_COL].tolist()
    test_seqs  = test_df[INP_COL].tolist()
    y_train = train_df[TARGET_COL].values.astype(np.float32)
    y_test = test_df[TARGET_COL].values.astype(np.float32)

    # optional hand-crafted features
    hc_train = hc_test = None
    hc_dim = 0
    if args.handcrafted:
        hc_train, hc_dim, hc_mean, hc_std = build_hc_features(train_seqs)
        hc_test,_,_,_ = build_hc_features(test_seqs, mean=hc_mean, std=hc_std)
        print(f"[features] Hand-crafted dim = {hc_dim}")

    train_ds = GRNADataset(train_seqs, y_train, hc_train)
    test_ds  = GRNADataset(test_seqs,  y_test,  hc_test)

    if not args.no_cv:
        print(f"[cv] {args.cv_folds}-fold cross-validation")
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        cv_results = []
        for fold, (tr_idx, val_idx) in enumerate(kf.split(range(len(train_ds))), 1):
            tr_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(tr_idx))
            val_loader= DataLoader(train_ds, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx))

            model = CNN(hc_dim=hc_dim, dropout=args.dropout).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

            best_val, patience_ctr = np.inf, 0
            for epoch in range(1, args.epochs + 1):
                print(f"Fold {fold} | Epoch {epoch}/{args.epochs}")
                train_epoch(model, tr_loader, opt, device, args.handcrafted)
                val_pred, val_true = eval_epoch(model, val_loader, device, args.handcrafted)
                val_loss = mean_squared_error(val_true, val_pred)
                sched.step()
                if val_loss < best_val:
                    best_val = val_loss
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                if patience_ctr >= args.patience:
                    print(f"fold {fold}: early stop at epoch {epoch}")
                    break

            m = evaluate(val_true, val_pred, prefix=f"Fold {fold}: ")
            cv_results.append(m)

        print("[cv] Aggregate:")
        for key in ["rmse","mae","pearson","spearman"]:
            vals = [m[key] for m in cv_results]
            print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        pd.DataFrame(cv_results).to_csv(
            os.path.join(OUTPUT_DIR, "cnn_cv_metrics.csv"), index=False)

    model = CNN(hc_dim=hc_dim, dropout=args.dropout).to(device)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[checkpoint] Loaded {args.checkpoint}")

    if args.eval_only:
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)
        y_pred_norm, y_true_norm = eval_epoch(model, test_loader, device, args.handcrafted)
        evaluate(y_true_norm, y_pred_norm, prefix="Test (norm)")
        evaluate(y_true_norm * t_std + t_mean, y_pred_norm * t_std + t_mean, prefix="Test (%)")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_test_rmse = np.inf
    ckpt_path = os.path.join(WEIGHTS_DIR, "cnn_best.pt")

    print(f"[train] Training max {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, opt, device, args.handcrafted)
        y_pred_norm, y_true_norm = eval_epoch(model, test_loader, device, args.handcrafted)
        rmse = np.sqrt(mean_squared_error(y_true_norm, y_pred_norm))
        sched.step()

        if epoch % 10 == 0 or epoch == 1:
            pr, _ = pearsonr(y_true_norm, y_pred_norm)
            print(f"epoch {epoch:3d}  train_loss={tr_loss:.4f} test_RMSE={rmse:.4f}  Pearson={pr:.4f}")

        if rmse < best_test_rmse:
            best_test_rmse = rmse
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    y_pred_norm, y_true_norm = eval_epoch(model, test_loader, device, args.handcrafted)

    print("[results] final test performance:")
    evaluate(y_true_norm, y_pred_norm, prefix="Normalised")
    evaluate(y_true_norm * t_std + t_mean, y_pred_norm * t_std + t_mean, prefix="Percentage")

    pred_df = pd.DataFrame({
        "context_sequence": test_seqs,
        "y_true_pct":  y_true_norm * t_std + t_mean,
        "y_pred_pct":  y_pred_norm * t_std + t_mean,
    })

    pred_df.to_csv(os.path.join(OUTPUT_DIR, "cnn_predictions.csv"), index=False)
    print(f"{ckpt_path}")
    print(f"{os.path.join(OUTPUT_DIR, 'cnn_predictions.csv')}")


if __name__ == "__main__":
    main()