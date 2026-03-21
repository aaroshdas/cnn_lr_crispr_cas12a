import argparse
import pickle
import numpy as np
from grna_features import build_features

MODEL_PATH  = "./linear_results/ridge_model.pkl"
MEAN_PATH   = "./weights/target_mean.npy"
STD_PATH    = "./weights/target_std.npy"

EXPECTED_LEN = 34  # length the model was trained on

def truncate_sequence(seq: str, target_len: int = EXPECTED_LEN) -> str:
    if len(seq) > target_len:
        truncated = seq[:target_len]
        print(f"  Sequence too long ({len(seq)}nt), truncated to {target_len}nt: {truncated}")
        return truncated
    elif len(seq) < target_len:
        raise ValueError(f"Sequence too short ({len(seq)}nt) — expected {target_len}nt minimum.")
    return seq

def predict(sequence: str) -> float:
    sequence = sequence.upper().strip()
    sequence = truncate_sequence(sequence)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    t_mean = float(np.load(MEAN_PATH))
    t_std  = float(np.load(STD_PATH))

    features = build_features([sequence], include_one_hot=True).values.astype(np.float32)
    features = np.nan_to_num(features)

    pred_norm = model.predict(features)[0]
    pred_pct  = pred_norm * t_std + t_mean

    return pred_pct

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("sequence", type=str, help="Input sequence (will be truncated to 34nt if longer)")
    args = p.parse_args()

    result = predict(args.sequence)
    print(f"Predicted indel frequency: {result:.2f}%")