import argparse
import pickle
import numpy as np
import csv
from feature_engineering import build_features
import os

MODEL_PATH = os.path.join("model", "weights", "K18_ridge_regression_model.pkl")
MEAN_PATH = "../weights/target_mean.npy"
STD_PATH = "../weights/target_std.npy"

PAM_START  = 4
WINDOW_LEN = 34


COMPLEMENT = str.maketrans("ACGT", "TGCA")

def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def find_window(seq: str, pam: str):
    pam = pam.upper()
    seq = seq.upper()

    # pam = reverse_complement(pam)
    # seq = reverse_complement(seq)


    i = seq.find(pam)
    if i == -1:
        raise ValueError(f"PAM {pam} not found in sequence {seq}")

    window_start = i - PAM_START
    window_end   = window_start + WINDOW_LEN

    if window_start < 0:
        raise ValueError(f"PAM found at position {i} but not enough upstream context | (need {PAM_START}nt before PAM, only {i}nt available).")
    if window_end > len(seq):
        raise ValueError(f"PAM found at position {i} but not enough downstream context | (need {window_end}nt total, sequence is {len(seq)}nt).")

    window = seq[window_start:window_end]
    print(f"PAM '{pam}' found at position {i}")

    return window


def predict(sequence: str, pam: str =None):
    sequence = sequence.upper().strip()
    if pam is not None:
        sequence = find_window(sequence, pam)
    elif len(sequence) > WINDOW_LEN:
        sequence[:WINDOW_LEN]

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    t_mean = float(np.load(MEAN_PATH))
    t_std  = float(np.load(STD_PATH))

    features = build_features([sequence], include_one_hot=True).values.astype(np.float32)
    features = np.nan_to_num(features)

    pred_norm = model.predict(features)[0]
    pred_pct  = pred_norm * t_std + t_mean

    return pred_pct

def predict_csv(csv_path: str):
    seq_col = "FOR MODEL - 47 bp match target sequence reverse complement"
    pam_col = "PAM (reverse complement on DNA target sequence)"
    act_col = "Updated QUiCKR Results (March 17)"
    name_col = "gRNA name"

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get(name_col, "").strip()]

    print(f"{'gRNA':<30} {'PAM':<6} {'Predicted':>10} {'Actual':>10}")
    print("-" * 60)

    for row in rows:
        name = row[name_col].strip()
        pam = row[pam_col].strip()
        sequence = row[seq_col].strip()
        actual = row[act_col].strip()

        if not sequence or not pam:
            print(f"{name:<30} {'':6} {'N/A':>10} {actual:>10}  (skipped - missing sequence/PAM)")
            continue


        pred = predict(sequence, pam=pam)
        print(f"{name:<30} {pam:<6} {pred:>9.1f}% {actual:>10}")
    


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("sequence", nargs="?", type=str, default=None)
    p.add_argument("--pam", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    args = p.parse_args()

    if args.csv:
        predict_csv(args.csv)
    elif args.sequence:
        result = predict(args.sequence, pam=args.pam)
        print(f"Predicted indel frequency: {result:.2f}%")
    else:
        p.error("Provided no sequence or -csv <path>")

# python model/scripts/predict.py CCTTTTGGGTGTGGGAGATCTCTGCTTCTGATGGCTCAAACACAGCG --pam TTTG
# python model/scripts/predict.py CTTTGAGGGGACAATTTCAAGGAGTAGTGAAAACAGAAGAACAGAGA --pam TTTC
