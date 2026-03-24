import argparse
import pickle
import numpy as np
import csv

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model/scripts'))
from feature_engineering import build_features # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cnn_model/scripts'))
from train_model import CNN, one_hot_encode # type: ignore

import pandas as pd
import torch


REG_WEIGHTS_PATH = os.path.join("model", "weights")
CNN_WEIGHTS_PATH = os.path.join("cnn_model", "weights")

REG_MODEL_PATH = os.path.join(REG_WEIGHTS_PATH, "K18_ridge_regression_model.pkl")
CNN_MODEL_PATH = os.path.join(CNN_WEIGHTS_PATH, "cnn_best.pt")

MEAN_PATH = "target_mean.npy"
STD_PATH = "target_std.npy"


OUTPUT_DIR = os.path.join("prediction_results")

PAM_START  = 4
WINDOW_LEN = 34


def find_window(seq: str, pam: str):
    pam = pam.upper()
    seq = seq.upper()

    i = seq[4:].find(pam)
    if i == -1:
        # print(f"PAM {pam} not found in sequence {seq}")
        return None
    i+=4

    window_start = i - PAM_START
    window_end   = window_start + WINDOW_LEN

    if window_start < 0:
        print(f'PAM found at position {i} but not enough upstream context | (need {PAM_START}nt before PAM, only {i}nt available).')
        return None
    if window_end > len(seq):
        print(f"PAM found at position {i} but not enough downstream context | (need {window_end}nt total, sequence is {len(seq)}nt).")
        return None

    window = seq[window_start:window_end]
    # print(f"PAM '{pam}' found at position {i}")

    return window

def predict(sequence: str, pam: str =None, model: str = "reg"):
    sequence = sequence.upper().strip()
    if pam is not None:
        sequence = find_window(sequence, pam)
        if sequence is None:
            return None
    elif len(sequence) > WINDOW_LEN:
        sequence[:WINDOW_LEN]

    if model == "reg":
        return reg_predict(sequence)
    elif model == "cnn":
        return cnn_predict(sequence)
    else:
        raise ValueError(f"model type: {model}")
    
def reg_predict(sequence: str):
    t_mean = float(np.load(os.path.join(REG_WEIGHTS_PATH, MEAN_PATH)))
    t_std  = float(np.load(os.path.join(REG_WEIGHTS_PATH, STD_PATH)))

    with open(REG_MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    features_path = os.path.join(REG_WEIGHTS_PATH, "feature_names.json")
    with open(features_path) as f:
        active_features= json.load(f)

    features = build_features([sequence], features_names=active_features).values.astype(np.float32)
    features = np.nan_to_num(features)

    pred_norm = model.predict(features)[0]
    pred_pct  = pred_norm * t_std + t_mean

    return pred_pct

def cnn_predict(sequence: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CNN(hc_dim=0).to(device)
    state = torch.load(CNN_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    t_mean = float(np.load(os.path.join(CNN_WEIGHTS_PATH, MEAN_PATH)))
    t_std  = float(np.load(os.path.join(CNN_WEIGHTS_PATH, STD_PATH)))


    device = next(model.parameters()).device
    x = torch.tensor(one_hot_encode(sequence)).unsqueeze(0).to(device)  # (1, 4, 34)
    with torch.no_grad():
        pred_norm = model(x).item()
    return pred_norm * t_std + t_mean


def predict_csv(csv_path: str, model: str):
    seq_col = "FOR MODEL - 47 bp match target sequence reverse complement"
    pam_col = "PAM (reverse complement on DNA target sequence)"
    act_col = "Updated QUiCKR Results (March 17)"
    name_col = "gRNA name"

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get(name_col, "").strip()]

    print(f"{'gRNA':<30} {'PAM':<6} {'Predicted':>10} {'Actual':>10}")
    print("-" * 60)
    result_df= pd.DataFrame(columns=["sequence", "Predicted", "Actual"])

    for row in rows:
        name = row[name_col].strip()
        pam = row[pam_col].strip()
        sequence = row[seq_col].strip()
        actual = row[act_col].strip()

        if not sequence or not pam:
            print(f"{name:<30} {'':6} {'N/A':>10} {actual:>10}  (skipped - missing sequence/PAM)")
            result_df.loc[len(result_df)] = {"sequence": sequence, "Predicted": -1, "Actual": actual}
        
            continue

        pred = predict(sequence, pam=pam, model=model)
        
        if pred != None:
            result_df.loc[len(result_df)] = {"sequence": sequence, "Predicted": pred, "Actual": actual}
            print(f"{name:<30} {pam:<6} {pred:>9.1f}% {actual:>10}")
            
        else:
            result_df.loc[len(result_df)] = {"sequence": sequence, "Predicted": -1, "Actual": actual}
            print(f"Error in predicting {name}, skipping")

        result_df["model"] = model
        result_df.to_csv(os.path.join(OUTPUT_DIR, "QKR_predictions.csv"), index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("sequence", nargs="?", type=str, default=None)
    p.add_argument("--pam", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--model", choices=["reg", "cnn"], default="reg")
    args = p.parse_args()

    if args.csv:
        predict_csv(args.csv, args.model)
    elif args.sequence:
        result = predict(args.sequence, pam=args.pam, model=args.model)
        print(f"Predicted indel frequency: {result:.2f}%")
    else:
        p.error("Provided no sequence or csv")
    

# python predict.py CCTTTTGGGTGTGGGAGATCTCTGCTTCTGATGGCTCAAACACAGCG --pam TTTG
# python predict.py CTTTGAGGGGACAATTTCAAGGAGTAGTGAAAACAGAAGAACAGAGA --pam TTTC
# python predict.py --csv data/quickr_sets/quickr_raw_sequences.csv --model reg

