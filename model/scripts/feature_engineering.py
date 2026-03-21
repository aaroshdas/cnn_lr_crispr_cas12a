"""
feature calculation + claude suggested features
"""
import pandas as pd
from typing import List


# Nearest-neighbour stacking energies (ΔG, kcal/mol, 37 °C, SantaLucia 1998)
NN_DG = {
    "AA": -1.0, "AT": -0.88, "TA": -0.58, "CA": -1.45,
    "GT": -1.44, "CT": -1.28, "GA": -1.30, "CG": -2.17,
    "GC": -2.24, "GG": -1.84, "AC": -1.44, "TC": -1.28,
    "AG": -1.30, "TG": -1.45, "TT": -1.0,  "CC": -1.84,
}

COMPLEMENT = str.maketrans("ACGT", "TGCA")

def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]

def gc_content(seq: str) -> float:
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq)


def positional_gc(seq: str, window: int = 4) -> List[float]:
    """GC content in non-overlapping windows of length window"""
    seq = seq.upper()
    features = []
    for i in range(0, len(seq) - window + 1, window):
        chunk = seq[i : i + window]
        features.append(gc_content(chunk))
    return features


def mononucleotide_composition(seq: str) -> List[float]:
    """Fraction of each nt"""
    seq = seq.upper()
    n = len(seq)
    return [seq.count(b) / n for b in "ACGT"]


def dinucleotide_composition(seq: str) -> List[float]:
    """Fraction of all dinucleotides"""
    seq = seq.upper()
    n = len(seq) - 1
    dinucs = [a + b for a in "ACGT" for b in "ACGT"]
    return [sum(seq[i : i + 2] == d for i in range(n)) / n for d in dinucs]


def positional_one_hot(seq: str) -> List[float]:
    mapping = {"A": [1,0,0,0], "C": [0,1,0,0], "G": [0,0,1,0], "T": [0,0,0,1]}
    seq = seq.upper()
    encoded = []
    for nt in seq:
        encoded.extend(mapping.get(nt, [0,0,0,0]))
    return encoded


def tm_estimate(seq: str) -> float:
    """Simple Wallace rule Tm (°C). Good proxy for duplex stability. Tm = 2*(A+T) + 4*(G+C)"""
    seq = seq.upper()
    at = seq.count("A") + seq.count("T")
    gc = seq.count("G") + seq.count("C")
    return 2 * at + 4 * gc


def nn_free_energy(seq: str) -> float:
    """Sum of nearest-neighbour ΔG across the sequence (kcal/mol)."""
    seq = seq.upper()
    total = 0.0
    for i in range(len(seq) - 1):
        dinuc = seq[i : i + 2]
        total += NN_DG.get(dinuc, -1.2)# fallback to mean
    return total


def self_complementarity(seq: str) -> float:
    """Fraction of positions that match the reverse complement — crude hairpin/self-folding proxy."""
    seq = seq.upper()
    rc = reverse_complement(seq)
    matches = sum(a == b for a, b in zip(seq, rc))
    return matches / len(seq)


def homopolymer_runs(seq: str, min_run: int = 3) -> int:
    """Count of homopolymer runs of length >= min_run."""
    seq = seq.upper()
    count = 0
    i = 0
    while i < len(seq):
        run = 1
        while i + run < len(seq) and seq[i + run] == seq[i]:
            run += 1
        if run >= min_run:
            count += 1
        i += run
    return count


def cas12a_specific_features(seq: str, pam_start: int = 4, spacer_start: int = 8, spacer_end: int = 28) -> dict:
    """Features specific to the Cas12a
  
    Kim 2018 "Context Sequence" layout (34nt):
        pos 0-3  : 4nt upstream flank
        pos 4-7  : PAM (TTTV for AsCpf1/LbCpf1)
        pos 8-27 : 20nt protospacer / guide
        pos 28-33: downstream flank

    Other stuff:
        TTTV PAM identity
        Seed region (nt 1-5 of spacer) GC
        cleavage-site GC (spacer nt 16-20)
        T-tract in PAM (Cas12a strongly prefers TTTV)
    """
    seq = seq.upper()
    pam = seq[pam_start : pam_start + 4]
    spacer = seq[spacer_start : spacer_end]
    seed = spacer[:5]
    cleavage_region = spacer[15:20]

    return {
        # PAM
        "pam_t_count": pam.count("T"),
        "pam_is_tttv": int(pam[:3] == "TTT"),
        # Spacer overall
        "spacer_gc": gc_content(spacer),
        # Seed region (positions 1-5, critical for Cas12a specificity)
        "seed_gc": gc_content(seed),
        "seed_a_count": seed.count("A"),
        # Cleavage region
        "cleavage_gc": gc_content(cleavage_region),
        # Thermodynamics of spacer
        "spacer_tm": tm_estimate(spacer),
        "spacer_nn_dg": nn_free_energy(spacer),
        # Full sequence
        "full_self_comp": self_complementarity(spacer),
        "homopolymer_count": homopolymer_runs(spacer),
    }



def build_features(sequences: List[str], feature_names: List[str] = None) -> pd.DataFrame:
    """
    Build a feature matrix from a list of gRNA input sequences
    sequences = input
    include_one_hot = only set to true if not using embeddings
    feature_names  = optional list of column names to include; if None, all features are returned
    """
    rows = []
    for seq in sequences:
        seq = seq.upper().strip()
        row = {}

        row["gc_content"] = gc_content(seq)
        row["tm"] = tm_estimate(seq)
        row["nn_dg"] = nn_free_energy(seq)
        row["self_comp"] = self_complementarity(seq)
        row["homopolymer_count"] = homopolymer_runs(seq)

        for nt, val in zip("ACGT", mononucleotide_composition(seq)):
            row[f"mono_{nt}"] = val
        
        if feature_names is None or "di_repeats" not in feature_names:
            for d, val in zip([a + b for a in "ACGT" for b in "ACGT"], dinucleotide_composition(seq)):
                row[f"di_{d}"] = val
        else:
            feature_names = [i for i in feature_names if i != "di_repeats"]

        if feature_names is None or "pos_gc" not in feature_names:
            for i, val in enumerate(positional_gc(seq, window=4)):
                row[f"pgc_w{i}"] = val
        else:
            feature_names = [i for i in feature_names if i != "pos_gc"]

        row.update(cas12a_specific_features(seq))

        upstream = seq[:4].upper()
        for nt in "ACGT":
            row[f"upstream_{nt}"] = upstream.count(nt) / 4

        if feature_names is None or "one_hot" not in feature_names:
            for i, val in enumerate(positional_one_hot(seq)):
                row[f"oh_{i}"] = val
        else:
            feature_names = [i for i in feature_names if i != "one_hot"]

        rows.append(row)

    df = pd.DataFrame(rows)
    if feature_names is not None:
        df = df[feature_names]
    return df

