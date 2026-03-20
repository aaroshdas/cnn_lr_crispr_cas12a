"""
Extract sequence embeddings using dnabert-2
"""

import numpy as np
import torch
from typing import List
from tqdm import tqdm


def get_dnabert2_embeddings(sequences: List[str], model_name: str = "zhihan1996/DNABERT-2-117M", batch_size: int = 32, max_length: int = 128, layer: str = "cls"):
    """
    model_name = huggingface model id
    layer = pooling strategy
        "cls" --> token hidden state
        "mean" --> mean of all token hidden states
        "last_mean" -> mean of last 4 layers
    """
    from transformers import AutoTokenizer, AutoModel

    if torch.cuda.is_available():
        device = "cuda" 
    else:
        device = "cpu"

    print(f"[Embeddings] Loading {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.to(device)

    all_embeddings = []

    for start in tqdm(range(0, len(sequences), batch_size), desc="Embedding batches"):
        batch = sequences[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=(layer == "last_mean"))

        if layer == "cls":
            # last_hidden_state[:, 0, :] is the [CLS] token
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        elif layer == "mean":
            # mean-pool over non-padding tokens
            mask = inputs["attention_mask"].unsqueeze(-1).float().cpu()
            hidden = outputs.last_hidden_state.cpu()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            emb = emb.numpy()
        elif layer == "last_mean":
            # mean of last 4 hidden layers, then mean-pool over tokens
            hidden_states = outputs.hidden_states 
            last4 = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)
            mask = inputs["attention_mask"].unsqueeze(-1).float().cpu()
            emb = (last4.cpu() * mask).sum(dim=1) / mask.sum(dim=1)
            emb = emb.numpy()

        all_embeddings.append(emb)

    return np.vstack(all_embeddings)