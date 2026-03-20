import torch
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from ctrbp_model import RNAModel, EnhancedRNADataset, config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_and_vocab(model_path, vocab_path):
    with open(vocab_path, 'rb') as f:
        kmer_vocab = pickle.load(f)
    # Ensure that the special mark is present.
    if '<PAD>' not in kmer_vocab:
        kmer_vocab['<PAD>'] = len(kmer_vocab)
    if '<UNK>' not in kmer_vocab:
        kmer_vocab['<UNK>'] = len(kmer_vocab)
    vocab_size = len(kmer_vocab)

    model = RNAModel(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, kmer_vocab


def predict_dataframe(df, model, kmer_vocab, batch_size=32):
    dataset = EnhancedRNADataset(df, kmer_vocab=kmer_vocab, max_length=config.max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            seq = batch['sequence'].to(device)
            struct = batch['structure'].to(device)
            pair = batch['pairing_prob'].to(device)
            logits = model(seq, struct, pair)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)