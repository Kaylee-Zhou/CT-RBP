#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# -------------------- Configuration class --------------------
class FixedConfig:
    max_length = 101
    batch_size = 32
    data_path = 'HNRNPC'

    # Model parameters
    d_model = 128
    nhead = 8
    num_layers = 3
    dropout = 0.3
    structure_embed_dim = 16
    pairing_embed_dim = 16
    conv_out_channels = 64
    conv_kernel_size = 5

    # Classifier dimension
    classifier_hidden_dims = [64, 32]

config = FixedConfig()

# -------------------- Data set class --------------------
class EnhancedRNADataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, kmer_vocab=None, max_length=101, augment=False):
        self.data = dataframe
        self.max_length = max_length
        self.augment = augment

        if kmer_vocab is None:
            self.kmer_vocab = self._build_vocab()
        else:
            self.kmer_vocab = kmer_vocab

        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        if self.pad_token not in self.kmer_vocab:
            self.kmer_vocab[self.pad_token] = len(self.kmer_vocab)
        if self.unk_token not in self.kmer_vocab:
            self.kmer_vocab[self.unk_token] = len(self.kmer_vocab)

        self.vocab_size = len(self.kmer_vocab)

    def _build_vocab(self):
        all_kmers = []
        for seq in self.data['sequence']:
            kmers = seq.split()
            all_kmers.extend(kmers)
        kmer_counts = Counter(all_kmers)
        return {kmer: idx for idx, (kmer, _) in enumerate(kmer_counts.most_common())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        sequence = row['sequence']
        kmers = sequence.split()
        seq_indices = [self.kmer_vocab.get(kmer, self.kmer_vocab[self.unk_token])
                      for kmer in kmers[:self.max_length]]

        if len(seq_indices) < self.max_length:
            seq_indices = seq_indices + [self.kmer_vocab[self.pad_token]] * (self.max_length - len(seq_indices))
        else:
            seq_indices = seq_indices[:self.max_length]

        structure = row['structure']
        if isinstance(structure, str):
            structure = eval(structure)
        if len(structure) < self.max_length:
            structure = structure + [0] * (self.max_length - len(structure))
        else:
            structure = structure[:self.max_length]

        pairing_prob = row['pairing_probabilities']
        if isinstance(pairing_prob, str):
            pairing_prob = eval(pairing_prob)
        if len(pairing_prob) < self.max_length:
            pairing_prob = pairing_prob + [0.0] * (self.max_length - len(pairing_prob))
        else:
            pairing_prob = pairing_prob[:self.max_length]

        label = row.get('label', 0)

        sequence_tensor = torch.tensor(seq_indices, dtype=torch.long)
        structure_tensor = torch.tensor(structure, dtype=torch.long)
        pairing_prob_tensor = torch.tensor(pairing_prob, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.float)

        return {
            'sequence': sequence_tensor,
            'structure': structure_tensor,
            'pairing_prob': pairing_prob_tensor,
            'label': label_tensor
        }

# -------------------- Position encoding --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# -------------------- Main model --------------------
class RNAModel(nn.Module):
    def __init__(self, vocab_size):
        super(RNAModel, self).__init__()

        self.sequence_embedding = nn.Embedding(vocab_size, config.d_model)
        self.structure_embedding = nn.Embedding(3, config.structure_embed_dim)
        self.pairing_prob_projection = nn.Linear(1, config.pairing_embed_dim)

        conv_input_dim = config.d_model + config.structure_embed_dim + config.pairing_embed_dim
        self.conv1 = nn.Conv1d(conv_input_dim, config.conv_out_channels,
                              kernel_size=config.conv_kernel_size, padding=config.conv_kernel_size//2)
        self.conv2 = nn.Conv1d(config.conv_out_channels, config.conv_out_channels*2,
                              kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(config.conv_out_channels)
        self.bn2 = nn.BatchNorm1d(config.conv_out_channels*2)
        self.conv_activation = nn.ReLU()
        self.conv_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.conv_out_channels*2,
            nhead=config.nhead,
            dim_feedforward=config.conv_out_channels*4,
            dropout=config.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.pos_encoding = PositionalEncoding(config.conv_out_channels*2, config.max_length)

        classifier_layers = []
        input_dim = config.conv_out_channels*2
        for hidden_dim in config.classifier_hidden_dims:
            classifier_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            input_dim = hidden_dim
        classifier_layers.append(nn.Linear(input_dim, 1))
        self.classifier = nn.Sequential(*classifier_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, sequence, structure, pairing_prob):
        batch_size = sequence.size(0)

        seq_emb = self.sequence_embedding(sequence)
        structure_emb = self.structure_embedding(structure)
        pairing_emb = self.pairing_prob_projection(pairing_prob.unsqueeze(-1))

        combined_emb = torch.cat([seq_emb, structure_emb, pairing_emb], dim=-1)

        conv_input = combined_emb.transpose(1, 2)
        conv1_out = self.conv1(conv_input)
        conv1_out = self.bn1(conv1_out)
        conv1_out = self.conv_activation(conv1_out)
        conv1_out = self.conv_dropout(conv1_out)

        conv2_out = self.conv2(conv1_out)
        conv2_out = self.bn2(conv2_out)
        conv2_out = self.conv_activation(conv2_out)
        conv2_out = self.conv_dropout(conv2_out)

        transformer_input = conv2_out.transpose(1, 2)  # (batch, seq_len, d_model)
        transformer_input = transformer_input.transpose(0, 1)  # (seq_len, batch, d_model)
        transformer_input = self.pos_encoding(transformer_input)
        transformer_out = self.transformer_encoder(transformer_input)
        transformer_out = transformer_out.transpose(0, 1)  # (batch, seq_len, d_model)

        pooled = transformer_out.mean(dim=1)  # (batch, d_model)

        logits = self.classifier(pooled)
        return logits.squeeze()