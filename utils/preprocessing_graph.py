# utils/preprocessing_graph.py

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from itertools import combinations

def build_graph_from_snapshot(df_t: pd.DataFrame, feature_cols=None, edge_strategy='correlation', threshold=0.5):
    """
    Bangun Graph PyG dari data snapshot satu timestep.
    
    Args:
        df_t (pd.DataFrame): Subset data untuk satu timestep, berisi semua ticker
        feature_cols (list): Kolom fitur per node
        edge_strategy (str): 'fully_connected', 'correlation'
        threshold (float): Jika edge_strategy == 'correlation', minimum korelasi
    
    Returns:
        PyG Data object
    """
    tickers = df_t['ticker'].values
    x = torch.tensor(df_t[feature_cols].values, dtype=torch.float)

    edge_index = []
    edge_attr = []

    if edge_strategy == 'fully_connected':
        for i, j in combinations(range(len(tickers)), 2):
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append([1.0])
            edge_attr.append([1.0])

    elif edge_strategy == 'correlation':
        corr_matrix = df_t[feature_cols].T.corr().values
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                if i != j and abs(corr_matrix[i, j]) >= threshold:
                    edge_index.append([i, j])
                    edge_attr.append([corr_matrix[i, j]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).T  # shape [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        tickers=tickers
    )

    return data

def generate_graph_sequence(df: pd.DataFrame, feature_cols=None, edge_strategy='correlation', threshold=0.5) -> list:
    """
    Proses seluruh DataFrame multi-time menjadi list of graphs per timestep.

    Args:
        df (pd.DataFrame): Data seluruh waktu, kolom ['date', 'ticker', features...]

    Returns:
        list of PyG Data objects
    """
    df = df.copy()
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in ['date', 'ticker']]

    graphs = []
    for date, df_t in df.groupby('date'):
        if df_t.isnull().any().any():
            continue  # Skip jika ada NaN

        g = build_graph_from_snapshot(df_t, feature_cols, edge_strategy, threshold)
        g.date = date  # bisa dipakai untuk logging nanti
        graphs.append(g)

    return graphs
