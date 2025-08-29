# utils/preprocessing.py

import pandas as pd
import numpy as np

def preprocess_flat(df: pd.DataFrame, feature_cols=None, normalize=True):
    """
    Preprocess data menjadi flat state vector untuk PPO tabular, per ticker.
    Kompatibel dengan Pandas <2.3 (tanpa include_group).
    """
    df = df.copy()

    if feature_cols is None:
        feature_cols = [
            col for col in df.columns 
            if col not in ['ticker', 'date']
        ]

    def _process_ticker(ticker_df):
        # Ambil nilai ticker secara eksplisit
        ticker = ticker_df['ticker'].iloc[0]
        ticker_df = ticker_df.sort_values('date')

        if normalize:
            for col in feature_cols:
                ticker_df[col] = (ticker_df[col] - ticker_df[col].mean()) / (ticker_df[col].std() + 1e-8)

        ticker_df['ticker'] = ticker  # pastikan kolom 'ticker' tetap ada
        return ticker_df[['date','ticker'] + feature_cols]

    df_processed = (
        df.groupby('ticker', group_keys=False)
          .apply(_process_ticker)
          .reset_index(drop=True)
    )

    return df_processed
