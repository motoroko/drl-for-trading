import pandas as pd
from finta import TA

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan indikator teknikal ke DataFrame multi-ticker.
    Data akan otomatis diproses per ticker berdasarkan kolom 'Tic'.
    """

    def _add_ta(sub_df):
        sub_df = sub_df.copy()
        sub_df.columns = [c.lower() for c in sub_df.columns]  # lowercase agar kompatibel

        # Tambahkan indikator
        sub_df['RSI'] = TA.RSI(sub_df)
        macd = TA.MACD(sub_df)
        sub_df['MACD'] = macd['MACD']
        sub_df['MACD_SIGNAL'] = macd['SIGNAL']
        sub_df['EMA_20'] = TA.EMA(sub_df, 20)
        sub_df['EMA_50'] = TA.EMA(sub_df, 50)
        sub_df['STOCH_K'] = TA.STOCH(sub_df)
        sub_df['STOCH_D'] = TA.STOCHD(sub_df)
        bb = TA.BBANDS(sub_df)
        sub_df['BB_UPPER'] = bb['BB_UPPER']
        sub_df['BB_LOWER'] = bb['BB_LOWER']

        # Drop baris dengan NaN (awal rolling window)
        return sub_df.dropna()

    # Jika kolom 'Tic' ada, group by; jika tidak, proses langsung
    if 'ticker' in df.columns:
        return (
            df.groupby('ticker', group_keys=False)
              .apply(_add_ta)
              .reset_index(drop=True)
        )
    else:
        return _add_ta(df).reset_index(drop=True)