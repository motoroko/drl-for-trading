import pandas as pd
import yfinance as yf
from pandas_datareader import data as wb
from datetime import datetime
from utils.indicators import add_indicators

def fetch_data_yf(
    ticker: str,
    start_date: str = "2015-01-01",
    end_date: str = None,
    with_indicators: bool = False
) -> pd.DataFrame:
    """
    Ambil data dari yfinance, bersihkan, dan tambahkan indikator teknikal.

    Args:
        ticker (str): Ticker saham, contoh "TLKM.JK"
        start_date (str): Tanggal mulai dalam format YYYY-MM-DD
        end_date (str): Tanggal akhir, default = hari ini
        with_indicators (bool): Jika True, tambahkan indikator FinTA

    Returns:
        pd.DataFrame: Data harga + indikator
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    df = df.astype(int)
    df['ticker'] = [ticker for i in range(df.shape[0])]
    
    # Rename agar cocok dengan FinTA
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Adj Close": "adj_close"
    })
    df = df.reset_index()

    # Drop adj_close jika tidak diperlukan
    df = df.drop(columns=["adj_close"], errors='ignore')
    
    # Flatten columns: join the tuples into single strings, ignore empty second level
    df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]

    # Set 'Date' as the index
    #df = df.set_index('Date')
    
    # Tambahkan indikator teknikal
    if with_indicators:
        df = add_indicators(df)

    return df

def fetch_multiple_yf(
    tickers: list,
    start_date: str = "2015-01-01",
    end_date: str = None,
    with_indicators: bool = False
) -> pd.DataFrame:
    """
    Ambil data beberapa ticker dari yfinance, gabungkan menjadi satu dataframe.
    """
    all_data = pd.DataFrame()

    for ticker in tickers:
        try:
            df = fetch_data_yf(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                with_indicators=with_indicators
            )
            all_data = pd.concat([all_data,df])
        except Exception as e:
            print(f"[!] Gagal fetch {ticker}: {e}")

    if all_data.empty:
        raise ValueError("Tidak ada data berhasil diambil.")

    return all_data.sort_values(['ticker','Date'])

