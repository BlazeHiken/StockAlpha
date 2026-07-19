import yfinance as yf
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def fetch_stock_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical adjusted close prices for a list of tickers using yfinance.
    
    Args:
        tickers (list[str]): List of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: A DataFrame with a DateTime index and one column per ticker 
                      containing the adjusted close prices.
    """
    logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}.")
    
    df_list = []
    
    for ticker in tickers:
        try:
            logger.info(f"Fetching data for {ticker}...")
            # Fetch data (returns OHLCV)
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                logger.warning(f"No data found for {ticker}. Skipping.")
                continue
            
            # Handle potential MultiIndex columns in newer yfinance versions
            if isinstance(data.columns, pd.MultiIndex):
                if 'Adj Close' in data.columns.get_level_values(0):
                    series = data['Adj Close'].iloc[:, 0]
                elif 'Close' in data.columns.get_level_values(0):
                    series = data['Close'].iloc[:, 0]
                else:
                    logger.warning(f"Could not find 'Adj Close' or 'Close' for {ticker}. Skipping.")
                    continue
            else:
                if 'Adj Close' in data.columns:
                    series = data['Adj Close']
                elif 'Close' in data.columns:
                    series = data['Close']
                else:
                    logger.warning(f"Could not find 'Adj Close' or 'Close' for {ticker}. Skipping.")
                    continue
            
            # Ensure it's a Series and name it
            if isinstance(series, pd.DataFrame):
                series = series.squeeze()
            series.name = ticker
            
            # Handle missing gaps
            if series.isnull().any():
                logger.warning(f"{ticker} has missing values. Forward filling small gaps.")
                series = series.ffill()
            
            df_list.append(series)
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            
    if not df_list:
        logger.error("No data fetched for any tickers.")
        return pd.DataFrame()
        
    # Combine all series into a single DataFrame
    combined_df = pd.concat(df_list, axis=1)
    
    # Ensure index is sorted chronologically
    combined_df.sort_index(inplace=True)

    # Fill small internal gaps using the previous day's price,
    # then remove any remaining rows with missing values
    combined_df = combined_df.ffill().dropna()
    
    logger.info(f"Successfully combined data for {len(combined_df.columns)} tickers. Shape: {combined_df.shape}")
    return combined_df

def save_price_data(df: pd.DataFrame, filepath: str | Path) -> None:
    """
    Saves a pandas DataFrame to a local Parquet file.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        filepath (str): The destination Parquet file path.
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info(f"Data successfully saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save data to {filepath}: {e}")
        raise

def load_price_data(filepath: str | Path) -> pd.DataFrame:
    """
    Loads a pandas DataFrame from a Parquet file.
    
    Args:
        filepath (str): The path to the Parquet file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        path = Path(filepath)
        df = pd.read_parquet(path)
        logger.info(f"Data successfully loaded from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {e}")
        raise

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Example demonstration
    test_tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS", "INVALID_TICKER_TEST"]
    
    # Calculate last 3 years
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=3*365)
    
    start_str = start_dt.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')
    
    # 1. Fetch
    print("--- 1. Fetching Data ---")
    prices_df = fetch_stock_data(test_tickers, start_date=start_str, end_date=end_str)
    
    if not prices_df.empty:
        # 2. Save
        print("\n--- 2. Saving Data ---")
        test_filepath = "example_prices.parquet"
        save_price_data(prices_df, test_filepath)
        
        # 3. Load
        print("\n--- 3. Loading Data ---")
        loaded_df = load_price_data(test_filepath)
        
        # 4. Print results
        print("\n--- Final Loaded Data Info ---")
        print(f"Shape: {loaded_df.shape}")
        print("First 5 Rows:")
        print(loaded_df.head())
    else:
        print("\nNo data was returned. Check your internet connection or ticker validity.")
