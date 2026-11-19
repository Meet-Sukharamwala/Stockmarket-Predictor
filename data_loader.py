# src/data_loader.py
import yfinance as yf
import pandas as pd
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

class StockDataLoader:
    def __init__(self, symbol: str, start: str, end: str, interval: str = "1d", cache_dir: str = "data"):
        self.symbol = symbol.upper()
        self.start = start
        self.end = end
        self.interval = interval
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._data = None

    def _get_cache_path(self):
        filename = f"{self.symbol}_{self.start}_{self.end}_{self.interval}.csv"
        return os.path.join(self.cache_dir, filename)

    def download(self) -> pd.DataFrame:
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            logger.info(f"Loading cached data for {self.symbol}")
            self._data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return self._data

        logger.info(f"Downloading data for {self.symbol} from {self.start} to {self.end}")
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(start=self.start, end=self.end, interval=self.interval)
            if df.empty:
                raise ValueError(f"No data returned for {self.symbol}")
            # Flatten multi-level columns if any
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df.to_csv(cache_path)
            self._data = df
            logger.info(f"Downloaded {len(df)} rows for {self.symbol}")
            return df
        except Exception as e:
            logger.error(f"Failed to download {self.symbol}: {e}")
            raise

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.download()
        return self._data.copy()
