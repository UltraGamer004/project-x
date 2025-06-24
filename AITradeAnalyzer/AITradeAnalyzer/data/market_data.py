from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')


class MarketDataFetcher:
    def __init__(self, binance_api_key='', binance_api_secret=''):
        self.client = Client(binance_api_key, binance_api_secret)

        # Binance-supported intervals
        self.supported_timeframes = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
            '1w': Client.KLINE_INTERVAL_1WEEK,
            '1mo': Client.KLINE_INTERVAL_1MONTH
        }

    def fetch_data(self, symbol, period_days=90, timeframe='1h'):
        try:
            interval = self.supported_timeframes.get(timeframe, Client.KLINE_INTERVAL_1HOUR)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=period_days)

            # Convert to milliseconds
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)

            df = self._fetch_klines(symbol, interval, start_ts, end_ts)

            if df.empty:
                print(f"No data received for {symbol}")
                return None

            df = self._clean_data(df)
            df = self._add_technical_indicators(df)

            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def _fetch_klines(self, symbol, interval, start_ts, end_ts):
        all_data = []
        limit = 1000
        while True:
            candles = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ts,
                endTime=end_ts,
                limit=limit
            )
            if not candles:
                break

            all_data.extend(candles)
            last_open_time = candles[-1][0]
            start_ts = last_open_time + 1

            if len(candles) < limit:
                break

            time.sleep(0.1)

        df = pd.DataFrame(all_data, columns=[
            'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close Time', 'Quote Asset Volume', 'Number of Trades',
            'Taker Buy Base Volume', 'Taker Buy Quote Volume', 'Ignore'
        ])

        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df.set_index('Open Time', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

        return df

    def _clean_data(self, data):
        data = data.dropna()

        # Ensure all required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                return pd.DataFrame()

        # Filter invalid prices
        for col in ['Open', 'High', 'Low', 'Close']:
            data = data[data[col] > 0]
        data = data[data['High'] >= data['Low']]
        data = data[data['High'] >= data[['Open', 'Close']].max(axis=1)]
        data = data[data['Low'] <= data[['Open', 'Close']].min(axis=1)]

        return data

    def _add_technical_indicators(self, data):
        # ATR
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(14).mean()

        # SMAs
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()

        # Bollinger Bands
        sma_20 = data['Close'].rolling(20).mean()
        std_20 = data['Close'].rolling(20).std()
        data['BB_Upper'] = sma_20 + 2 * std_20
        data['BB_Lower'] = sma_20 - 2 * std_20

        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        data['MACD'] = ema_12 - ema_26
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

        # Volume indicators
        data['Volume_MA'] = data['Volume'].rolling(20).mean()
        data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

        return data

    def validate_symbol(self, symbol):
        try:
            self.client.get_symbol_info(symbol)
            return True
        except:
            return False

    def get_latest_price(self, symbol):
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price']) if 'price' in ticker else None
        except:
            return None
