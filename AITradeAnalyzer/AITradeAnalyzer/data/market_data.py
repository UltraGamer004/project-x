import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MarketDataFetcher:
    def __init__(self):
        self.supported_timeframes = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1wk': '1wk',
            '1mo': '1mo'
        }
    
    def fetch_data(self, symbol, period_days=90, timeframe='1h'):
        """
        Fetch market data from Yahoo Finance
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD=X', 'AAPL')
            period_days: Number of days to fetch
            timeframe: Timeframe for the data
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Calculate period for yfinance
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # Adjust period for yfinance API limitations
            if timeframe in ['1m', '5m']:
                # For minute data, limit to last 7 days
                period_days = min(period_days, 7)
                start_date = end_date - timedelta(days=period_days)
            elif timeframe in ['15m', '30m', '1h']:
                # For hourly data, limit to last 60 days
                period_days = min(period_days, 60)
                start_date = end_date - timedelta(days=period_days)
            
            # Map timeframe
            yf_interval = self.supported_timeframes.get(timeframe, '1h')
            
            # Fetch data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
                auto_adjust=True,
                prepost=False
            )
            
            if data.empty:
                print(f"No data received for {symbol}")
                return None
            
            # Clean and prepare data
            data = self._clean_data(data)
            
            # Add additional calculations
            data = self._add_technical_indicators(data)
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _clean_data(self, data):
        """Clean and prepare the raw data"""
        # Remove any NaN values
        data = data.dropna()
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                if col == 'Volume':
                    data[col] = 1000000  # Default volume for forex pairs
                else:
                    print(f"Missing required column: {col}")
                    return pd.DataFrame()
        
        # Remove any rows with zero or negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            data = data[data[col] > 0]
        
        # Ensure High >= Low
        data = data[data['High'] >= data['Low']]
        
        # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        data = data[data['High'] >= data[['Open', 'Close']].max(axis=1)]
        data = data[data['Low'] <= data[['Open', 'Close']].min(axis=1)]
        
        return data
    
    def _add_technical_indicators(self, data):
        """Add basic technical indicators"""
        # Average True Range (ATR)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data['ATR'] = true_range.rolling(14).mean()
        
        # Simple moving averages
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        
        # Bollinger Bands
        sma_20 = data['Close'].rolling(20).mean()
        std_20 = data['Close'].rolling(20).std()
        data['BB_Upper'] = sma_20 + (std_20 * 2)
        data['BB_Lower'] = sma_20 - (std_20 * 2)
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        data['MACD'] = ema_12 - ema_26
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Volume indicators (if volume data is available)
        if 'Volume' in data.columns and data['Volume'].sum() > 0:
            # Volume Moving Average
            data['Volume_MA'] = data['Volume'].rolling(20).mean()
            
            # On Balance Volume
            data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()
        
        return data
    
    def get_symbol_info(self, symbol):
        """Get information about a trading symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
        except:
            return {'symbol': symbol, 'name': symbol}
    
    def validate_symbol(self, symbol):
        """Validate if a symbol exists and has data"""
        try:
            ticker = yf.Ticker(symbol)
            # Try to fetch 5 days of data
            test_data = ticker.history(period='5d', interval='1d')
            return not test_data.empty
        except:
            return False
    
    def get_latest_price(self, symbol):
        """Get the latest price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                return data['Close'].iloc[-1]
            return None
        except:
            return None
