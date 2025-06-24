import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingViewDataFetcher:
    def __init__(self):
        # TradingView symbol mapping to Yahoo Finance
        self.symbol_mapping = {
            'BTCUSDT.PS': 'BTC-USD',
            'ETHUSDT.PS': 'ETH-USD',
            'ADAUSDT.PS': 'ADA-USD',
            'DOTUSDT.PS': 'DOT-USD',
            'LINKUSDT.PS': 'LINK-USD',
            'BNBUSDT.PS': 'BNB-USD',
            'XRPUSDT.PS': 'XRP-USD',
            'SOLUSDT.PS': 'SOL-USD',
            'MATICUSDT.PS': 'MATIC-USD',
            'AVAXUSDT.PS': 'AVAX-USD'
        }
        
        self.supported_timeframes = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
            '1wk': '1wk'
        }
    
    def fetch_crypto_data(self, tv_symbol, period_days=90, timeframe='1h'):
        """
        Fetch cryptocurrency data using TradingView symbol notation
        
        Args:
            tv_symbol: TradingView symbol (e.g., 'BTCUSDT.PS')
            period_days: Number of days to fetch
            timeframe: Timeframe for the data
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Map TradingView symbol to Yahoo Finance symbol
            yf_symbol = self.symbol_mapping.get(tv_symbol, tv_symbol.replace('.PS', '-USD'))
            
            # Create ticker object
            ticker = yf.Ticker(yf_symbol)
            
            # Calculate period for yfinance
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # Adjust period for API limitations
            if timeframe in ['1m', '5m']:
                period_days = min(period_days, 7)
                start_date = end_date - timedelta(days=period_days)
            elif timeframe in ['15m', '30m', '1h']:
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
                print(f"No data received for {tv_symbol} ({yf_symbol})")
                return None
            
            # Clean and prepare data
            data = self._clean_crypto_data(data)
            
            # Add crypto-specific indicators
            data = self._add_crypto_indicators(data)
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {tv_symbol}: {e}")
            return None
    
    def _clean_crypto_data(self, data):
        """Clean and prepare cryptocurrency data"""
        # Remove any NaN values
        data = data.dropna()
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in data.columns:
                print(f"Missing required column: {col}")
                return pd.DataFrame()
        
        # Remove any rows with zero or negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            data = data[data[col] > 0]
        
        # Ensure High >= Low and price consistency
        data = data[data['High'] >= data['Low']]
        data = data[data['High'] >= data[['Open', 'Close']].max(axis=1)]
        data = data[data['Low'] <= data[['Open', 'Close']].min(axis=1)]
        
        # Remove rows with zero volume (if any)
        data = data[data['Volume'] > 0]
        
        return data
    
    def _add_crypto_indicators(self, data):
        """Add cryptocurrency-specific technical indicators"""
        # Standard technical indicators
        data = self._add_standard_indicators(data)
        
        # Crypto-specific indicators
        data = self._add_crypto_volatility_indicators(data)
        data = self._add_momentum_indicators(data)
        data = self._add_volume_indicators(data)
        
        return data
    
    def _add_standard_indicators(self, data):
        """Add standard technical indicators"""
        # Simple Moving Averages
        for period in [10, 20, 50, 100, 200]:
            data[f'SMA_{period}'] = data['Close'].rolling(period).mean()
        
        # Exponential Moving Averages
        for period in [8, 13, 21, 55, 89, 144, 233]:
            data[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()
        
        # Bollinger Bands
        sma_20 = data['Close'].rolling(20).mean()
        std_20 = data['Close'].rolling(20).std()
        data['BB_Upper'] = sma_20 + (std_20 * 2)
        data['BB_Lower'] = sma_20 - (std_20 * 2)
        data['BB_Middle'] = sma_20
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
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
        
        return data
    
    def _add_crypto_volatility_indicators(self, data):
        """Add cryptocurrency volatility indicators"""
        # Average True Range
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data['ATR'] = true_range.rolling(14).mean()
        data['ATR_Percent'] = (data['ATR'] / data['Close']) * 100
        
        # Volatility indicators
        data['Price_Volatility'] = data['Close'].pct_change().rolling(20).std() * 100
        data['High_Low_Ratio'] = (data['High'] - data['Low']) / data['Close'] * 100
        
        # Crypto Fear & Greed proxy (based on price volatility and volume)
        data['Volatility_MA'] = data['Price_Volatility'].rolling(50).mean()
        data['Volume_MA'] = data['Volume'].rolling(50).mean()
        data['Fear_Greed_Proxy'] = (
            (1 - (data['Price_Volatility'] / (data['Volatility_MA'] + 1))) * 50 +
            (data['Volume'] / (data['Volume_MA'] + 1) - 1) * 25 + 50
        ).clip(0, 100)
        
        return data
    
    def _add_momentum_indicators(self, data):
        """Add momentum indicators for crypto trading"""
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(14).min()
        high_14 = data['High'].rolling(14).max()
        data['Stoch_K'] = 100 * (data['Close'] - low_14) / (high_14 - low_14)
        data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()
        
        # Williams %R
        data['Williams_R'] = -100 * (high_14 - data['Close']) / (high_14 - low_14)
        
        # Rate of Change
        for period in [5, 10, 20]:
            data[f'ROC_{period}'] = data['Close'].pct_change(period) * 100
        
        # Commodity Channel Index (CCI)
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        data['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return data
    
    def _add_volume_indicators(self, data):
        """Add volume-based indicators"""
        # Volume Moving Averages
        data['Volume_SMA_20'] = data['Volume'].rolling(20).mean()
        data['Volume_SMA_50'] = data['Volume'].rolling(50).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
        
        # On Balance Volume
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = data['Volume'].iloc[0]
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        data['OBV'] = obv
        data['OBV_MA'] = data['OBV'].rolling(20).mean()
        
        # Volume Weighted Average Price (VWAP)
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        cumulative_tp_volume = (typical_price * data['Volume']).rolling(20).sum()
        cumulative_volume = data['Volume'].rolling(20).sum()
        data['VWAP'] = cumulative_tp_volume / cumulative_volume
        
        # Money Flow Index
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_mf = pd.Series(index=data.index, dtype=float).fillna(0)
        negative_mf = pd.Series(index=data.index, dtype=float).fillna(0)
        
        for i in range(1, len(data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_mf.iloc[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_mf.iloc[i] = money_flow.iloc[i]
        
        positive_mf_sum = positive_mf.rolling(14).sum()
        negative_mf_sum = negative_mf.rolling(14).sum()
        
        money_ratio = positive_mf_sum / (negative_mf_sum + 1e-10)
        data['MFI'] = 100 - (100 / (1 + money_ratio))
        
        return data
    
    def get_crypto_info(self, tv_symbol):
        """Get information about a cryptocurrency"""
        yf_symbol = self.symbol_mapping.get(tv_symbol, tv_symbol.replace('.PS', '-USD'))
        
        try:
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            
            return {
                'symbol': tv_symbol,
                'yf_symbol': yf_symbol,
                'name': info.get('longName', tv_symbol),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Crypto'),
                'sector': 'Cryptocurrency'
            }
        except:
            return {
                'symbol': tv_symbol,
                'yf_symbol': yf_symbol,
                'name': tv_symbol,
                'market_cap': 'N/A',
                'currency': 'USD',
                'exchange': 'Crypto',
                'sector': 'Cryptocurrency'
            }
    
    def validate_crypto_symbol(self, tv_symbol):
        """Validate if a crypto symbol exists and has data"""
        yf_symbol = self.symbol_mapping.get(tv_symbol, tv_symbol.replace('.PS', '-USD'))
        
        try:
            ticker = yf.Ticker(yf_symbol)
            test_data = ticker.history(period='5d', interval='1d')
            return not test_data.empty
        except:
            return False
    
    def get_latest_crypto_price(self, tv_symbol):
        """Get the latest price for a cryptocurrency"""
        yf_symbol = self.symbol_mapping.get(tv_symbol, tv_symbol.replace('.PS', '-USD'))
        
        try:
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                return {
                    'price': data['Close'].iloc[-1],
                    'change': data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0,
                    'change_percent': ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0,
                    'volume': data['Volume'].iloc[-1],
                    'high_24h': data['High'].max(),
                    'low_24h': data['Low'].min()
                }
            return None
        except:
            return None