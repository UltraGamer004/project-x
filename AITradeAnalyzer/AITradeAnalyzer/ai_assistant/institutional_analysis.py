import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class InstitutionalAnalyzer:
    def __init__(self):
        self.large_trade_threshold = 0.01  # 1% of average volume
        self.whale_trade_threshold = 0.05  # 5% of average volume
        self.session_times = {
            'asian': {'start': 0, 'end': 8},
            'london': {'start': 8, 'end': 16},
            'new_york': {'start': 13, 'end': 21},
            'sydney': {'start': 21, 'end': 5}
        }
    
    def analyze(self, data):
        """Comprehensive institutional analysis"""
        analysis = {
            'whale_orders': self._detect_whale_orders(data),
            'large_trades': self._detect_large_trades(data),
            'institutional_levels': self._find_institutional_levels(data),
            'session_analysis': self._analyze_trading_sessions(data),
            'risk_index': self._calculate_risk_index(data),
            'fear_greed_index': self._calculate_fear_greed_index(data),
            'smart_money_flow': self._analyze_smart_money_flow(data),
            'features': self._create_features(data)
        }
        
        return analysis
    
    def _detect_whale_orders(self, data):
        """Detect whale orders based on volume and price impact"""
        whale_orders = pd.DataFrame(index=data.index)
        whale_orders['is_whale'] = False
        whale_orders['volume_score'] = 0.0
        whale_orders['price_impact'] = 0.0
        
        if 'Volume' not in data.columns:
            return whale_orders
        
        # Calculate volume metrics
        avg_volume = data['Volume'].rolling(50).mean()
        volume_std = data['Volume'].rolling(50).std()
        
        # Price impact calculation
        price_change = data['Close'].pct_change()
        volume_ratio = data['Volume'] / avg_volume
        
        # Whale detection criteria
        whale_volume_threshold = avg_volume + (volume_std * 3)
        high_volume = data['Volume'] > whale_volume_threshold
        significant_price_impact = abs(price_change) > 0.005  # 0.5% price move
        
        whale_orders['is_whale'] = high_volume & significant_price_impact
        whale_orders['volume_score'] = np.where(
            whale_orders['is_whale'],
            volume_ratio.clip(0, 10),
            0
        )
        whale_orders['price_impact'] = abs(price_change) * 100
        
        return whale_orders
    
    def _detect_large_trades(self, data):
        """Detect large institutional trades"""
        large_trades = pd.DataFrame(index=data.index)
        large_trades['is_large'] = False
        large_trades['trade_size'] = 0.0
        large_trades['direction'] = 0  # 1 for buying pressure, -1 for selling
        
        if 'Volume' not in data.columns:
            return large_trades
        
        # Calculate trade size relative to average
        avg_volume = data['Volume'].rolling(20).mean()
        trade_size_ratio = data['Volume'] / avg_volume
        
        # Large trade threshold
        large_threshold = 2.0  # 2x average volume
        large_trades['is_large'] = trade_size_ratio > large_threshold
        large_trades['trade_size'] = trade_size_ratio
        
        # Determine direction based on price action
        price_change = data['Close'] - data['Open']
        close_position = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        
        # Buying pressure if close near high with volume
        # Selling pressure if close near low with volume
        large_trades['direction'] = np.where(
            large_trades['is_large'],
            np.where(close_position > 0.7, 1, np.where(close_position < 0.3, -1, 0)),
            0
        )
        
        return large_trades
    
    def _find_institutional_levels(self, data):
        """Find key institutional support/resistance levels"""
        levels = pd.DataFrame(index=data.index)
        levels['institutional_support'] = np.nan
        levels['institutional_resistance'] = np.nan
        levels['level_strength'] = 0.0
        
        # High volume nodes (institutional interest levels)
        if 'Volume' in data.columns:
            # Price-volume analysis
            price_bins = pd.cut(data['Close'], bins=50)
            volume_by_price = data.groupby(price_bins)['Volume'].sum()
            
            # Find high volume price levels
            high_volume_threshold = volume_by_price.quantile(0.8)
            significant_levels = volume_by_price[volume_by_price > high_volume_threshold]
            
            for i, (price_range, volume) in enumerate(significant_levels.items()):
                level_price = (price_range.left + price_range.right) / 2
                current_price = data['Close'].iloc[-1]
                
                # Classify as support or resistance
                if level_price < current_price:
                    levels.loc[levels.index[-1], 'institutional_support'] = level_price
                else:
                    levels.loc[levels.index[-1], 'institutional_resistance'] = level_price
                
                # Calculate level strength
                strength = min(volume / volume_by_price.max(), 1.0)
                levels.loc[levels.index[-1], 'level_strength'] = strength
        
        return levels
    
    def _analyze_trading_sessions(self, data):
        """Analyze trading activity by session"""
        session_analysis = pd.DataFrame(index=data.index)
        session_analysis['active_session'] = 'off_hours'
        session_analysis['session_volume'] = 0.0
        session_analysis['session_volatility'] = 0.0
        
        # Determine current session based on UTC time
        for idx in data.index:
            hour_utc = idx.hour
            
            if self.session_times['asian']['start'] <= hour_utc < self.session_times['asian']['end']:
                session = 'asian'
            elif self.session_times['london']['start'] <= hour_utc < self.session_times['london']['end']:
                session = 'london'
            elif self.session_times['new_york']['start'] <= hour_utc < self.session_times['new_york']['end']:
                session = 'new_york'
            elif hour_utc >= self.session_times['sydney']['start'] or hour_utc < self.session_times['sydney']['end']:
                session = 'sydney'
            else:
                session = 'off_hours'
            
            session_analysis.loc[idx, 'active_session'] = session
        
        # Calculate session metrics
        if 'Volume' in data.columns:
            session_analysis['session_volume'] = data['Volume'] / data['Volume'].rolling(24).mean()
        
        # Session volatility
        session_analysis['session_volatility'] = (data['High'] - data['Low']) / data['Close'] * 100
        
        return session_analysis
    
    def _calculate_risk_index(self, data):
        """Calculate market risk index"""
        risk_index = pd.DataFrame(index=data.index)
        risk_index['risk_score'] = 50.0  # Base risk
        risk_index['volatility_risk'] = 0.0
        risk_index['volume_risk'] = 0.0
        risk_index['trend_risk'] = 0.0
        
        # Volatility risk
        price_volatility = data['Close'].pct_change().rolling(20).std() * 100
        volatility_percentile = price_volatility.rolling(100).rank(pct=True) * 100
        risk_index['volatility_risk'] = volatility_percentile.fillna(50)
        
        # Volume risk (unusual volume can indicate risk)
        if 'Volume' in data.columns:
            volume_ratio = data['Volume'] / data['Volume'].rolling(20).mean()
            volume_risk = np.where(volume_ratio > 2, 80, np.where(volume_ratio < 0.5, 30, 50))
            risk_index['volume_risk'] = volume_risk
        
        # Trend risk (trending markets are generally less risky for scalping)
        ema_short = data['Close'].ewm(span=10).mean()
        ema_long = data['Close'].ewm(span=30).mean()
        trend_strength = abs(ema_short - ema_long) / ema_long * 100
        trend_risk = np.where(trend_strength > 1, 30, np.where(trend_strength < 0.2, 80, 50))
        risk_index['trend_risk'] = trend_risk
        
        # Composite risk score
        risk_index['risk_score'] = (
            risk_index['volatility_risk'] * 0.4 +
            risk_index['volume_risk'] * 0.3 +
            risk_index['trend_risk'] * 0.3
        ).clip(0, 100)
        
        return risk_index
    
    def _calculate_fear_greed_index(self, data):
        """Calculate crypto fear and greed index"""
        fear_greed = pd.DataFrame(index=data.index)
        fear_greed['fear_greed_score'] = 50.0  # Neutral
        fear_greed['market_momentum'] = 0.0
        fear_greed['volatility_component'] = 0.0
        fear_greed['volume_component'] = 0.0
        
        # Market momentum component (25%)
        returns = data['Close'].pct_change()
        momentum_5d = returns.rolling(5).sum() * 100
        momentum_score = np.where(momentum_5d > 5, 80, np.where(momentum_5d < -5, 20, 50))
        fear_greed['market_momentum'] = momentum_score
        
        # Volatility component (25%) - high volatility = fear
        volatility = returns.rolling(14).std() * 100
        volatility_ma = volatility.rolling(30).mean()
        volatility_ratio = volatility / volatility_ma
        volatility_score = np.where(volatility_ratio > 1.5, 20, np.where(volatility_ratio < 0.7, 80, 50))
        fear_greed['volatility_component'] = volatility_score
        
        # Volume component (25%) - high volume in uptrend = greed
        if 'Volume' in data.columns:
            volume_ma = data['Volume'].rolling(20).mean()
            volume_ratio = data['Volume'] / volume_ma
            price_up = returns > 0
            
            volume_score = np.where(
                (volume_ratio > 1.5) & price_up, 80,
                np.where((volume_ratio > 1.5) & ~price_up, 20, 50)
            )
            fear_greed['volume_component'] = volume_score
        else:
            fear_greed['volume_component'] = 50
        
        # Combine components
        fear_greed['fear_greed_score'] = (
            fear_greed['market_momentum'] * 0.4 +
            fear_greed['volatility_component'] * 0.3 +
            fear_greed['volume_component'] * 0.3
        ).clip(0, 100)
        
        return fear_greed
    
    def _analyze_smart_money_flow(self, data):
        """Analyze smart money flow patterns"""
        smart_money = pd.DataFrame(index=data.index)
        smart_money['money_flow'] = 0.0
        smart_money['accumulation'] = 0.0
        smart_money['distribution'] = 0.0
        
        # Money Flow Index-based analysis
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        
        if 'Volume' in data.columns:
            raw_money_flow = typical_price * data['Volume']
            
            # Positive and negative money flow
            positive_flow = pd.Series(0.0, index=data.index)
            negative_flow = pd.Series(0.0, index=data.index)
            
            for i in range(1, len(data)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow.iloc[i] = raw_money_flow.iloc[i]
                elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                    negative_flow.iloc[i] = raw_money_flow.iloc[i]
            
            # Calculate money flow ratio
            positive_mf_sum = positive_flow.rolling(14).sum()
            negative_mf_sum = negative_flow.rolling(14).sum()
            
            money_flow_ratio = positive_mf_sum / (negative_mf_sum + 1e-10)
            money_flow_index = 100 - (100 / (1 + money_flow_ratio))
            
            smart_money['money_flow'] = money_flow_index.fillna(50)
            
            # Accumulation/Distribution detection
            # Accumulation: Price stable/up with increasing volume
            price_change = data['Close'].pct_change(5)
            volume_change = data['Volume'].pct_change(5)
            
            accumulation_signal = (
                (price_change >= -0.01) &  # Price stable or slightly up
                (volume_change > 0.2) &    # Volume increasing
                (money_flow_index > 60)    # Money flowing in
            )
            
            distribution_signal = (
                (price_change <= 0.01) &   # Price stable or slightly down
                (volume_change > 0.2) &    # Volume increasing
                (money_flow_index < 40)    # Money flowing out
            )
            
            smart_money['accumulation'] = accumulation_signal.astype(int)
            smart_money['distribution'] = distribution_signal.astype(int)
        
        return smart_money
    
    def _create_features(self, data):
        """Create institutional analysis features for ML"""
        features = pd.DataFrame(index=data.index)
        
        # Get all analysis components
        whale_orders = self._detect_whale_orders(data)
        large_trades = self._detect_large_trades(data)
        session_analysis = self._analyze_trading_sessions(data)
        risk_index = self._calculate_risk_index(data)
        fear_greed = self._calculate_fear_greed_index(data)
        smart_money = self._analyze_smart_money_flow(data)
        
        # Whale and large trade features
        features['whale_detected'] = whale_orders['is_whale'].astype(int)
        features['whale_volume_score'] = whale_orders['volume_score']
        features['large_trade_detected'] = large_trades['is_large'].astype(int)
        features['trade_direction'] = large_trades['direction']
        
        # Session features
        session_dummies = pd.get_dummies(session_analysis['active_session'], prefix='session')
        for col in session_dummies.columns:
            features[col] = session_dummies[col]
        
        features['session_volume'] = session_analysis['session_volume']
        features['session_volatility'] = session_analysis['session_volatility']
        
        # Risk features
        features['risk_score'] = risk_index['risk_score']
        features['volatility_risk'] = risk_index['volatility_risk']
        features['volume_risk'] = risk_index['volume_risk']
        features['trend_risk'] = risk_index['trend_risk']
        
        # Fear & Greed features
        features['fear_greed_score'] = fear_greed['fear_greed_score']
        features['market_momentum'] = fear_greed['market_momentum']
        
        # Smart money features
        features['money_flow_index'] = smart_money['money_flow']
        features['accumulation'] = smart_money['accumulation']
        features['distribution'] = smart_money['distribution']
        
        # Derived features
        features['is_high_risk'] = (features['risk_score'] > 70).astype(int)
        features['is_extreme_fear'] = (features['fear_greed_score'] < 25).astype(int)
        features['is_extreme_greed'] = (features['fear_greed_score'] > 75).astype(int)
        features['institutional_activity'] = (
            features['whale_detected'] + features['large_trade_detected']
        ).clip(0, 1)
        
        return features.fillna(0)
    
    def get_current_summary(self, data):
        """Get current institutional analysis summary"""
        if data is None or data.empty:
            return {}
        
        analysis = self.analyze(data)
        latest_idx = data.index[-1]
        
        summary = {
            'timestamp': latest_idx,
            'whale_activity': bool(analysis['whale_orders']['is_whale'].iloc[-1]),
            'large_trades': bool(analysis['large_trades']['is_large'].iloc[-1]),
            'current_session': analysis['session_analysis']['active_session'].iloc[-1],
            'risk_level': analysis['risk_index']['risk_score'].iloc[-1],
            'fear_greed_level': analysis['fear_greed_index']['fear_greed_score'].iloc[-1],
            'smart_money_flow': analysis['smart_money_flow']['money_flow'].iloc[-1],
            'accumulation_detected': bool(analysis['smart_money_flow']['accumulation'].iloc[-1]),
            'distribution_detected': bool(analysis['smart_money_flow']['distribution'].iloc[-1])
        }
        
        # Add interpretations
        if summary['risk_level'] > 70:
            summary['risk_interpretation'] = 'High Risk'
        elif summary['risk_level'] < 30:
            summary['risk_interpretation'] = 'Low Risk'
        else:
            summary['risk_interpretation'] = 'Medium Risk'
        
        if summary['fear_greed_level'] > 75:
            summary['sentiment_interpretation'] = 'Extreme Greed'
        elif summary['fear_greed_level'] > 55:
            summary['sentiment_interpretation'] = 'Greed'
        elif summary['fear_greed_level'] < 25:
            summary['sentiment_interpretation'] = 'Extreme Fear'
        elif summary['fear_greed_level'] < 45:
            summary['sentiment_interpretation'] = 'Fear'
        else:
            summary['sentiment_interpretation'] = 'Neutral'
        
        return summary