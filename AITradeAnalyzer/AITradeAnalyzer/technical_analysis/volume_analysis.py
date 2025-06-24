import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class VolumeAnalyzer:
    def __init__(self):
        self.volume_sma_period = 20
        self.volume_threshold_multiplier = 1.5  # For unusual volume detection
        self.vwap_period = 20
        
    def analyze(self, data):
        """Main volume analysis function"""
        # Calculate volume indicators
        volume_indicators = self._calculate_volume_indicators(data)
        
        # Detect volume patterns
        volume_patterns = self._detect_volume_patterns(data, volume_indicators)
        
        # Analyze volume-price relationship
        vp_analysis = self._analyze_volume_price_relationship(data, volume_indicators)
        
        # Create features for ML model
        features = self._create_volume_features(data, volume_indicators, volume_patterns, vp_analysis)
        
        # Calculate volume score
        scores = self._calculate_volume_scores(features)
        
        return {
            'features': features,
            'scores': scores,
            'volume_indicators': volume_indicators,
            'volume_patterns': volume_patterns,
            'vp_analysis': vp_analysis
        }
    
    def _calculate_volume_indicators(self, data):
        """Calculate various volume-based indicators"""
        indicators = pd.DataFrame(index=data.index)
        
        if 'Volume' not in data.columns or data['Volume'].sum() == 0:
            # Create synthetic volume features for forex pairs
            indicators['volume'] = self._estimate_volume_from_price(data)
        else:
            indicators['volume'] = data['Volume']
        
        # Volume Moving Average
        indicators['volume_sma'] = indicators['volume'].rolling(self.volume_sma_period).mean()
        
        # Volume Ratio (current volume vs average)
        indicators['volume_ratio'] = indicators['volume'] / (indicators['volume_sma'] + 1e-10)
        
        # Volume Weighted Average Price (VWAP)
        indicators['vwap'] = self._calculate_vwap(data, indicators['volume'])
        
        # On Balance Volume (OBV)
        indicators['obv'] = self._calculate_obv(data, indicators['volume'])
        
        # Volume Price Trend (VPT)
        indicators['vpt'] = self._calculate_vpt(data, indicators['volume'])
        
        # Money Flow Index (MFI)
        indicators['mfi'] = self._calculate_mfi(data, indicators['volume'])
        
        # Accumulation/Distribution Line
        indicators['ad_line'] = self._calculate_ad_line(data, indicators['volume'])
        
        # Volume Rate of Change
        indicators['volume_roc'] = indicators['volume'].pct_change(periods=5) * 100
        
        # Relative Volume (compared to same time period)
        indicators['relative_volume'] = self._calculate_relative_volume(indicators['volume'])
        
        return indicators
    
    def _estimate_volume_from_price(self, data):
        """Estimate volume proxy from price movement for forex pairs"""
        # Use price volatility and range as volume proxy
        price_range = (data['High'] - data['Low']) / data['Close']
        price_change = abs(data['Close'].pct_change())
        
        # Combine range and momentum as volume proxy
        estimated_volume = (price_range + price_change) * 1000000
        return estimated_volume.fillna(1000000)
    
    def _calculate_vwap(self, data, volume):
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        cumulative_tp_volume = (typical_price * volume).rolling(self.vwap_period).sum()
        cumulative_volume = volume.rolling(self.vwap_period).sum()
        
        vwap = cumulative_tp_volume / cumulative_volume
        return vwap.fillna(data['Close'])
    
    def _calculate_obv(self, data, volume):
        """Calculate On Balance Volume"""
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_vpt(self, data, volume):
        """Calculate Volume Price Trend"""
        price_change_ratio = data['Close'].pct_change()
        vpt = (volume * price_change_ratio).cumsum()
        return vpt.fillna(0)
    
    def _calculate_mfi(self, data, volume, period=14):
        """Calculate Money Flow Index"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * volume
        
        positive_mf = pd.Series(index=data.index, dtype=float)
        negative_mf = pd.Series(index=data.index, dtype=float)
        
        for i in range(1, len(data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_mf.iloc[i] = money_flow.iloc[i]
                negative_mf.iloc[i] = 0
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_mf.iloc[i] = 0
                negative_mf.iloc[i] = money_flow.iloc[i]
            else:
                positive_mf.iloc[i] = 0
                negative_mf.iloc[i] = 0
        
        positive_mf = positive_mf.fillna(0)
        negative_mf = negative_mf.fillna(0)
        
        positive_mf_sum = positive_mf.rolling(period).sum()
        negative_mf_sum = negative_mf.rolling(period).sum()
        
        money_ratio = positive_mf_sum / (negative_mf_sum + 1e-10)  # Avoid division by zero
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi.fillna(50)
    
    def _calculate_ad_line(self, data, volume):
        """Calculate Accumulation/Distribution Line"""
        money_flow_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'] + 1e-10)
        money_flow_volume = money_flow_multiplier * volume
        ad_line = money_flow_volume.cumsum()
        return ad_line.fillna(0)
    
    def _calculate_relative_volume(self, volume):
        """Calculate relative volume compared to historical average"""
        # Use 50-period rolling average as baseline
        baseline_volume = volume.rolling(50).mean()
        relative_volume = volume / (baseline_volume + 1e-10)
        return relative_volume.fillna(1)
    
    def _detect_volume_patterns(self, data, volume_indicators):
        """Detect various volume patterns"""
        patterns = pd.DataFrame(index=data.index)
        
        # High Volume Pattern
        patterns['high_volume'] = (volume_indicators['volume_ratio'] > self.volume_threshold_multiplier).astype(int)
        
        # Volume Spike Detection
        volume_std = volume_indicators['volume'].rolling(20).std()
        volume_mean = volume_indicators['volume'].rolling(20).mean()
        patterns['volume_spike'] = (volume_indicators['volume'] > volume_mean + 2 * volume_std).astype(int)
        
        # Volume Breakout (high volume with price movement)
        price_change = abs(data['Close'].pct_change())
        patterns['volume_breakout'] = ((patterns['high_volume'] == 1) & (price_change > data['Close'].pct_change().rolling(20).std())).astype(int)
        
        # Volume Confirmation (volume increasing with trend)
        price_trend = data['Close'] > data['Close'].shift(5)
        volume_trend = volume_indicators['volume'] > volume_indicators['volume'].shift(5)
        patterns['volume_confirmation'] = (price_trend == volume_trend).astype(int)
        
        # Climax Volume (extreme volume at potential reversal points)
        volume_percentile = volume_indicators['volume'].rolling(50).apply(lambda x: (x.iloc[-1] >= np.percentile(x, 95)).astype(int))
        patterns['climax_volume'] = volume_percentile.fillna(0).astype(int)
        
        # Accumulation Pattern (increasing OBV with sideways price)
        obv_trend = volume_indicators['obv'] > volume_indicators['obv'].shift(10)
        price_sideways = abs(data['Close'].pct_change(10)) < 0.02  # Less than 2% move
        patterns['accumulation'] = (obv_trend & price_sideways).astype(int)
        
        # Distribution Pattern (decreasing OBV with sideways price)
        obv_decline = volume_indicators['obv'] < volume_indicators['obv'].shift(10)
        patterns['distribution'] = (obv_decline & price_sideways).astype(int)
        
        return patterns
    
    def _analyze_volume_price_relationship(self, data, volume_indicators):
        """Analyze the relationship between volume and price movements"""
        vp_analysis = pd.DataFrame(index=data.index)
        
        # Price above/below VWAP
        vp_analysis['price_above_vwap'] = (data['Close'] > volume_indicators['vwap']).astype(int)
        vp_analysis['vwap_distance'] = (data['Close'] - volume_indicators['vwap']) / volume_indicators['vwap']
        
        # Volume confirmation of price moves
        price_up = data['Close'] > data['Close'].shift(1)
        volume_up = volume_indicators['volume'] > volume_indicators['volume'].shift(1)
        vp_analysis['bullish_volume_confirmation'] = (price_up & volume_up).astype(int)
        
        price_down = data['Close'] < data['Close'].shift(1)
        volume_up_on_down = volume_indicators['volume'] > volume_indicators['volume'].shift(1)
        vp_analysis['bearish_volume_confirmation'] = (price_down & volume_up_on_down).astype(int)
        
        # Volume divergence
        price_momentum = data['Close'].pct_change(5)
        obv_momentum = volume_indicators['obv'].pct_change(5)
        vp_analysis['volume_divergence'] = abs(np.sign(price_momentum) - np.sign(obv_momentum))
        
        # Money flow direction
        vp_analysis['money_flow_bullish'] = (volume_indicators['mfi'] > 50).astype(int)
        vp_analysis['money_flow_oversold'] = (volume_indicators['mfi'] < 20).astype(int)
        vp_analysis['money_flow_overbought'] = (volume_indicators['mfi'] > 80).astype(int)
        
        return vp_analysis
    
    def _create_volume_features(self, data, volume_indicators, volume_patterns, vp_analysis):
        """Create volume features for ML model"""
        features = pd.DataFrame(index=data.index)
        
        # Basic volume features
        features['volume_ratio'] = volume_indicators['volume_ratio']
        features['relative_volume'] = volume_indicators['relative_volume']
        features['volume_roc'] = volume_indicators['volume_roc']
        
        # VWAP features
        features['price_vwap_ratio'] = data['Close'] / volume_indicators['vwap']
        features['vwap_distance'] = vp_analysis['vwap_distance']
        
        # OBV features
        features['obv_momentum'] = volume_indicators['obv'].pct_change(5)
        features['obv_trend'] = (volume_indicators['obv'] > volume_indicators['obv'].shift(10)).astype(int)
        
        # MFI features
        features['mfi'] = volume_indicators['mfi']
        features['mfi_oversold'] = vp_analysis['money_flow_oversold']
        features['mfi_overbought'] = vp_analysis['money_flow_overbought']
        
        # Pattern features
        features['high_volume'] = volume_patterns['high_volume']
        features['volume_spike'] = volume_patterns['volume_spike']
        features['volume_breakout'] = volume_patterns['volume_breakout']
        features['volume_confirmation'] = volume_patterns['volume_confirmation']
        features['accumulation'] = volume_patterns['accumulation']
        features['distribution'] = volume_patterns['distribution']
        
        # Relationship features
        features['bullish_volume_conf'] = vp_analysis['bullish_volume_confirmation']
        features['bearish_volume_conf'] = vp_analysis['bearish_volume_confirmation']
        features['volume_divergence'] = vp_analysis['volume_divergence']
        
        # Volume strength indicators
        features['volume_strength'] = self._calculate_volume_strength(volume_indicators, volume_patterns)
        
        return features.fillna(0)
    
    def _calculate_volume_strength(self, volume_indicators, volume_patterns):
        """Calculate overall volume strength score"""
        strength_components = [
            volume_indicators['volume_ratio'].clip(0, 3) / 3 * 0.3,  # Normalize to 0-1
            volume_patterns['volume_spike'] * 0.2,
            volume_patterns['volume_breakout'] * 0.2,
            volume_patterns['volume_confirmation'] * 0.15,
            volume_patterns['accumulation'] * 0.15
        ]
        
        volume_strength = sum(strength_components)
        return volume_strength.clip(0, 1)
    
    def _calculate_volume_scores(self, features):
        """Calculate composite volume scores for trading signals"""
        scores = pd.DataFrame(index=features.index)
        
        # Bullish volume score
        bullish_components = [
            features['high_volume'] * 0.2,
            features['volume_breakout'] * 0.25,
            features['bullish_volume_conf'] * 0.2,
            features['accumulation'] * 0.15,
            (features['mfi'] > 50).astype(int) * 0.1,
            (features['price_vwap_ratio'] > 1).astype(int) * 0.1
        ]
        scores['volume_bullish_score'] = sum(bullish_components)
        
        # Bearish volume score
        bearish_components = [
            features['distribution'] * 0.25,
            features['bearish_volume_conf'] * 0.2,
            features['volume_divergence'] * 0.15,
            features['mfi_overbought'] * 0.15,
            (features['price_vwap_ratio'] < 1).astype(int) * 0.15,
            features['volume_spike'] * 0.1  # Can indicate selling climax
        ]
        scores['volume_bearish_score'] = sum(bearish_components)
        
        # Overall volume score (bullish - bearish)
        scores['volume_score'] = scores['volume_bullish_score'] - scores['volume_bearish_score']
        
        # Normalize to 0-1 range
        scores['volume_score'] = (scores['volume_score'] + 1) / 2
        scores['volume_score'] = scores['volume_score'].clip(0, 1)
        
        # Volume confidence (based on volume strength)
        scores['volume_confidence'] = features['volume_strength'] * 100
        
        return scores