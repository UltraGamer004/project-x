import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class ICTSMCAnalyzer:
    def __init__(self):
        self.swing_length = 5  # Periods to look for swing points
        self.liquidity_threshold = 0.001  # 0.1% for liquidity zones
        self.fvg_threshold = 0.0005  # 0.05% for fair value gaps
        
    def analyze(self, data):
        """Main analysis function that returns ICT/SMC signals and features"""
        # Calculate all ICT/SMC components
        swing_points = self._find_swing_points(data)
        market_structure = self._analyze_market_structure(data, swing_points)
        liquidity_zones = self._find_liquidity_zones(data, swing_points)
        fair_value_gaps = self._find_fair_value_gaps(data)
        order_blocks = self._find_order_blocks(data, swing_points)
        bos_choch = self._detect_bos_choch(data, swing_points)
        
        # Create features DataFrame
        features = self._create_features(
            data, swing_points, market_structure, liquidity_zones, 
            fair_value_gaps, order_blocks, bos_choch
        )
        
        # Calculate composite scores
        scores = self._calculate_scores(features)
        
        return {
            'features': features,
            'scores': scores,
            'swing_points': swing_points,
            'market_structure': market_structure,
            'liquidity_zones': liquidity_zones,
            'fair_value_gaps': fair_value_gaps,
            'order_blocks': order_blocks,
            'bos_choch': bos_choch
        }
    
    def _find_swing_points(self, data):
        """Find swing highs and lows"""
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find peaks and troughs
        high_peaks, _ = find_peaks(highs, distance=self.swing_length)
        low_peaks, _ = find_peaks(-lows, distance=self.swing_length)
        
        swing_points = pd.DataFrame(index=data.index)
        swing_points['swing_high'] = False
        swing_points['swing_low'] = False
        swing_points['swing_high_price'] = np.nan
        swing_points['swing_low_price'] = np.nan
        
        if len(high_peaks) > 0:
            swing_points.iloc[high_peaks, swing_points.columns.get_loc('swing_high')] = True
            swing_points.iloc[high_peaks, swing_points.columns.get_loc('swing_high_price')] = highs[high_peaks]
        
        if len(low_peaks) > 0:
            swing_points.iloc[low_peaks, swing_points.columns.get_loc('swing_low')] = True
            swing_points.iloc[low_peaks, swing_points.columns.get_loc('swing_low_price')] = lows[low_peaks]
        
        return swing_points
    
    def _analyze_market_structure(self, data, swing_points):
        """Analyze market structure (Higher Highs, Lower Lows, etc.)"""
        structure = pd.DataFrame(index=data.index)
        structure['trend'] = 0  # 1 = uptrend, -1 = downtrend, 0 = sideways
        structure['structure_break'] = False
        
        # Get swing highs and lows
        swing_highs = swing_points[swing_points['swing_high']]['swing_high_price'].dropna()
        swing_lows = swing_points[swing_points['swing_low']]['swing_low_price'].dropna()
        
        # Analyze trend based on swing points
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            latest_high = swing_highs.iloc[-1]
            prev_high = swing_highs.iloc[-2] if len(swing_highs) >= 2 else latest_high
            
            latest_low = swing_lows.iloc[-1]
            prev_low = swing_lows.iloc[-2] if len(swing_lows) >= 2 else latest_low
            
            # Determine trend
            if latest_high > prev_high and latest_low > prev_low:
                structure.iloc[-1, structure.columns.get_loc('trend')] = 1  # Uptrend
            elif latest_high < prev_high and latest_low < prev_low:
                structure.iloc[-1, structure.columns.get_loc('trend')] = -1  # Downtrend
            else:
                structure.iloc[-1, structure.columns.get_loc('trend')] = 0  # Sideways
        
        # Forward fill trend
        structure['trend'] = structure['trend'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        return structure
    
    def _find_liquidity_zones(self, data, swing_points):
        """Find liquidity zones around swing points"""
        liquidity = pd.DataFrame(index=data.index)
        liquidity['buy_liquidity'] = 0.0
        liquidity['sell_liquidity'] = 0.0
        liquidity['liquidity_zone'] = False
        
        # Find liquidity around swing lows (buy liquidity)
        swing_lows = swing_points[swing_points['swing_low']]['swing_low_price'].dropna()
        for idx, low_price in swing_lows.items():
            # Create liquidity zone around the swing low
            zone_start = max(0, data.index.get_loc(idx) - 5)
            zone_end = min(len(data), data.index.get_loc(idx) + 5)
            
            liquidity.iloc[zone_start:zone_end, liquidity.columns.get_loc('buy_liquidity')] = 1.0
            liquidity.iloc[zone_start:zone_end, liquidity.columns.get_loc('liquidity_zone')] = True
        
        # Find liquidity around swing highs (sell liquidity)
        swing_highs = swing_points[swing_points['swing_high']]['swing_high_price'].dropna()
        for idx, high_price in swing_highs.items():
            # Create liquidity zone around the swing high
            zone_start = max(0, data.index.get_loc(idx) - 5)
            zone_end = min(len(data), data.index.get_loc(idx) + 5)
            
            liquidity.iloc[zone_start:zone_end, liquidity.columns.get_loc('sell_liquidity')] = 1.0
            liquidity.iloc[zone_start:zone_end, liquidity.columns.get_loc('liquidity_zone')] = True
        
        return liquidity
    
    def _find_fair_value_gaps(self, data):
        """Find Fair Value Gaps (FVG)"""
        fvg = pd.DataFrame(index=data.index)
        fvg['bullish_fvg'] = False
        fvg['bearish_fvg'] = False
        fvg['fvg_strength'] = 0.0
        
        for i in range(2, len(data)):
            # Bullish FVG: High of candle i-2 < Low of candle i
            if data['High'].iloc[i-2] < data['Low'].iloc[i]:
                gap_size = (data['Low'].iloc[i] - data['High'].iloc[i-2]) / data['Close'].iloc[i-1]
                if gap_size > self.fvg_threshold:
                    fvg.iloc[i, fvg.columns.get_loc('bullish_fvg')] = True
                    fvg.iloc[i, fvg.columns.get_loc('fvg_strength')] = gap_size
            
            # Bearish FVG: Low of candle i-2 > High of candle i
            if data['Low'].iloc[i-2] > data['High'].iloc[i]:
                gap_size = (data['Low'].iloc[i-2] - data['High'].iloc[i]) / data['Close'].iloc[i-1]
                if gap_size > self.fvg_threshold:
                    fvg.iloc[i, fvg.columns.get_loc('bearish_fvg')] = True
                    fvg.iloc[i, fvg.columns.get_loc('fvg_strength')] = gap_size
        
        return fvg
    
    def _find_order_blocks(self, data, swing_points):
        """Find Order Blocks"""
        order_blocks = pd.DataFrame(index=data.index)
        order_blocks['bullish_ob'] = False
        order_blocks['bearish_ob'] = False
        order_blocks['ob_strength'] = 0.0
        
        # Find order blocks around swing points
        swing_highs = swing_points[swing_points['swing_high']].index
        swing_lows = swing_points[swing_points['swing_low']].index
        
        # Bearish Order Blocks (around swing highs)
        for swing_idx in swing_highs:
            try:
                swing_pos = data.index.get_loc(swing_idx)
                if swing_pos >= 3:
                    # Look for the last green candle before the swing high
                    for i in range(swing_pos - 3, swing_pos):
                        if data['Close'].iloc[i] > data['Open'].iloc[i]:  # Green candle
                            volume_strength = data['Volume'].iloc[i] / data['Volume'].iloc[max(0, i-10):i+1].mean() if 'Volume' in data.columns else 1.0
                            order_blocks.iloc[i, order_blocks.columns.get_loc('bearish_ob')] = True
                            order_blocks.iloc[i, order_blocks.columns.get_loc('ob_strength')] = min(volume_strength, 5.0)
                            break
            except:
                continue
        
        # Bullish Order Blocks (around swing lows)
        for swing_idx in swing_lows:
            try:
                swing_pos = data.index.get_loc(swing_idx)
                if swing_pos >= 3:
                    # Look for the last red candle before the swing low
                    for i in range(swing_pos - 3, swing_pos):
                        if data['Close'].iloc[i] < data['Open'].iloc[i]:  # Red candle
                            volume_strength = data['Volume'].iloc[i] / data['Volume'].iloc[max(0, i-10):i+1].mean() if 'Volume' in data.columns else 1.0
                            order_blocks.iloc[i, order_blocks.columns.get_loc('bullish_ob')] = True
                            order_blocks.iloc[i, order_blocks.columns.get_loc('ob_strength')] = min(volume_strength, 5.0)
                            break
            except:
                continue
        
        return order_blocks
    
    def _detect_bos_choch(self, data, swing_points):
        """Detect Break of Structure (BOS) and Change of Character (CHoCH)"""
        bos_choch = pd.DataFrame(index=data.index)
        bos_choch['bos'] = False
        bos_choch['choch'] = False
        bos_choch['signal_strength'] = 0.0
        
        # Get swing points
        swing_highs = swing_points[swing_points['swing_high']]['swing_high_price'].dropna()
        swing_lows = swing_points[swing_points['swing_low']]['swing_low_price'].dropna()
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Check for BOS (continuation of trend)
            latest_high_idx = swing_highs.index[-1]
            latest_high = swing_highs.iloc[-1]
            prev_high = swing_highs.iloc[-2] if len(swing_highs) >= 2 else latest_high
            
            latest_low_idx = swing_lows.index[-1]
            latest_low = swing_lows.iloc[-1]
            prev_low = swing_lows.iloc[-2] if len(swing_lows) >= 2 else latest_low
            
            # BOS Bullish: Breaking previous high in uptrend
            if latest_high > prev_high and latest_low > prev_low:
                try:
                    idx_pos = data.index.get_loc(latest_high_idx)
                    bos_choch.iloc[idx_pos, bos_choch.columns.get_loc('bos')] = True
                    bos_choch.iloc[idx_pos, bos_choch.columns.get_loc('signal_strength')] = (latest_high - prev_high) / prev_high
                except:
                    pass
            
            # BOS Bearish: Breaking previous low in downtrend
            elif latest_high < prev_high and latest_low < prev_low:
                try:
                    idx_pos = data.index.get_loc(latest_low_idx)
                    bos_choch.iloc[idx_pos, bos_choch.columns.get_loc('bos')] = True
                    bos_choch.iloc[idx_pos, bos_choch.columns.get_loc('signal_strength')] = (prev_low - latest_low) / prev_low
                except:
                    pass
            
            # CHoCH: Change of character (trend reversal)
            else:
                try:
                    recent_idx = max(latest_high_idx, latest_low_idx)
                    idx_pos = data.index.get_loc(recent_idx)
                    bos_choch.iloc[idx_pos, bos_choch.columns.get_loc('choch')] = True
                    bos_choch.iloc[idx_pos, bos_choch.columns.get_loc('signal_strength')] = 0.5
                except:
                    pass
        
        return bos_choch
    
    def _create_features(self, data, swing_points, market_structure, liquidity_zones, fair_value_gaps, order_blocks, bos_choch):
        """Create feature matrix for ML model"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['price_momentum'] = data['Close'].pct_change(5)
        features['volatility'] = data['High'].rolling(20).std() / data['Close'].rolling(20).mean()
        
        # Swing point features
        features['swing_high_signal'] = swing_points['swing_high'].astype(int)
        features['swing_low_signal'] = swing_points['swing_low'].astype(int)
        
        # Market structure features
        features['trend'] = market_structure['trend']
        features['structure_break'] = market_structure['structure_break'].astype(int)
        
        # Liquidity features
        features['buy_liquidity'] = liquidity_zones['buy_liquidity']
        features['sell_liquidity'] = liquidity_zones['sell_liquidity']
        features['liquidity_zone'] = liquidity_zones['liquidity_zone'].astype(int)
        
        # Fair Value Gap features
        features['bullish_fvg'] = fair_value_gaps['bullish_fvg'].astype(int)
        features['bearish_fvg'] = fair_value_gaps['bearish_fvg'].astype(int)
        features['fvg_strength'] = fair_value_gaps['fvg_strength']
        
        # Order Block features
        features['bullish_ob'] = order_blocks['bullish_ob'].astype(int)
        features['bearish_ob'] = order_blocks['bearish_ob'].astype(int)
        features['ob_strength'] = order_blocks['ob_strength']
        
        # BOS/CHoCH features
        features['bos'] = bos_choch['bos'].astype(int)
        features['choch'] = bos_choch['choch'].astype(int)
        features['signal_strength'] = bos_choch['signal_strength']
        
        return features.fillna(0)
    
    def _calculate_scores(self, features):
        """Calculate composite ICT/SMC scores"""
        scores = pd.DataFrame(index=features.index)
        
        # ICT Score (based on FVG, Order Blocks, Liquidity)
        ict_components = [
            features['fvg_strength'] * 0.3,
            features['ob_strength'] * 0.3,
            features['liquidity_zone'] * 0.2,
            features['signal_strength'] * 0.2
        ]
        scores['ict_score'] = sum(ict_components)
        
        # SMC Score (based on BOS, CHoCH, Market Structure)
        smc_components = [
            features['bos'] * 0.4,
            features['choch'] * 0.3,
            (features['trend'].abs()) * 0.3
        ]
        scores['smc_score'] = sum(smc_components)
        
        # Normalize scores to 0-1 range
        scores['ict_score'] = scores['ict_score'].clip(0, 1)
        scores['smc_score'] = scores['smc_score'].clip(0, 1)
        
        return scores
