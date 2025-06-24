import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class EMAAnalyzer:
    def __init__(self):
        # EMA periods for different timeframes
        self.short_periods = [8, 13, 21]
        self.medium_periods = [50, 89, 144]
        self.long_periods = [200, 233, 377]
        
    def analyze(self, data):
        """Main EMA analysis function"""
        # Calculate all EMAs
        ema_data = self._calculate_emas(data)
        
        # Analyze EMA alignment and signals
        alignment = self._analyze_ema_alignment(ema_data)
        crossovers = self._detect_crossovers(ema_data)
        support_resistance = self._find_ema_support_resistance(data, ema_data)
        
        # Create features for ML model
        features = self._create_ema_features(data, ema_data, alignment, crossovers, support_resistance)
        
        # Calculate composite EMA score
        scores = self._calculate_ema_scores(features)
        
        return {
            'features': features,
            'scores': scores,
            'ema_data': ema_data,
            'alignment': alignment,
            'crossovers': crossovers,
            'support_resistance': support_resistance
        }
    
    def _calculate_emas(self, data):
        """Calculate all EMA lines"""
        ema_data = pd.DataFrame(index=data.index)
        
        # Calculate short-term EMAs
        for period in self.short_periods:
            ema_data[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
        
        # Calculate medium-term EMAs
        for period in self.medium_periods:
            ema_data[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
        
        # Calculate long-term EMAs
        for period in self.long_periods:
            ema_data[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
        
        return ema_data
    
    def _analyze_ema_alignment(self, ema_data):
        """Analyze EMA alignment for trend identification"""
        alignment = pd.DataFrame(index=ema_data.index)
        
        # Short-term alignment
        short_emas = [f'ema_{p}' for p in self.short_periods]
        alignment['short_bullish'] = self._check_bullish_alignment(ema_data[short_emas])
        alignment['short_bearish'] = self._check_bearish_alignment(ema_data[short_emas])
        
        # Medium-term alignment
        medium_emas = [f'ema_{p}' for p in self.medium_periods]
        alignment['medium_bullish'] = self._check_bullish_alignment(ema_data[medium_emas])
        alignment['medium_bearish'] = self._check_bearish_alignment(ema_data[medium_emas])
        
        # Long-term alignment
        long_emas = [f'ema_{p}' for p in self.long_periods]
        alignment['long_bullish'] = self._check_bullish_alignment(ema_data[long_emas])
        alignment['long_bearish'] = self._check_bearish_alignment(ema_data[long_emas])
        
        # Overall alignment score
        alignment['bullish_alignment'] = (
            alignment['short_bullish'] * 0.5 +
            alignment['medium_bullish'] * 0.3 +
            alignment['long_bullish'] * 0.2
        )
        
        alignment['bearish_alignment'] = (
            alignment['short_bearish'] * 0.5 +
            alignment['medium_bearish'] * 0.3 +
            alignment['long_bearish'] * 0.2
        )
        
        return alignment
    
    def _check_bullish_alignment(self, ema_subset):
        """Check if EMAs are in bullish alignment (faster > slower)"""
        bullish_signals = []
        
        for i in range(len(ema_subset.columns) - 1):
            faster_ema = ema_subset.iloc[:, i]
            slower_ema = ema_subset.iloc[:, i + 1]
            bullish_signals.append((faster_ema > slower_ema).astype(int))
        
        # Return percentage of bullish alignments
        return sum(bullish_signals) / len(bullish_signals) if bullish_signals else pd.Series(0, index=ema_subset.index)
    
    def _check_bearish_alignment(self, ema_subset):
        """Check if EMAs are in bearish alignment (faster < slower)"""
        bearish_signals = []
        
        for i in range(len(ema_subset.columns) - 1):
            faster_ema = ema_subset.iloc[:, i]
            slower_ema = ema_subset.iloc[:, i + 1]
            bearish_signals.append((faster_ema < slower_ema).astype(int))
        
        # Return percentage of bearish alignments
        return sum(bearish_signals) / len(bearish_signals) if bearish_signals else pd.Series(0, index=ema_subset.index)
    
    def _detect_crossovers(self, ema_data):
        """Detect EMA crossovers"""
        crossovers = pd.DataFrame(index=ema_data.index)
        
        # Key crossover pairs
        crossover_pairs = [
            ('ema_8', 'ema_21'),
            ('ema_13', 'ema_50'),
            ('ema_21', 'ema_89'),
            ('ema_50', 'ema_200')
        ]
        
        for fast_ema, slow_ema in crossover_pairs:
            if fast_ema in ema_data.columns and slow_ema in ema_data.columns:
                # Golden cross (bullish)
                golden_cross = ((ema_data[fast_ema] > ema_data[slow_ema]) & 
                               (ema_data[fast_ema].shift(1) <= ema_data[slow_ema].shift(1)))
                
                # Death cross (bearish)
                death_cross = ((ema_data[fast_ema] < ema_data[slow_ema]) & 
                              (ema_data[fast_ema].shift(1) >= ema_data[slow_ema].shift(1)))
                
                crossovers[f'{fast_ema}_{slow_ema}_golden'] = golden_cross.astype(int)
                crossovers[f'{fast_ema}_{slow_ema}_death'] = death_cross.astype(int)
        
        # Overall crossover signals
        golden_columns = [col for col in crossovers.columns if 'golden' in col]
        death_columns = [col for col in crossovers.columns if 'death' in col]
        
        crossovers['total_golden_crosses'] = crossovers[golden_columns].sum(axis=1)
        crossovers['total_death_crosses'] = crossovers[death_columns].sum(axis=1)
        
        return crossovers
    
    def _find_ema_support_resistance(self, data, ema_data):
        """Find EMA support and resistance levels"""
        support_resistance = pd.DataFrame(index=data.index)
        
        # Key EMAs for support/resistance
        key_emas = ['ema_21', 'ema_50', 'ema_89', 'ema_200']
        
        for ema_name in key_emas:
            if ema_name in ema_data.columns:
                ema_line = ema_data[ema_name]
                
                # Support: Price bounces up from EMA
                support_signals = []
                for i in range(2, len(data)):
                    if (data['Low'].iloc[i-1] <= ema_line.iloc[i-1] and
                        data['Close'].iloc[i] > ema_line.iloc[i] and
                        data['Close'].iloc[i-1] <= ema_line.iloc[i-1]):
                        support_signals.append(i)
                
                # Resistance: Price bounces down from EMA
                resistance_signals = []
                for i in range(2, len(data)):
                    if (data['High'].iloc[i-1] >= ema_line.iloc[i-1] and
                        data['Close'].iloc[i] < ema_line.iloc[i] and
                        data['Close'].iloc[i-1] >= ema_line.iloc[i-1]):
                        resistance_signals.append(i)
                
                # Create signals
                support_resistance[f'{ema_name}_support'] = 0
                support_resistance[f'{ema_name}_resistance'] = 0
                
                if support_signals:
                    support_resistance.iloc[support_signals, support_resistance.columns.get_loc(f'{ema_name}_support')] = 1
                
                if resistance_signals:
                    support_resistance.iloc[resistance_signals, support_resistance.columns.get_loc(f'{ema_name}_resistance')] = 1
        
        # Calculate distance from key EMAs
        for ema_name in key_emas:
            if ema_name in ema_data.columns:
                distance = (data['Close'] - ema_data[ema_name]) / ema_data[ema_name] * 100
                support_resistance[f'{ema_name}_distance'] = distance
        
        return support_resistance
    
    def _create_ema_features(self, data, ema_data, alignment, crossovers, support_resistance):
        """Create EMA features for ML model"""
        features = pd.DataFrame(index=data.index)
        
        # Price relative to EMAs
        for period in self.short_periods + self.medium_periods:
            ema_col = f'ema_{period}'
            if ema_col in ema_data.columns:
                features[f'price_above_{ema_col}'] = (data['Close'] > ema_data[ema_col]).astype(int)
                features[f'distance_to_{ema_col}'] = (data['Close'] - ema_data[ema_col]) / ema_data[ema_col]
        
        # EMA slopes (trend strength)
        for period in [8, 21, 50, 200]:
            ema_col = f'ema_{period}'
            if ema_col in ema_data.columns:
                features[f'{ema_col}_slope'] = ema_data[ema_col].pct_change(5)
        
        # Alignment features
        features['bullish_alignment'] = alignment['bullish_alignment']
        features['bearish_alignment'] = alignment['bearish_alignment']
        features['short_term_bullish'] = alignment['short_bullish']
        features['medium_term_bullish'] = alignment['medium_bullish']
        features['long_term_bullish'] = alignment['long_bullish']
        
        # Crossover features
        features['recent_golden_crosses'] = crossovers['total_golden_crosses'].rolling(5).sum()
        features['recent_death_crosses'] = crossovers['total_death_crosses'].rolling(5).sum()
        
        # Support/Resistance features
        support_cols = [col for col in support_resistance.columns if 'support' in col and 'distance' not in col]
        resistance_cols = [col for col in support_resistance.columns if 'resistance' in col and 'distance' not in col]
        
        features['ema_support_signals'] = support_resistance[support_cols].sum(axis=1)
        features['ema_resistance_signals'] = support_resistance[resistance_cols].sum(axis=1)
        
        # EMA confluence (multiple EMAs close together)
        ema_values = ema_data[[f'ema_{p}' for p in [21, 50, 89, 200] if f'ema_{p}' in ema_data.columns]]
        if not ema_values.empty:
            ema_std = ema_values.std(axis=1)
            ema_mean = ema_values.mean(axis=1)
            features['ema_confluence'] = (ema_std / ema_mean).fillna(0)
        else:
            features['ema_confluence'] = 0
        
        return features.fillna(0)
    
    def _calculate_ema_scores(self, features):
        """Calculate composite EMA scores"""
        scores = pd.DataFrame(index=features.index)
        
        # Bullish EMA score
        bullish_components = []
        if 'bullish_alignment' in features.columns:
            bullish_components.append(features['bullish_alignment'] * 0.3)
        if 'recent_golden_crosses' in features.columns:
            bullish_components.append((features['recent_golden_crosses'] > 0).astype(int) * 0.2)
        if 'ema_support_signals' in features.columns:
            bullish_components.append((features['ema_support_signals'] > 0).astype(int) * 0.2)
        if 'ema_8_slope' in features.columns:
            bullish_components.append((features['ema_8_slope'] > 0).astype(int) * 0.15)
        if 'ema_21_slope' in features.columns:
            bullish_components.append((features['ema_21_slope'] > 0).astype(int) * 0.15)
        
        # Bearish EMA score
        bearish_components = []
        if 'bearish_alignment' in features.columns:
            bearish_components.append(features['bearish_alignment'] * 0.3)
        if 'recent_death_crosses' in features.columns:
            bearish_components.append((features['recent_death_crosses'] > 0).astype(int) * 0.2)
        if 'ema_resistance_signals' in features.columns:
            bearish_components.append((features['ema_resistance_signals'] > 0).astype(int) * 0.2)
        if 'ema_8_slope' in features.columns:
            bearish_components.append((features['ema_8_slope'] < 0).astype(int) * 0.15)
        if 'ema_21_slope' in features.columns:
            bearish_components.append((features['ema_21_slope'] < 0).astype(int) * 0.15)
        
        # Calculate scores
        scores['ema_bullish_score'] = sum(bullish_components) if bullish_components else 0
        scores['ema_bearish_score'] = sum(bearish_components) if bearish_components else 0
        
        # Overall EMA score (bullish - bearish)
        scores['ema_score'] = scores['ema_bullish_score'] - scores['ema_bearish_score']
        
        # Normalize to 0-1 range
        scores['ema_score'] = (scores['ema_score'] + 1) / 2  # Convert from -1,1 to 0,1
        scores['ema_score'] = scores['ema_score'].clip(0, 1)
        
        return scores
