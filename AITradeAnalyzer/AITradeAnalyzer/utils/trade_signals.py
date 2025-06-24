import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class TradeSignalGenerator:
    def __init__(self):
        self.min_strength_threshold = 30  # Minimum signal strength to generate trade
        self.confluence_weight = 0.3  # Weight for confluence of signals
        
    def generate_signals(self, data, ict_smc_signals, ema_signals, ml_predictions, 
                        risk_reward_ratio=2.0, max_risk_percent=2.0, volume_signals=None, 
                        news_features=None, volume_weight=0.3):
        """
        Generate trade signals by combining all analysis components
        
        Args:
            data: OHLCV price data
            ict_smc_signals: ICT/SMC analysis results
            ema_signals: EMA analysis results
            ml_predictions: ML model predictions
            risk_reward_ratio: Target risk/reward ratio
            max_risk_percent: Maximum risk percentage per trade
        
        Returns:
            DataFrame with trade signals and levels
        """
        signals = pd.DataFrame(index=data.index)
        
        # Extract scores from analysis components
        ict_scores = ict_smc_signals.get('scores', pd.DataFrame(index=data.index))
        ema_scores = ema_signals.get('scores', pd.DataFrame(index=data.index))
        volume_scores = volume_signals.get('scores', pd.DataFrame(index=data.index)) if volume_signals else pd.DataFrame(index=data.index)
        
        # Combine all signals
        signals = self._combine_signals(data, ict_scores, ema_scores, ml_predictions, volume_scores, news_features, volume_weight)
        
        # Calculate entry, stop loss, and take profit levels
        signals = self._calculate_trade_levels(signals, data, risk_reward_ratio, max_risk_percent)
        
        # Filter signals based on strength and confluence
        signals = self._filter_signals(signals)
        
        return signals
    
    def _combine_signals(self, data, ict_scores, ema_scores, ml_predictions, volume_scores=None, news_features=None, volume_weight=0.3):
        """Combine all signal components into unified signals"""
        signals = pd.DataFrame(index=data.index)
        
        # Initialize scores
        signals['ict_score'] = ict_scores.get('ict_score', 0) if 'ict_score' in ict_scores.columns else 0
        signals['smc_score'] = ict_scores.get('smc_score', 0) if 'smc_score' in ict_scores.columns else 0
        signals['ema_score'] = ema_scores.get('ema_score', 0) if 'ema_score' in ema_scores.columns else 0
        
        # Volume scores
        if volume_scores is not None and not volume_scores.empty:
            signals['volume_score'] = volume_scores.get('volume_score', 0) if 'volume_score' in volume_scores.columns else 0
            signals['volume_confidence'] = volume_scores.get('volume_confidence', 0) if 'volume_confidence' in volume_scores.columns else 0
        else:
            signals['volume_score'] = 0
            signals['volume_confidence'] = 0
        
        # News sentiment features
        if news_features:
            signals['news_sentiment'] = news_features.get('news_sentiment_score', 0)
            signals['news_confidence'] = news_features.get('news_confidence', 0)
            signals['news_bullish_ratio'] = news_features.get('news_bullish_ratio', 0)
        else:
            signals['news_sentiment'] = 0
            signals['news_confidence'] = 0
            signals['news_bullish_ratio'] = 0
        
        # ML prediction scores
        if isinstance(ml_predictions, pd.DataFrame):
            signals['ml_prediction'] = ml_predictions.get('prediction', 0)
            signals['ml_confidence'] = ml_predictions.get('confidence', 0)
        else:
            signals['ml_prediction'] = 0
            signals['ml_confidence'] = 0
        
        # Calculate composite scores
        signals['bullish_score'] = self._calculate_bullish_score(signals)
        signals['bearish_score'] = self._calculate_bearish_score(signals)
        
        # Generate primary signal
        signals['signal'] = self._generate_primary_signal(signals)
        
        # Calculate overall signal strength
        signals['strength'] = self._calculate_signal_strength(signals)
        
        return signals
    
    def _calculate_bullish_score(self, signals):
        """Calculate bullish signal score"""
        components = []
        
        # ICT/SMC bullish signals
        if 'ict_score' in signals.columns:
            components.append(signals['ict_score'] * 0.2)
        if 'smc_score' in signals.columns:
            components.append(signals['smc_score'] * 0.2)
        
        # EMA bullish signals
        if 'ema_score' in signals.columns:
            components.append(signals['ema_score'] * 0.2)
        
        # Volume bullish signals
        if 'volume_score' in signals.columns:
            components.append(signals['volume_score'] * 0.15)
        
        # News sentiment bullish signals
        if 'news_sentiment' in signals.columns:
            news_bullish = ((signals['news_sentiment'] > 0).astype(int) * 
                           abs(signals['news_sentiment']))
            components.append(news_bullish * 0.1)
        
        # ML bullish prediction
        if 'ml_prediction' in signals.columns and 'ml_confidence' in signals.columns:
            ml_bullish = ((signals['ml_prediction'] == 1).astype(int) * 
                         signals['ml_confidence'] / 100)
            components.append(ml_bullish * 0.15)
        
        return sum(components) if components else pd.Series(0, index=signals.index)
    
    def _calculate_bearish_score(self, signals):
        """Calculate bearish signal score"""
        components = []
        
        # ICT/SMC bearish signals (inverse of scores for bearish)
        if 'ict_score' in signals.columns:
            components.append((1 - signals['ict_score']) * 0.2)
        if 'smc_score' in signals.columns:
            components.append((1 - signals['smc_score']) * 0.2)
        
        # EMA bearish signals
        if 'ema_score' in signals.columns:
            components.append((1 - signals['ema_score']) * 0.2)
        
        # Volume bearish signals
        if 'volume_score' in signals.columns:
            components.append((1 - signals['volume_score']) * 0.15)
        
        # News sentiment bearish signals
        if 'news_sentiment' in signals.columns:
            news_bearish = ((signals['news_sentiment'] < 0).astype(int) * 
                           abs(signals['news_sentiment']))
            components.append(news_bearish * 0.1)
        
        # ML bearish prediction
        if 'ml_prediction' in signals.columns and 'ml_confidence' in signals.columns:
            ml_bearish = ((signals['ml_prediction'] == -1).astype(int) * 
                         signals['ml_confidence'] / 100)
            components.append(ml_bearish * 0.15)
        
        return sum(components) if components else pd.Series(0, index=signals.index)
    
    def _generate_primary_signal(self, signals):
        """Generate primary trading signal"""
        signal = pd.Series(0, index=signals.index)
        
        # Bullish signal conditions
        bullish_condition = (
            (signals['bullish_score'] > 0.6) & 
            (signals['bullish_score'] > signals['bearish_score'] + 0.2)
        )
        
        # Bearish signal conditions
        bearish_condition = (
            (signals['bearish_score'] > 0.6) & 
            (signals['bearish_score'] > signals['bullish_score'] + 0.2)
        )
        
        signal[bullish_condition] = 1   # Buy signal
        signal[bearish_condition] = -1  # Sell signal
        
        return signal
    
    def _calculate_signal_strength(self, signals):
        """Calculate overall signal strength (0-100%)"""
        # Base strength from score difference
        score_diff = abs(signals['bullish_score'] - signals['bearish_score'])
        base_strength = score_diff * 50  # Convert to 0-50 scale
        
        # Confluence bonus (multiple signals agreeing)
        confluence_bonus = 0
        
        # Check agreement between different methods
        agreements = 0
        total_methods = 0
        
        # ICT vs EMA agreement
        if 'ict_score' in signals.columns and 'ema_score' in signals.columns:
            ict_bullish = signals['ict_score'] > 0.5
            ema_bullish = signals['ema_score'] > 0.5
            agreements += (ict_bullish == ema_bullish).astype(int)
            total_methods += 1
        
        # ML vs Technical agreement
        if 'ml_prediction' in signals.columns:
            ml_bullish = signals['ml_prediction'] == 1
            tech_bullish = signals['bullish_score'] > signals['bearish_score']
            agreements += (ml_bullish == tech_bullish).astype(int)
            total_methods += 1
        
        if total_methods > 0:
            confluence_ratio = agreements / total_methods
            confluence_bonus = confluence_ratio * 30  # Up to 30% bonus
        
        # ML confidence bonus
        ml_bonus = 0
        if 'ml_confidence' in signals.columns:
            ml_bonus = signals['ml_confidence'] * 0.2  # Up to 20% bonus
        
        # Total strength
        total_strength = base_strength + confluence_bonus + ml_bonus
        
        return total_strength.clip(0, 100)
    
    def _calculate_trade_levels(self, signals, data, risk_reward_ratio, max_risk_percent):
        """Calculate entry, stop loss, and take profit levels"""
        # Initialize levels
        signals['entry_price'] = data['Close']
        signals['sl'] = 0.0
        signals['tp'] = 0.0
        signals['rr_ratio'] = 0.0
        
        # Calculate ATR for dynamic levels
        atr = data.get('ATR', data['High'] - data['Low'])
        atr = atr.fillna(data['Close'] * 0.01)  # 1% fallback if ATR not available
        
        for i in range(len(signals)):
            if signals['signal'].iloc[i] != 0:
                entry = signals['entry_price'].iloc[i]
                current_atr = atr.iloc[i]
                
                if signals['signal'].iloc[i] == 1:  # Buy signal
                    # Stop loss below recent low or based on ATR
                    recent_low = data['Low'].iloc[max(0, i-5):i+1].min()
                    atr_sl = entry - (current_atr * 1.5)
                    stop_loss = min(recent_low, atr_sl)
                    
                    # Ensure minimum risk
                    min_sl = entry * (1 - max_risk_percent / 100)
                    stop_loss = max(stop_loss, min_sl)
                    
                    # Take profit based on risk/reward ratio
                    risk = entry - stop_loss
                    take_profit = entry + (risk * risk_reward_ratio)
                    
                elif signals['signal'].iloc[i] == -1:  # Sell signal
                    # Stop loss above recent high or based on ATR
                    recent_high = data['High'].iloc[max(0, i-5):i+1].max()
                    atr_sl = entry + (current_atr * 1.5)
                    stop_loss = max(recent_high, atr_sl)
                    
                    # Ensure minimum risk
                    max_sl = entry * (1 + max_risk_percent / 100)
                    stop_loss = min(stop_loss, max_sl)
                    
                    # Take profit based on risk/reward ratio
                    risk = stop_loss - entry
                    take_profit = entry - (risk * risk_reward_ratio)
                
                # Update signals
                signals.iloc[i, signals.columns.get_loc('sl')] = stop_loss
                signals.iloc[i, signals.columns.get_loc('tp')] = take_profit
                signals.iloc[i, signals.columns.get_loc('rr_ratio')] = risk_reward_ratio
        
        return signals
    
    def _filter_signals(self, signals):
        """Filter signals based on quality criteria"""
        # Only keep signals above minimum strength threshold
        quality_filter = (
            (signals['strength'] >= self.min_strength_threshold) &
            (signals['signal'] != 0)
        )
        
        # Set weak signals to neutral
        signals.loc[~quality_filter, 'signal'] = 0
        
        # Remove consecutive duplicate signals (keep only first occurrence)
        signals['signal_change'] = signals['signal'] != signals['signal'].shift(1)
        
        # Only keep signals where there's a change or it's the first signal
        first_signal_mask = signals.index == signals.index[0]
        keep_signal = signals['signal_change'] | first_signal_mask
        
        # Set filtered out signals to 0
        signals.loc[~keep_signal, 'signal'] = 0
        
        # Clean up temporary column
        signals = signals.drop('signal_change', axis=1)
        
        return signals
    
    def calculate_position_size(self, account_balance, risk_percent, entry_price, stop_loss):
        """Calculate position size based on risk management"""
        risk_amount = account_balance * (risk_percent / 100)
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk > 0:
            position_size = risk_amount / price_risk
            return position_size
        else:
            return 0
    
    def backtest_signals(self, signals, data, initial_balance=10000):
        """Simple backtesting of generated signals"""
        backtest_results = []
        balance = initial_balance
        open_trades = []
        
        for i, (timestamp, signal_row) in enumerate(signals.iterrows()):
            current_price = data.loc[timestamp, 'Close']
            
            # Check for trade entries
            if signal_row['signal'] != 0 and signal_row['strength'] >= self.min_strength_threshold:
                trade = {
                    'entry_time': timestamp,
                    'entry_price': signal_row['entry_price'],
                    'signal': signal_row['signal'],
                    'sl': signal_row['sl'],
                    'tp': signal_row['tp'],
                    'strength': signal_row['strength']
                }
                open_trades.append(trade)
            
            # Check for trade exits
            for trade in open_trades[:]:
                if trade['signal'] == 1:  # Buy trade
                    if current_price <= trade['sl']:  # Stop loss hit
                        pnl = trade['sl'] - trade['entry_price']
                        result = 'loss'
                    elif current_price >= trade['tp']:  # Take profit hit
                        pnl = trade['tp'] - trade['entry_price']
                        result = 'win'
                    else:
                        continue
                
                elif trade['signal'] == -1:  # Sell trade
                    if current_price >= trade['sl']:  # Stop loss hit
                        pnl = trade['entry_price'] - trade['sl']
                        result = 'loss'
                    elif current_price <= trade['tp']:  # Take profit hit
                        pnl = trade['entry_price'] - trade['tp']
                        result = 'win'
                    else:
                        continue
                
                # Record trade result
                pnl_percent = (pnl / trade['entry_price']) * 100
                balance += balance * (pnl_percent / 100)
                
                backtest_results.append({
                    'entry_time': trade['entry_time'],
                    'exit_time': timestamp,
                    'signal': trade['signal'],
                    'entry_price': trade['entry_price'],
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'result': result,
                    'strength': trade['strength'],
                    'balance': balance
                })
                
                open_trades.remove(trade)
        
        return pd.DataFrame(backtest_results)
