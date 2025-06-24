import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ScalpingBacktester:
    def __init__(self):
        self.commission = 0.001  # 0.1% per trade
        self.slippage = 0.0005   # 0.05% slippage
        self.initial_balance = 10000
        
    def backtest_strategy(self, data, signals, timeframe='1m', lookback_days=7):
        """Comprehensive backtesting for scalping strategies"""
        
        # Filter data for backtesting period
        end_date = data.index[-1]
        start_date = end_date - timedelta(days=lookback_days)
        backtest_data = data[data.index >= start_date].copy()
        backtest_signals = signals[signals.index >= start_date].copy()
        
        # Initialize results
        results = {
            'trades': [],
            'equity_curve': [],
            'performance_metrics': {},
            'risk_metrics': {},
            'detailed_analysis': {}
        }
        
        # Run backtest simulation
        trades = self._simulate_trades(backtest_data, backtest_signals, timeframe)
        results['trades'] = trades
        
        # Calculate performance metrics
        results['performance_metrics'] = self._calculate_performance_metrics(trades)
        results['risk_metrics'] = self._calculate_risk_metrics(trades)
        results['equity_curve'] = self._build_equity_curve(trades)
        results['detailed_analysis'] = self._analyze_trade_patterns(trades, backtest_data)
        
        return results
    
    def _simulate_trades(self, data, signals, timeframe):
        """Simulate trade execution"""
        trades = []
        open_trade = None
        balance = self.initial_balance
        
        for i, (timestamp, signal_row) in enumerate(signals.iterrows()):
            if timestamp not in data.index:
                continue
                
            current_data = data.loc[timestamp]
            current_price = current_data['Close']
            
            # Close existing trade if needed
            if open_trade is not None:
                exit_reason, exit_price = self._check_exit_conditions(
                    open_trade, current_data, data.iloc[max(0, i-5):i+1]
                )
                
                if exit_reason:
                    # Close trade
                    trade_result = self._close_trade(open_trade, exit_price, exit_reason, timestamp)
                    trades.append(trade_result)
                    
                    # Update balance
                    balance += trade_result['pnl_absolute']
                    open_trade = None
            
            # Open new trade if signal present and no open trade
            if open_trade is None and signal_row.get('signal', 0) != 0:
                if signal_row.get('strength', 0) >= 30:  # Minimum signal strength
                    open_trade = self._open_trade(signal_row, current_price, timestamp, timeframe)
        
        # Close any remaining open trade
        if open_trade is not None:
            final_price = data['Close'].iloc[-1]
            trade_result = self._close_trade(open_trade, final_price, 'end_of_data', data.index[-1])
            trades.append(trade_result)
        
        return trades
    
    def _open_trade(self, signal_row, entry_price, timestamp, timeframe):
        """Open a new trade"""
        # Adjust entry price for slippage
        direction = signal_row['signal']
        adjusted_entry = entry_price * (1 + self.slippage * direction)
        
        # Calculate position size (risk-based)
        account_risk = 0.02  # 2% risk per trade
        stop_loss = signal_row.get('sl', 0)
        
        if stop_loss > 0:
            risk_per_unit = abs(adjusted_entry - stop_loss)
            position_size = (self.initial_balance * account_risk) / risk_per_unit
        else:
            # Default position size if no stop loss
            position_size = self.initial_balance * 0.1  # 10% of balance
        
        trade = {
            'entry_time': timestamp,
            'direction': direction,
            'entry_price': adjusted_entry,
            'position_size': position_size,
            'stop_loss': signal_row.get('sl', 0),
            'take_profit': signal_row.get('tp', 0),
            'signal_strength': signal_row.get('strength', 0),
            'timeframe': timeframe,
            'commission_paid': position_size * adjusted_entry * self.commission
        }
        
        return trade
    
    def _check_exit_conditions(self, trade, current_data, recent_data):
        """Check if trade should be exited"""
        current_price = current_data['Close']
        high = current_data['High']
        low = current_data['Low']
        
        # Stop loss hit
        if trade['stop_loss'] > 0:
            if trade['direction'] == 1 and low <= trade['stop_loss']:
                return 'stop_loss', trade['stop_loss']
            elif trade['direction'] == -1 and high >= trade['stop_loss']:
                return 'stop_loss', trade['stop_loss']
        
        # Take profit hit
        if trade['take_profit'] > 0:
            if trade['direction'] == 1 and high >= trade['take_profit']:
                return 'take_profit', trade['take_profit']
            elif trade['direction'] == -1 and low <= trade['take_profit']:
                return 'take_profit', trade['take_profit']
        
        # Time-based exit for scalping (max 1 hour for 1m, max 4 hours for 15m)
        time_diff = current_data.name - trade['entry_time']
        max_duration = timedelta(hours=1) if trade['timeframe'] == '1m' else timedelta(hours=4)
        
        if time_diff > max_duration:
            return 'time_exit', current_price
        
        # Trailing stop for profitable trades (basic implementation)
        entry_price = trade['entry_price']
        unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * trade['direction']
        
        if unrealized_pnl_pct > 0.005:  # 0.5% profit
            trailing_stop_pct = 0.002  # 0.2% trailing stop
            if trade['direction'] == 1:
                trail_stop = current_price * (1 - trailing_stop_pct)
                if low <= trail_stop:
                    return 'trailing_stop', trail_stop
            else:
                trail_stop = current_price * (1 + trailing_stop_pct)
                if high >= trail_stop:
                    return 'trailing_stop', trail_stop
        
        return None, None
    
    def _close_trade(self, trade, exit_price, exit_reason, exit_time):
        """Close trade and calculate results"""
        # Adjust exit price for slippage
        adjusted_exit = exit_price * (1 - self.slippage * trade['direction'])
        
        # Calculate PnL
        price_diff = (adjusted_exit - trade['entry_price']) * trade['direction']
        pnl_absolute = price_diff * trade['position_size']
        pnl_percentage = (price_diff / trade['entry_price']) * 100
        
        # Account for commission
        exit_commission = trade['position_size'] * adjusted_exit * self.commission
        total_commission = trade['commission_paid'] + exit_commission
        net_pnl = pnl_absolute - total_commission
        
        # Trade duration
        duration = exit_time - trade['entry_time']
        
        trade_result = {
            'entry_time': trade['entry_time'],
            'exit_time': exit_time,
            'duration': duration,
            'direction': 'Long' if trade['direction'] == 1 else 'Short',
            'entry_price': trade['entry_price'],
            'exit_price': adjusted_exit,
            'position_size': trade['position_size'],
            'pnl_absolute': net_pnl,
            'pnl_percentage': pnl_percentage,
            'exit_reason': exit_reason,
            'signal_strength': trade['signal_strength'],
            'commission': total_commission,
            'timeframe': trade['timeframe'],
            'win': net_pnl > 0
        }
        
        return trade_result
    
    def _calculate_performance_metrics(self, trades):
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {}
        
        trades_df = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['win'])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = sum(t['pnl_absolute'] for t in trades)
        total_pnl_pct = sum(t['pnl_percentage'] for t in trades)
        avg_win = np.mean([t['pnl_absolute'] for t in trades if t['win']]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl_absolute'] for t in trades if not t['win']]) if losing_trades > 0 else 0
        
        # Risk metrics
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        largest_win = max((t['pnl_absolute'] for t in trades), default=0)
        largest_loss = min((t['pnl_absolute'] for t in trades), default=0)
        
        # Duration metrics
        avg_duration = np.mean([t['duration'].total_seconds() / 60 for t in trades])  # minutes
        avg_winning_duration = np.mean([
            t['duration'].total_seconds() / 60 for t in trades if t['win']
        ]) if winning_trades > 0 else 0
        avg_losing_duration = np.mean([
            t['duration'].total_seconds() / 60 for t in trades if not t['win']
        ]) if losing_trades > 0 else 0
        
        # Commission impact
        total_commission = sum(t['commission'] for t in trades)
        commission_pct = (total_commission / self.initial_balance) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_percentage': total_pnl_pct,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'average_duration_minutes': avg_duration,
            'avg_winning_duration': avg_winning_duration,
            'avg_losing_duration': avg_losing_duration,
            'total_commission': total_commission,
            'commission_percentage': commission_pct,
            'net_return_percentage': (total_pnl / self.initial_balance) * 100
        }
    
    def _calculate_risk_metrics(self, trades):
        """Calculate risk-adjusted performance metrics"""
        if not trades:
            return {}
        
        pnl_series = pd.Series([t['pnl_absolute'] for t in trades])
        
        # Drawdown analysis
        cumulative_pnl = pnl_series.cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / self.initial_balance) * 100
        
        # Sharpe ratio (simplified)
        returns = pnl_series / self.initial_balance
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Recovery factor
        recovery_factor = abs(cumulative_pnl.iloc[-1] / max_drawdown) if max_drawdown < 0 else float('inf')
        
        # Consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for trade in trades:
            if not trade['win']:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_percentage': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'recovery_factor': recovery_factor,
            'max_consecutive_losses': max_consecutive_losses,
            'volatility': returns.std() * 100,
            'var_95': np.percentile(pnl_series, 5),  # Value at Risk (95%)
            'expectancy': pnl_series.mean()
        }
    
    def _build_equity_curve(self, trades):
        """Build equity curve for visualization"""
        equity_curve = []
        balance = self.initial_balance
        
        for trade in trades:
            balance += trade['pnl_absolute']
            equity_curve.append({
                'timestamp': trade['exit_time'],
                'balance': balance,
                'trade_pnl': trade['pnl_absolute'],
                'cumulative_return': ((balance - self.initial_balance) / self.initial_balance) * 100
            })
        
        return equity_curve
    
    def _analyze_trade_patterns(self, trades, data):
        """Analyze trading patterns and performance by conditions"""
        if not trades:
            return {}
        
        analysis = {}
        
        # Performance by timeframe
        timeframe_perf = {}
        for trade in trades:
            tf = trade['timeframe']
            if tf not in timeframe_perf:
                timeframe_perf[tf] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
            
            timeframe_perf[tf]['trades'] += 1
            if trade['win']:
                timeframe_perf[tf]['wins'] += 1
            timeframe_perf[tf]['total_pnl'] += trade['pnl_absolute']
        
        analysis['timeframe_performance'] = timeframe_perf
        
        # Performance by signal strength
        strength_ranges = [(0, 40), (40, 60), (60, 80), (80, 100)]
        strength_perf = {}
        
        for low, high in strength_ranges:
            range_key = f"{low}-{high}%"
            strength_perf[range_key] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
            
            for trade in trades:
                if low <= trade['signal_strength'] < high:
                    strength_perf[range_key]['trades'] += 1
                    if trade['win']:
                        strength_perf[range_key]['wins'] += 1
                    strength_perf[range_key]['total_pnl'] += trade['pnl_absolute']
        
        analysis['signal_strength_performance'] = strength_perf
        
        # Performance by exit reason
        exit_perf = {}
        for trade in trades:
            reason = trade['exit_reason']
            if reason not in exit_perf:
                exit_perf[reason] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
            
            exit_perf[reason]['trades'] += 1
            if trade['win']:
                exit_perf[reason]['wins'] += 1
            exit_perf[reason]['total_pnl'] += trade['pnl_absolute']
        
        analysis['exit_reason_performance'] = exit_perf
        
        # Best/worst performing hours (if enough data)
        if len(trades) > 10:
            hourly_perf = {}
            for trade in trades:
                hour = trade['entry_time'].hour
                if hour not in hourly_perf:
                    hourly_perf[hour] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
                
                hourly_perf[hour]['trades'] += 1
                if trade['win']:
                    hourly_perf[hour]['wins'] += 1
                hourly_perf[hour]['total_pnl'] += trade['pnl_absolute']
            
            analysis['hourly_performance'] = hourly_perf
        
        return analysis
    
    def get_strategy_recommendations(self, backtest_results):
        """Generate strategy improvement recommendations based on backtest results"""
        recommendations = []
        
        if not backtest_results['trades']:
            return ["Insufficient trade data for recommendations"]
        
        perf = backtest_results['performance_metrics']
        risk = backtest_results['risk_metrics']
        patterns = backtest_results['detailed_analysis']
        
        # Win rate recommendations
        if perf['win_rate'] < 40:
            recommendations.append("Low win rate detected. Consider tightening entry criteria or improving signal quality.")
        elif perf['win_rate'] > 70:
            recommendations.append("Excellent win rate. Consider increasing position size slightly if drawdown allows.")
        
        # Profit factor recommendations
        if perf['profit_factor'] < 1.2:
            recommendations.append("Poor profit factor. Focus on cutting losses faster or letting winners run longer.")
        elif perf['profit_factor'] > 2.0:
            recommendations.append("Strong profit factor. Strategy shows good risk/reward balance.")
        
        # Drawdown recommendations
        if risk['max_drawdown_percentage'] > 10:
            recommendations.append("High drawdown detected. Consider reducing position size or improving risk management.")
        
        # Duration recommendations
        if perf['avg_losing_duration'] > perf['avg_winning_duration'] * 1.5:
            recommendations.append("Losing trades held too long. Implement stricter stop losses or time exits.")
        
        # Signal strength recommendations
        if 'signal_strength_performance' in patterns:
            strength_data = patterns['signal_strength_performance']
            low_strength = strength_data.get('0-40%', {})
            high_strength = strength_data.get('80-100%', {})
            
            if low_strength.get('trades', 0) > 0:
                low_win_rate = (low_strength.get('wins', 0) / low_strength['trades']) * 100
                if low_win_rate < 30:
                    recommendations.append("Low-strength signals performing poorly. Consider filtering out signals below 40% strength.")
        
        # Commission impact
        if perf['commission_percentage'] > 2:
            recommendations.append("High commission costs. Consider reducing trade frequency or finding lower-cost broker.")
        
        return recommendations