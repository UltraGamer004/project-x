import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class TradingChartPlotter:
    def __init__(self):
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff4444',
            'neutral': '#888888',
            'background': '#0e1117',
            'grid': '#262730',
            'text': '#fafafa'
        }
    
    def create_main_chart(self, data, ict_smc_signals, ema_signals, trade_signals):
        """Create the main trading chart with all signals and levels"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.15, 0.15],
            subplot_titles=('Price Chart with Signals', 'Signal Strength', 'Volume')
        )
        
        # Main price chart (candlesticks)
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish']
            ),
            row=1, col=1
        )
        
        # Add EMA lines
        self._add_ema_lines(fig, data, ema_signals)
        
        # Add ICT/SMC levels
        self._add_ict_smc_levels(fig, data, ict_smc_signals)
        
        # Add trade signals
        self._add_trade_signals(fig, data, trade_signals)
        
        # Add signal strength chart
        self._add_signal_strength_chart(fig, trade_signals)
        
        # Add volume chart
        self._add_volume_chart(fig, data)
        
        # Update layout
        fig.update_layout(
            title='AI Trading System - ICT/SMC Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def _add_ema_lines(self, fig, data, ema_signals):
        """Add EMA lines to the chart"""
        if 'ema_data' not in ema_signals:
            return
        
        ema_data = ema_signals['ema_data']
        
        # Key EMA lines to display
        key_emas = [
            ('ema_8', '#ff6b6b', 1),
            ('ema_21', '#4ecdc4', 2),
            ('ema_50', '#45b7d1', 2),
            ('ema_200', '#f39c12', 3)
        ]
        
        for ema_name, color, width in key_emas:
            if ema_name in ema_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=ema_data.index,
                        y=ema_data[ema_name],
                        mode='lines',
                        name=ema_name.upper(),
                        line=dict(color=color, width=width),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
    
    def _add_ict_smc_levels(self, fig, data, ict_smc_signals):
        """Add ICT/SMC levels and zones to the chart"""
        # Add Fair Value Gaps
        if 'fair_value_gaps' in ict_smc_signals:
            fvg = ict_smc_signals['fair_value_gaps']
            
            # Bullish FVGs
            bullish_fvg = fvg[fvg['bullish_fvg']].index
            for idx in bullish_fvg:
                try:
                    pos = data.index.get_loc(idx)
                    if pos >= 2:
                        gap_high = data['Low'].iloc[pos]
                        gap_low = data['High'].iloc[pos-2]
                        
                        fig.add_shape(
                            type="rect",
                            x0=data.index[pos-1], x1=data.index[min(pos+5, len(data)-1)],
                            y0=gap_low, y1=gap_high,
                            fillcolor=self.colors['bullish'],
                            opacity=0.2,
                            line=dict(width=0),
                            row=1, col=1
                        )
                except:
                    continue
            
            # Bearish FVGs
            bearish_fvg = fvg[fvg['bearish_fvg']].index
            for idx in bearish_fvg:
                try:
                    pos = data.index.get_loc(idx)
                    if pos >= 2:
                        gap_high = data['Low'].iloc[pos-2]
                        gap_low = data['High'].iloc[pos]
                        
                        fig.add_shape(
                            type="rect",
                            x0=data.index[pos-1], x1=data.index[min(pos+5, len(data)-1)],
                            y0=gap_low, y1=gap_high,
                            fillcolor=self.colors['bearish'],
                            opacity=0.2,
                            line=dict(width=0),
                            row=1, col=1
                        )
                except:
                    continue
        
        # Add Order Blocks
        if 'order_blocks' in ict_smc_signals:
            ob = ict_smc_signals['order_blocks']
            
            # Bullish Order Blocks
            bullish_ob = ob[ob['bullish_ob']].index
            for idx in bullish_ob:
                try:
                    pos = data.index.get_loc(idx)
                    fig.add_trace(
                        go.Scatter(
                            x=[idx],
                            y=[data['Low'].iloc[pos]],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up',
                                size=12,
                                color=self.colors['bullish']
                            ),
                            name='Bullish OB',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                except:
                    continue
            
            # Bearish Order Blocks
            bearish_ob = ob[ob['bearish_ob']].index
            for idx in bearish_ob:
                try:
                    pos = data.index.get_loc(idx)
                    fig.add_trace(
                        go.Scatter(
                            x=[idx],
                            y=[data['High'].iloc[pos]],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                size=12,
                                color=self.colors['bearish']
                            ),
                            name='Bearish OB',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                except:
                    continue
    
    def _add_trade_signals(self, fig, data, trade_signals):
        """Add trade signals and levels to the chart"""
        # Buy signals
        buy_signals = trade_signals[trade_signals['signal'] == 1]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['entry_price'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color=self.colors['bullish'],
                        line=dict(width=2, color='white')
                    ),
                    name='Buy Signals',
                    text=[f"Strength: {s:.1f}%" for s in buy_signals['strength']],
                    hovertemplate='<b>BUY SIGNAL</b><br>' +
                                'Entry: %{y:.5f}<br>' +
                                '%{text}<br>' +
                                '<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add TP/SL lines for buy signals
            for idx, signal in buy_signals.iterrows():
                if signal['tp'] > 0 and signal['sl'] > 0:
                    # Take Profit line
                    fig.add_shape(
                        type="line",
                        x0=idx, x1=data.index[-1],
                        y0=signal['tp'], y1=signal['tp'],
                        line=dict(color=self.colors['bullish'], width=1, dash="dot"),
                        row=1, col=1
                    )
                    
                    # Stop Loss line
                    fig.add_shape(
                        type="line",
                        x0=idx, x1=data.index[-1],
                        y0=signal['sl'], y1=signal['sl'],
                        line=dict(color=self.colors['bearish'], width=1, dash="dot"),
                        row=1, col=1
                    )
        
        # Sell signals
        sell_signals = trade_signals[trade_signals['signal'] == -1]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['entry_price'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color=self.colors['bearish'],
                        line=dict(width=2, color='white')
                    ),
                    name='Sell Signals',
                    text=[f"Strength: {s:.1f}%" for s in sell_signals['strength']],
                    hovertemplate='<b>SELL SIGNAL</b><br>' +
                                'Entry: %{y:.5f}<br>' +
                                '%{text}<br>' +
                                '<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add TP/SL lines for sell signals
            for idx, signal in sell_signals.iterrows():
                if signal['tp'] > 0 and signal['sl'] > 0:
                    # Take Profit line
                    fig.add_shape(
                        type="line",
                        x0=idx, x1=data.index[-1],
                        y0=signal['tp'], y1=signal['tp'],
                        line=dict(color=self.colors['bullish'], width=1, dash="dot"),
                        row=1, col=1
                    )
                    
                    # Stop Loss line
                    fig.add_shape(
                        type="line",
                        x0=idx, x1=data.index[-1],
                        y0=signal['sl'], y1=signal['sl'],
                        line=dict(color=self.colors['bearish'], width=1, dash="dot"),
                        row=1, col=1
                    )
    
    def _add_signal_strength_chart(self, fig, trade_signals):
        """Add signal strength chart"""
        # Signal strength line
        fig.add_trace(
            go.Scatter(
                x=trade_signals.index,
                y=trade_signals['strength'],
                mode='lines',
                name='Signal Strength',
                line=dict(color=self.colors['bullish'], width=2),
                fill='tonexty',
                fillcolor=f'rgba(0, 255, 136, 0.1)'
            ),
            row=2, col=1
        )
        
        # Add threshold line
        fig.add_hline(
            y=50, line_dash="dash", line_color=self.colors['neutral'],
            annotation_text="Threshold", row=2, col=1
        )
        
        # Color background based on signal strength
        strong_signals = trade_signals[trade_signals['strength'] >= 70]
        for idx, signal in strong_signals.iterrows():
            fig.add_vrect(
                x0=idx, x1=idx,
                fillcolor=self.colors['bullish'], opacity=0.2,
                layer="below", line_width=0,
                row=2, col=1
            )
    
    def _add_volume_chart(self, fig, data):
        """Add volume chart"""
        if 'Volume' in data.columns:
            colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                     for i in range(len(data))]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=3, col=1
            )
    
    def create_structure_chart(self, ict_smc_signals):
        """Create market structure analysis chart"""
        fig = go.Figure()
        
        if 'market_structure' in ict_smc_signals:
            structure = ict_smc_signals['market_structure']
            
            # Trend visualization
            trend_colors = []
            for trend in structure['trend']:
                if trend == 1:
                    trend_colors.append(self.colors['bullish'])
                elif trend == -1:
                    trend_colors.append(self.colors['bearish'])
                else:
                    trend_colors.append(self.colors['neutral'])
            
            fig.add_trace(
                go.Scatter(
                    x=structure.index,
                    y=structure['trend'],
                    mode='lines+markers',
                    name='Market Trend',
                    line=dict(width=3),
                    marker=dict(size=8, color=trend_colors)
                )
            )
        
        fig.update_layout(
            title='Market Structure Analysis',
            xaxis_title='Time',
            yaxis_title='Trend Direction',
            template='plotly_dark',
            height=300
        )
        
        return fig
    
    def create_performance_chart(self, backtest_results):
        """Create performance analysis chart"""
        if backtest_results.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No backtest data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Equity Curve', 'Trade Results')
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=backtest_results['exit_time'],
                y=backtest_results['balance'],
                mode='lines',
                name='Balance',
                line=dict(color=self.colors['bullish'], width=3)
            ),
            row=1, col=1
        )
        
        # Trade results
        wins = backtest_results[backtest_results['result'] == 'win']
        losses = backtest_results[backtest_results['result'] == 'loss']
        
        fig.add_trace(
            go.Scatter(
                x=wins['exit_time'],
                y=wins['pnl_percent'],
                mode='markers',
                name='Winning Trades',
                marker=dict(color=self.colors['bullish'], size=8, symbol='triangle-up')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=losses['exit_time'],
                y=losses['pnl_percent'],
                mode='markers',
                name='Losing Trades',
                marker=dict(color=self.colors['bearish'], size=8, symbol='triangle-down')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Backtesting Performance',
            height=600,
            template='plotly_dark'
        )
        
        return fig
