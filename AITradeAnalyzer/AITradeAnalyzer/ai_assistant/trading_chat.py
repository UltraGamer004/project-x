import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

class TradingChatAssistant:
    def __init__(self):
        self.session_state_key = "trading_chat_history"
        self.analysis_cache = {}
        
        # Initialize chat history
        if self.session_state_key not in st.session_state:
            st.session_state[self.session_state_key] = [
                {
                    "role": "assistant", 
                    "content": "Hello! I'm your AI trading assistant specialized in 1-minute and 15-minute scalping. I'll help you analyze market conditions, identify optimal entries, and provide real-time trade guidance. How can I assist you today?",
                    "timestamp": datetime.now()
                }
            ]
    
    def analyze_market_conditions(self, data, signals, volume_analysis, news_data, timeframe):
        """Comprehensive market analysis for scalping"""
        analysis = {
            "timeframe": timeframe,
            "timestamp": datetime.now(),
            "market_summary": {},
            "scalping_conditions": {},
            "risk_assessment": {},
            "trade_recommendations": []
        }
        
        if data is None or data.empty:
            return analysis
        
        latest_candle = data.iloc[-1]
        prev_candles = data.tail(20)
        
        # Market Summary
        analysis["market_summary"] = {
            "current_price": latest_candle['Close'],
            "price_change_24h": ((latest_candle['Close'] - data.iloc[-1440]['Close']) / data.iloc[-1440]['Close'] * 100) if len(data) > 1440 else 0,
            "volatility": prev_candles['High'].std() / prev_candles['Close'].mean() * 100,
            "volume_trend": "High" if latest_candle.get('Volume', 0) > prev_candles['Volume'].mean() * 1.5 else "Normal",
            "trend_direction": self._determine_trend(prev_candles)
        }
        
        # Scalping Conditions Assessment
        analysis["scalping_conditions"] = self._assess_scalping_conditions(data, volume_analysis, timeframe)
        
        # Risk Assessment
        analysis["risk_assessment"] = self._assess_risk_levels(data, signals, news_data)
        
        # Trade Recommendations
        analysis["trade_recommendations"] = self._generate_trade_recommendations(data, signals, analysis)
        
        return analysis
    
    def _determine_trend(self, data):
        """Determine short-term trend direction"""
        if len(data) < 10:
            return "Uncertain"
        
        recent_highs = data['High'].tail(10)
        recent_lows = data['Low'].tail(10)
        
        if recent_highs.is_monotonic_increasing and recent_lows.is_monotonic_increasing:
            return "Strong Bullish"
        elif recent_highs.iloc[-1] > recent_highs.iloc[-5] and recent_lows.iloc[-1] > recent_lows.iloc[-5]:
            return "Bullish"
        elif recent_highs.is_monotonic_decreasing and recent_lows.is_monotonic_decreasing:
            return "Strong Bearish"
        elif recent_highs.iloc[-1] < recent_highs.iloc[-5] and recent_lows.iloc[-1] < recent_lows.iloc[-5]:
            return "Bearish"
        else:
            return "Sideways"
    
    def _assess_scalping_conditions(self, data, volume_analysis, timeframe):
        """Assess if market conditions are favorable for scalping"""
        conditions = {
            "favorable": False,
            "volatility_score": 0,
            "liquidity_score": 0,
            "trend_clarity": 0,
            "reasons": []
        }
        
        if len(data) < 20:
            conditions["reasons"].append("Insufficient data for analysis")
            return conditions
        
        recent_data = data.tail(20)
        
        # Volatility Assessment (ideal range for scalping)
        atr_pct = recent_data.get('ATR', pd.Series([0])).iloc[-1] / recent_data['Close'].iloc[-1] * 100
        if timeframe == "1m":
            ideal_volatility = 0.05 <= atr_pct <= 0.15  # 0.05% to 0.15% ATR for 1-min
        else:  # 15m
            ideal_volatility = 0.15 <= atr_pct <= 0.4   # 0.15% to 0.4% ATR for 15-min
        
        conditions["volatility_score"] = min(100, atr_pct * 200) if ideal_volatility else max(0, 100 - abs(atr_pct - 0.1) * 500)
        
        # Liquidity Assessment
        if volume_analysis and 'features' in volume_analysis:
            volume_features = volume_analysis['features'].iloc[-1]
            volume_ratio = volume_features.get('volume_ratio', 1)
            conditions["liquidity_score"] = min(100, volume_ratio * 50)
            
            if volume_ratio > 1.5:
                conditions["reasons"].append("High volume supports scalping")
            elif volume_ratio < 0.8:
                conditions["reasons"].append("Low volume - reduce position size")
        
        # Trend Clarity
        price_range = (recent_data['High'].max() - recent_data['Low'].min()) / recent_data['Close'].iloc[-1] * 100
        if timeframe == "1m":
            good_range = 0.1 <= price_range <= 0.5
        else:
            good_range = 0.3 <= price_range <= 1.0
        
        conditions["trend_clarity"] = 100 if good_range else max(0, 100 - abs(price_range - 0.3) * 200)
        
        # Overall Assessment
        avg_score = (conditions["volatility_score"] + conditions["liquidity_score"] + conditions["trend_clarity"]) / 3
        conditions["favorable"] = avg_score > 60
        
        if conditions["favorable"]:
            conditions["reasons"].append(f"Good scalping conditions (Score: {avg_score:.0f}/100)")
        else:
            conditions["reasons"].append(f"Suboptimal scalping conditions (Score: {avg_score:.0f}/100)")
        
        return conditions
    
    def _assess_risk_levels(self, data, signals, news_data):
        """Assess current risk levels"""
        risk = {
            "overall_risk": "Medium",
            "risk_score": 50,
            "factors": []
        }
        
        if data is None or data.empty:
            return risk
        
        recent_data = data.tail(50)
        risk_factors = []
        risk_score = 50
        
        # Volatility Risk
        volatility = recent_data['Close'].pct_change().std() * 100
        if volatility > 1.0:
            risk_factors.append("High volatility detected")
            risk_score += 20
        elif volatility < 0.1:
            risk_factors.append("Very low volatility - limited profit potential")
            risk_score += 10
        
        # News Risk
        if news_data is not None and not news_data.empty:
            recent_news = news_data[news_data['published'] >= datetime.now() - timedelta(hours=2)]
            if len(recent_news) > 5:
                risk_factors.append("High news activity - increased volatility expected")
                risk_score += 15
        
        # Market Structure Risk
        if signals is not None and not signals.empty:
            latest_signal = signals.iloc[-1]
            if latest_signal.get('strength', 0) < 40:
                risk_factors.append("Weak signal strength")
                risk_score += 10
        
        # Time-based Risk (avoid major news times)
        current_hour = datetime.now().hour
        if current_hour in [8, 9, 13, 14, 20, 21]:  # Major market open/close times
            risk_factors.append("High-impact trading session")
            risk_score += 10
        
        risk["risk_score"] = min(100, risk_score)
        risk["factors"] = risk_factors
        
        if risk_score < 40:
            risk["overall_risk"] = "Low"
        elif risk_score > 70:
            risk["overall_risk"] = "High"
        
        return risk
    
    def _generate_trade_recommendations(self, data, signals, analysis):
        """Generate specific trade recommendations"""
        recommendations = []
        
        if data is None or data.empty:
            return recommendations
        
        current_price = data['Close'].iloc[-1]
        scalping_conditions = analysis["scalping_conditions"]
        risk_assessment = analysis["risk_assessment"]
        
        # Entry Recommendations
        if signals is not None and not signals.empty:
            latest_signal = signals.iloc[-1]
            
            if latest_signal.get('signal', 0) != 0 and scalping_conditions["favorable"]:
                direction = "BUY" if latest_signal['signal'] == 1 else "SELL"
                
                recommendation = {
                    "type": "ENTRY",
                    "direction": direction,
                    "entry_price": current_price,
                    "stop_loss": latest_signal.get('sl', 0),
                    "take_profit": latest_signal.get('tp', 0),
                    "confidence": latest_signal.get('strength', 0),
                    "rationale": f"Strong {direction.lower()} signal with favorable scalping conditions"
                }
                
                # Adjust for scalping (tighter levels)
                if recommendation["stop_loss"] and recommendation["take_profit"]:
                    risk = abs(current_price - recommendation["stop_loss"]) / current_price
                    if risk > 0.002:  # If risk > 0.2%, adjust for scalping
                        if direction == "BUY":
                            recommendation["stop_loss"] = current_price * 0.998  # 0.2% risk
                            recommendation["take_profit"] = current_price * 1.004  # 0.4% profit (1:2 RR)
                        else:
                            recommendation["stop_loss"] = current_price * 1.002
                            recommendation["take_profit"] = current_price * 0.996
                
                recommendations.append(recommendation)
        
        # Risk Management Recommendations
        if risk_assessment["risk_score"] > 70:
            recommendations.append({
                "type": "RISK_WARNING",
                "message": "High risk detected. Consider reducing position size or waiting for better conditions.",
                "factors": risk_assessment["factors"]
            })
        
        # Market Condition Recommendations
        if not scalping_conditions["favorable"]:
            recommendations.append({
                "type": "WAIT",
                "message": "Market conditions not optimal for scalping. Consider waiting or switching to higher timeframe.",
                "reasons": scalping_conditions["reasons"]
            })
        
        return recommendations
    
    def process_user_message(self, message, market_data=None):
        """Process user message and generate AI response"""
        
        # Add user message to history
        st.session_state[self.session_state_key].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now()
        })
        
        # Generate AI response based on message content
        response = self._generate_ai_response(message, market_data)
        
        # Add AI response to history
        st.session_state[self.session_state_key].append({
            "role": "assistant", 
            "content": response,
            "timestamp": datetime.now()
        })
        
        return response
    
    def _generate_ai_response(self, message, market_data):
        """Generate contextual AI response"""
        message_lower = message.lower()
        
        # Market analysis requests
        if any(word in message_lower for word in ['analyze', 'analysis', 'market', 'conditions']):
            if market_data:
                analysis = self.analyze_market_conditions(
                    market_data.get('data'),
                    market_data.get('signals'),
                    market_data.get('volume_analysis'),
                    market_data.get('news_data'),
                    market_data.get('timeframe', '1m')
                )
                return self._format_market_analysis_response(analysis)
            else:
                return "I need current market data to provide analysis. Please run the market analysis first."
        
        # Trade entry requests
        elif any(word in message_lower for word in ['entry', 'trade', 'buy', 'sell', 'signal']):
            if market_data and market_data.get('signals') is not None:
                return self._format_trade_entry_response(market_data)
            else:
                return "Let me analyze current signals for you. Please run the market analysis to get trade recommendations."
        
        # Risk assessment requests
        elif any(word in message_lower for word in ['risk', 'safe', 'dangerous', 'stop']):
            if market_data:
                risk = self._assess_risk_levels(
                    market_data.get('data'),
                    market_data.get('signals'),
                    market_data.get('news_data')
                )
                return f"Current risk level: {risk['overall_risk']} (Score: {risk['risk_score']}/100)\n\nRisk factors:\n" + \
                       "\n".join([f"• {factor}" for factor in risk['factors']])
            else:
                return "I need current market data to assess risk levels."
        
        # General scalping advice
        elif any(word in message_lower for word in ['scalp', 'scalping', 'quick', 'fast']):
            return """For successful scalping on 1m and 15m timeframes:

**Key Rules:**
• Only trade during high-volume periods
• Use tight stop-losses (0.1-0.3% for 1m, 0.2-0.5% for 15m)
• Target 1:2 or 1:3 risk-reward ratios
• Avoid major news events unless you're experienced
• Focus on liquid pairs like BTCUSDT

**Best Times to Scalp:**
• 08:00-10:00 UTC (European open)
• 13:00-15:00 UTC (US open overlap)
• 20:00-22:00 UTC (Asian session start)

Would you like me to analyze current market conditions for scalping opportunities?"""
        
        # Default helpful response
        else:
            return """I'm here to help with your scalping strategy! I can assist with:

• **Market Analysis**: Real-time conditions assessment
• **Trade Signals**: Entry/exit recommendations with precise levels
• **Risk Management**: Position sizing and stop-loss guidance
• **News Impact**: How current events affect your trades
• **Session Analysis**: Best times to trade

What would you like me to help you with? Try asking:
- "Analyze current market conditions"
- "Should I enter a trade now?"
- "What's the current risk level?"
- "Give me scalping advice"""
    
    def _format_market_analysis_response(self, analysis):
        """Format market analysis into readable response"""
        summary = analysis["market_summary"]
        conditions = analysis["scalping_conditions"]
        risk = analysis["risk_assessment"]
        
        response = f"""**Market Analysis for {analysis['timeframe']} Scalping:**

**Current Market:**
• Price: ${summary['current_price']:.5f}
• 24h Change: {summary['price_change_24h']:.2f}%
• Volatility: {summary['volatility']:.2f}%
• Trend: {summary['trend_direction']}
• Volume: {summary['volume_trend']}

**Scalping Conditions: {'✅ FAVORABLE' if conditions['favorable'] else '❌ UNFAVORABLE'}**
• Volatility Score: {conditions['volatility_score']:.0f}/100
• Liquidity Score: {conditions['liquidity_score']:.0f}/100
• Trend Clarity: {conditions['trend_clarity']:.0f}/100

**Risk Assessment: {risk['overall_risk']} ({risk['risk_score']}/100)**
"""
        
        if risk['factors']:
            response += "\n**Risk Factors:**\n" + "\n".join([f"• {factor}" for factor in risk['factors']])
        
        if analysis["trade_recommendations"]:
            response += "\n\n**Trade Recommendations:**"
            for rec in analysis["trade_recommendations"]:
                if rec["type"] == "ENTRY":
                    response += f"\n• **{rec['direction']}** at ${rec['entry_price']:.5f}"
                    response += f"\n  SL: ${rec['stop_loss']:.5f} | TP: ${rec['take_profit']:.5f}"
                    response += f"\n  Confidence: {rec['confidence']:.0f}%"
                elif rec["type"] == "RISK_WARNING":
                    response += f"\n⚠️ **{rec['message']}**"
                elif rec["type"] == "WAIT":
                    response += f"\n⏸️ **{rec['message']}**"
        
        return response
    
    def _format_trade_entry_response(self, market_data):
        """Format trade entry recommendations"""
        signals = market_data.get('signals')
        if signals is None or signals.empty:
            return "No active trade signals at the moment. The market may be in a consolidation phase."
        
        latest_signal = signals.iloc[-1]
        
        if latest_signal.get('signal', 0) == 0:
            return "Currently no clear entry signals. I recommend waiting for better setup or analyzing higher timeframe for trend direction."
        
        direction = "BUY 🟢" if latest_signal['signal'] == 1 else "SELL 🔴"
        
        response = f"""**{direction} Signal Detected**

**Entry Details:**
• Direction: {direction}
• Entry Price: ${latest_signal.get('entry_price', 0):.5f}
• Stop Loss: ${latest_signal.get('sl', 0):.5f}
• Take Profit: ${latest_signal.get('tp', 0):.5f}
• Signal Strength: {latest_signal.get('strength', 0):.0f}%

**Signal Components:**
• ICT Score: {latest_signal.get('ict_score', 0):.2f}
• SMC Score: {latest_signal.get('smc_score', 0):.2f}
• EMA Score: {latest_signal.get('ema_score', 0):.2f}
• Volume Score: {latest_signal.get('volume_score', 0):.2f}

**Risk/Reward Ratio: 1:{latest_signal.get('rr_ratio', 0):.1f}**
"""
        
        # Add scalping-specific advice
        if latest_signal.get('strength', 0) > 70:
            response += "\n✅ **High-confidence signal - good for scalping**"
        elif latest_signal.get('strength', 0) > 50:
            response += "\n⚠️ **Medium-confidence signal - reduce position size**"
        else:
            response += "\n❌ **Low-confidence signal - consider waiting for better setup**"
        
        return response
    
    def get_chat_history(self):
        """Get formatted chat history"""
        return st.session_state.get(self.session_state_key, [])
    
    def clear_chat_history(self):
        """Clear chat history"""
        st.session_state[self.session_state_key] = [
            {
                "role": "assistant", 
                "content": "Chat history cleared. How can I assist you with your trading today?",
                "timestamp": datetime.now()
            }
        ]