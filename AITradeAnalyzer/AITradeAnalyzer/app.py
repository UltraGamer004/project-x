import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from ml_models.trade_predictor import TradePredictorML
from technical_analysis.ict_smc import ICTSMCAnalyzer
from technical_analysis.ema_analysis import EMAAnalyzer
from technical_analysis.volume_analysis import VolumeAnalyzer
from data.market_data import MarketDataFetcher
from data.tradingview_integration import TradingViewDataFetcher
from data.news_fetcher import NewsFetcher
from utils.trade_signals import TradeSignalGenerator
from utils.plotting import TradingChartPlotter
from ai_assistant.trading_chat import TradingChatAssistant
from ai_assistant.institutional_analysis import InstitutionalAnalyzer
from ai_assistant.backtesting_engine import ScalpingBacktester

# Page configuration
st.set_page_config(
    page_title="AI Trading System - ICT/SMC",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🤖 Interactive AI Scalping Assistant")
    st.subheader("1m & 15m Scalping with Advanced Analytics")
    
    # Initialize AI assistant and components
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = TradingChatAssistant()
        st.session_state.institutional_analyzer = InstitutionalAnalyzer()
        st.session_state.backtester = ScalpingBacktester()
    
    # Create main layout with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 AI Assistant", "📊 Market Analysis", "🔬 Backtesting", "⚙️ Settings"
    ])
    
    with tab4:  # Settings tab
        st.header("Trading Parameters")
        
        # Symbol selection with crypto focus
        symbol_type = st.radio(
            "Market Type",
            ["Cryptocurrency", "Forex", "Stocks"],
            index=0
        )
        
        if symbol_type == "Cryptocurrency":
            symbol = st.selectbox(
                "Select Crypto Pair",
                ["BTCUSDT.PS", "ETHUSDT.PS", "ADAUSDT.PS", "DOTUSDT.PS", 
                 "LINKUSDT.PS", "BNBUSDT.PS", "XRPUSDT.PS", "SOLUSDT.PS"],
                index=0
            )
        elif symbol_type == "Forex":
            symbol = st.selectbox(
                "Select Forex Pair",
                ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"],
                index=0
            )
        else:
            symbol = st.selectbox(
                "Select Stock",
                ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
                index=0
            )
        
        # Scalping timeframe selection
        timeframe = st.selectbox(
            "Scalping Timeframe",
            ["1m", "15m", "1h"],
            index=0,
            help="1m and 15m are optimal for scalping"
        )
        
        # Period for analysis
        period_days = st.slider("Analysis Period (Days)", 30, 365, 90)
        
        # ML Model parameters
        st.subheader("ML Model Settings")
        model_type = st.selectbox(
            "Model Type",
            ["Random Forest", "SVM", "Gradient Boosting"],
            index=0
        )
        
        retrain_model = st.checkbox("Retrain Model", value=False)
        
        # Risk parameters
        st.subheader("Risk Management")
        risk_reward_ratio = st.slider("Risk/Reward Ratio", 1.0, 5.0, 2.0, 0.1)
        max_risk_percent = st.slider("Max Risk %", 0.5, 5.0, 2.0, 0.1)
        
        # News analysis settings
        st.subheader("News Analysis")
        enable_news = st.checkbox("Enable News Sentiment Analysis", value=True)
        news_hours = st.slider("News Lookback (Hours)", 6, 48, 24)
        
        # Volume analysis settings
        st.subheader("Volume Analysis")
        enable_volume = st.checkbox("Enable Advanced Volume Analysis", value=True)
        volume_weight = st.slider("Volume Signal Weight", 0.1, 0.5, 0.3, 0.05)
        
        analyze_button = st.button("🔍 Analyze Market", type="primary")
        
        # Store settings in session state
        st.session_state.update({
            'symbol_type': symbol_type,
            'symbol': symbol,
            'timeframe': timeframe,
            'period_days': period_days,
            'model_type': model_type,
            'retrain_model': retrain_model,
            'risk_reward_ratio': risk_reward_ratio,
            'max_risk_percent': max_risk_percent,
            'enable_news': enable_news,
            'news_hours': news_hours,
            'enable_volume': enable_volume,
            'volume_weight': volume_weight
        })
    
    # AI Assistant Tab
    with tab1:
        st.header("💬 Interactive AI Trading Assistant")
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            chat_history = st.session_state.ai_assistant.get_chat_history()
            
            for message in chat_history[-10:]:  # Show last 10 messages
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    st.caption(f"🕐 {message['timestamp'].strftime('%H:%M:%S')}")
        
        # Chat input
        if prompt := st.chat_input("Ask me about market conditions, trade signals, or scalping strategy..."):
            # Get current market data for context
            market_data = st.session_state.get('current_market_data', None)
            
            # Process message with AI assistant
            response = st.session_state.ai_assistant.process_user_message(prompt, market_data)
            
            # Refresh to show new messages
            st.rerun()
        
        # Quick action buttons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📊 Analyze Market"):
                market_data = st.session_state.get('current_market_data', None)
                if market_data:
                    response = st.session_state.ai_assistant.process_user_message(
                        "Analyze current market conditions", market_data
                    )
                    st.rerun()
                else:
                    st.warning("Run market analysis first in the Market Analysis tab")
        
        with col2:
            if st.button("🎯 Get Entry Signal"):
                market_data = st.session_state.get('current_market_data', None)
                if market_data:
                    response = st.session_state.ai_assistant.process_user_message(
                        "Should I enter a trade now?", market_data
                    )
                    st.rerun()
                else:
                    st.warning("Run market analysis first")
        
        with col3:
            if st.button("⚠️ Risk Assessment"):
                market_data = st.session_state.get('current_market_data', None)
                if market_data:
                    response = st.session_state.ai_assistant.process_user_message(
                        "What's the current risk level?", market_data
                    )
                    st.rerun()
                else:
                    st.warning("Run market analysis first")
        
        with col4:
            if st.button("🧠 Scalping Tips"):
                response = st.session_state.ai_assistant.process_user_message(
                    "Give me scalping advice", None
                )
                st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.ai_assistant.clear_chat_history()
            st.rerun()
    
    # Market Analysis Tab  
    with tab2:
        st.header("📊 Advanced Market Analysis")
        
        # Display current analysis if available
        if 'current_market_data' in st.session_state:
            market_data = st.session_state.current_market_data
            
            # Current market overview
            col1, col2, col3, col4 = st.columns(4)
            
            if market_data.get('data') is not None:
                latest_price = market_data['data']['Close'].iloc[-1]
                price_change = ((latest_price - market_data['data']['Close'].iloc[-2]) / market_data['data']['Close'].iloc[-2] * 100) if len(market_data['data']) > 1 else 0
                
                with col1:
                    st.metric(
                        "Current Price", 
                        f"${latest_price:.5f}",
                        f"{price_change:+.2f}%"
                    )
                
                with col2:
                    if 'institutional_summary' in market_data:
                        inst_summary = market_data['institutional_summary']
                        st.metric("Risk Level", inst_summary.get('risk_interpretation', 'N/A'))
                
                with col3:
                    if 'institutional_summary' in market_data:
                        inst_summary = market_data['institutional_summary']
                        fear_greed = inst_summary.get('fear_greed_level', 50)
                        st.metric("Fear & Greed", f"{fear_greed:.0f}/100")
                
                with col4:
                    if market_data.get('signals') is not None and not market_data['signals'].empty:
                        latest_signal = market_data['signals'].iloc[-1]
                        signal_strength = latest_signal.get('strength', 0)
                        st.metric("Signal Strength", f"{signal_strength:.0f}%")
            
            # Institutional Analysis Display
            if 'institutional_summary' in market_data:
                st.subheader("🏛️ Institutional Analysis")
                inst_summary = market_data['institutional_summary']
                
                inst_col1, inst_col2, inst_col3 = st.columns(3)
                
                with inst_col1:
                    st.info("**Trading Session**")
                    st.write(f"Active: {inst_summary.get('current_session', 'Unknown').title()}")
                    
                    whale_activity = "🐋 Yes" if inst_summary.get('whale_activity', False) else "No"
                    st.write(f"Whale Activity: {whale_activity}")
                
                with inst_col2:
                    st.info("**Smart Money Flow**")
                    money_flow = inst_summary.get('smart_money_flow', 50)
                    if money_flow > 60:
                        st.success(f"Bullish: {money_flow:.0f}/100")
                    elif money_flow < 40:
                        st.error(f"Bearish: {money_flow:.0f}/100")
                    else:
                        st.warning(f"Neutral: {money_flow:.0f}/100")
                    
                    if inst_summary.get('accumulation_detected', False):
                        st.success("✅ Accumulation Detected")
                    elif inst_summary.get('distribution_detected', False):
                        st.error("❌ Distribution Detected")
                
                with inst_col3:
                    st.info("**Market Sentiment**")
                    sentiment = inst_summary.get('sentiment_interpretation', 'Neutral')
                    if sentiment in ['Extreme Greed', 'Greed']:
                        st.error(f"⚠️ {sentiment}")
                    elif sentiment in ['Extreme Fear', 'Fear']:
                        st.success(f"💚 {sentiment}")
                    else:
                        st.info(f"➖ {sentiment}")
        else:
            st.info("Click 'Analyze Market' in the Settings tab to begin analysis")
    
    # Backtesting Tab
    with tab3:
        st.header("🔬 Strategy Backtesting")
        
        if 'current_market_data' in st.session_state:
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("Backtest Settings")
                backtest_days = st.slider("Backtest Period (Days)", 1, 14, 7)
                min_signal_strength = st.slider("Min Signal Strength", 0, 100, 30)
                
                if st.button("🚀 Run Backtest", type="primary"):
                    with st.spinner("Running backtest simulation..."):
                        market_data = st.session_state.current_market_data
                        
                        # Filter signals by strength
                        signals = market_data.get('signals', pd.DataFrame())
                        if not signals.empty:
                            filtered_signals = signals[signals.get('strength', 0) >= min_signal_strength]
                            
                            # Run backtest
                            backtest_results = st.session_state.backtester.backtest_strategy(
                                market_data['data'], 
                                filtered_signals, 
                                st.session_state.get('timeframe', '1m'),
                                backtest_days
                            )
                            
                            st.session_state.backtest_results = backtest_results
                            st.rerun()
            
            with col1:
                if 'backtest_results' in st.session_state:
                    results = st.session_state.backtest_results
                    
                    # Performance metrics
                    st.subheader("📈 Performance Results")
                    perf = results['performance_metrics']
                    risk = results['risk_metrics']
                    
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric("Total Trades", perf.get('total_trades', 0))
                        st.metric("Win Rate", f"{perf.get('win_rate', 0):.1f}%")
                    
                    with metric_col2:
                        st.metric("Net Return", f"{perf.get('net_return_percentage', 0):.2f}%")
                        st.metric("Profit Factor", f"{perf.get('profit_factor', 0):.2f}")
                    
                    with metric_col3:
                        st.metric("Max Drawdown", f"{risk.get('max_drawdown_percentage', 0):.2f}%")
                        st.metric("Sharpe Ratio", f"{risk.get('sharpe_ratio', 0):.2f}")
                    
                    with metric_col4:
                        st.metric("Avg Trade Duration", f"{perf.get('average_duration_minutes', 0):.1f}m")
                        st.metric("Commission Cost", f"{perf.get('commission_percentage', 0):.2f}%")
                    
                    # Strategy recommendations
                    recommendations = st.session_state.backtester.get_strategy_recommendations(results)
                    if recommendations:
                        st.subheader("💡 Strategy Recommendations")
                        for rec in recommendations:
                            st.info(f"• {rec}")
                    
                    # Equity curve visualization
                    if results['equity_curve']:
                        st.subheader("📊 Equity Curve")
                        equity_df = pd.DataFrame(results['equity_curve'])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=equity_df['timestamp'],
                            y=equity_df['balance'],
                            mode='lines',
                            name='Balance',
                            line=dict(color='#00ff88', width=2)
                        ))
                        
                        fig.update_layout(
                            title='Account Balance Over Time',
                            xaxis_title='Time',
                            yaxis_title='Balance ($)',
                            template='plotly_dark',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run market analysis first to enable backtesting")
    
    # Execute analysis when button is clicked
    if analyze_button:
        with st.spinner("Fetching market data and analyzing..."):
            try:
                # Initialize components
                if symbol_type == "Cryptocurrency":
                    data_fetcher = TradingViewDataFetcher()
                    data = data_fetcher.fetch_crypto_data(symbol, period_days, timeframe)
                else:
                    data_fetcher = MarketDataFetcher()
                    data = data_fetcher.fetch_data(symbol, period_days, timeframe)
                
                ict_smc_analyzer = ICTSMCAnalyzer()
                ema_analyzer = EMAAnalyzer()
                volume_analyzer = VolumeAnalyzer()
                news_fetcher = NewsFetcher()
                institutional_analyzer = st.session_state.institutional_analyzer
                ml_predictor = TradePredictorML(model_type=model_type.lower().replace(" ", "_"))
                signal_generator = TradeSignalGenerator()
                chart_plotter = TradingChartPlotter()
                
                if data is None or data.empty:
                    st.error("Failed to fetch market data. Please try again.")
                    return
                
                # News Analysis (if enabled)
                news_data = pd.DataFrame()
                news_features = {}
                if enable_news and symbol_type == "Cryptocurrency":
                    with st.spinner("Analyzing market news and sentiment..."):
                        news_data = news_fetcher.fetch_crypto_news(hours_back=news_hours)
                        if not news_data.empty:
                            news_data = news_fetcher.analyze_sentiment(news_data)
                            news_features = news_fetcher.create_news_features(news_data)
                
                # Technical Analysis
                st.header("📊 Technical Analysis")
                
                # ICT/SMC Analysis
                ict_smc_signals = ict_smc_analyzer.analyze(data)
                
                # EMA Analysis
                ema_signals = ema_analyzer.analyze(data)
                
                # Volume Analysis (if enabled)
                volume_signals = {}
                if enable_volume:
                    volume_signals = volume_analyzer.analyze(data)
                
                # Institutional Analysis
                institutional_analysis = institutional_analyzer.analyze(data)
                institutional_summary = institutional_analyzer.get_current_summary(data)
                
                # Combine all features for ML
                feature_sets = [ict_smc_signals['features'], ema_signals['features']]
                
                if enable_volume and volume_signals:
                    feature_sets.append(volume_signals['features'])
                
                if institutional_analysis:
                    feature_sets.append(institutional_analysis['features'])
                
                if enable_news and news_features:
                    # Add news features as repeated values for each time period
                    news_df = pd.DataFrame([news_features] * len(data), index=data.index)
                    feature_sets.append(news_df)
                
                features_df = pd.concat(feature_sets, axis=1)
                
                # ML Prediction
                if retrain_model or not ml_predictor.is_trained():
                    with st.spinner("Training ML model..."):
                        ml_predictor.train(features_df, data)
                
                predictions = ml_predictor.predict(features_df)
                
                # Generate trade signals with enhanced components
                trade_signals = signal_generator.generate_signals(
                    data, ict_smc_signals, ema_signals, predictions,
                    risk_reward_ratio, max_risk_percent, volume_signals, news_features, volume_weight
                )
                
                # Store complete market data for AI assistant
                st.session_state.current_market_data = {
                    'data': data,
                    'signals': trade_signals,
                    'ict_smc_signals': ict_smc_signals,
                    'ema_signals': ema_signals,
                    'volume_analysis': volume_signals,
                    'news_data': news_data,
                    'institutional_analysis': institutional_analysis,
                    'institutional_summary': institutional_summary,
                    'timeframe': timeframe,
                    'symbol': symbol,
                    'symbol_type': symbol_type
                }
                
                # Success message and chart display
                st.success(f"✅ Analysis complete for {symbol} on {timeframe} timeframe")
                
                # Display results in Market Analysis tab (automatically switch)
                with tab2:
                    # Main trading chart
                    st.subheader("📈 Live Trading Chart with AI Signals")
                    fig = chart_plotter.create_main_chart(
                        data, ict_smc_signals, ema_signals, trade_signals
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Current signals and analysis
                    st.subheader("🎯 Current Signals")
                    
                    latest_signal = trade_signals.iloc[-1] if not trade_signals.empty else None
                    
                    if latest_signal is not None and latest_signal['signal'] != 0:
                        signal_type = "🟢 BUY" if latest_signal['signal'] == 1 else "🔴 SELL"
                        st.markdown(f"### {signal_type}")
                        
                        st.metric("Trade Strength", f"{latest_signal['strength']:.1f}%")
                        st.metric("Entry Price", f"{latest_signal['entry_price']:.5f}")
                        st.metric("Take Profit", f"{latest_signal['tp']:.5f}")
                        st.metric("Stop Loss", f"{latest_signal['sl']:.5f}")
                        st.metric("Risk/Reward", f"1:{latest_signal['rr_ratio']:.2f}")
                        
                        # Signal components
                        st.subheader("📋 Signal Components")
                        if latest_signal['ict_score'] > 0.5:
                            st.success(f"✅ ICT Score: {latest_signal['ict_score']:.2f}")
                        else:
                            st.info(f"➖ ICT Score: {latest_signal['ict_score']:.2f}")
                            
                        if latest_signal['smc_score'] > 0.5:
                            st.success(f"✅ SMC Score: {latest_signal['smc_score']:.2f}")
                        else:
                            st.info(f"➖ SMC Score: {latest_signal['smc_score']:.2f}")
                            
                        if latest_signal['ema_score'] > 0.5:
                            st.success(f"✅ EMA Score: {latest_signal['ema_score']:.2f}")
                        else:
                            st.info(f"➖ EMA Score: {latest_signal['ema_score']:.2f}")
                        
                        # Volume analysis display
                        if enable_volume and 'volume_score' in latest_signal:
                            if latest_signal['volume_score'] > 0.5:
                                st.success(f"✅ Volume Score: {latest_signal['volume_score']:.2f}")
                            else:
                                st.info(f"➖ Volume Score: {latest_signal['volume_score']:.2f}")
                            st.metric("Volume Confidence", f"{latest_signal.get('volume_confidence', 0):.1f}%")
                        
                        # News sentiment display
                        if enable_news and symbol_type == "Cryptocurrency" and 'news_sentiment' in latest_signal:
                            sentiment_value = latest_signal['news_sentiment']
                            if sentiment_value > 0.1:
                                st.success(f"✅ News Sentiment: +{sentiment_value:.2f}")
                            elif sentiment_value < -0.1:
                                st.error(f"❌ News Sentiment: {sentiment_value:.2f}")
                            else:
                                st.info(f"➖ News Sentiment: {sentiment_value:.2f}")
                            st.metric("News Confidence", f"{latest_signal.get('news_confidence', 0):.1f}%")
                            
                        st.metric("ML Confidence", f"{latest_signal['ml_confidence']:.1f}%")
                    else:
                        st.info("🔍 No active signals at this time")
                        st.markdown("Market conditions are not optimal for entry")
                
                # Enhanced Analysis Sections
                st.header("📈 Detailed Analysis")
                
                # News and Volume Analysis Row
                if enable_news and symbol_type == "Cryptocurrency" and not news_data.empty:
                    st.subheader("📰 News Sentiment Analysis")
                    col_news1, col_news2, col_news3 = st.columns(3)
                    
                    sentiment_summary = news_fetcher.get_market_sentiment_score(news_data)
                    
                    with col_news1:
                        sentiment_color = "normal"
                        if sentiment_summary['overall_sentiment'] > 0.1:
                            sentiment_color = "normal"  # Bullish
                        elif sentiment_summary['overall_sentiment'] < -0.1:
                            sentiment_color = "inverse"  # Bearish
                        
                        st.metric(
                            "Overall Sentiment", 
                            f"{sentiment_summary['overall_sentiment']:.3f}",
                            delta=None
                        )
                    
                    with col_news2:
                        st.metric("Bullish Articles", sentiment_summary['bullish_count'])
                        st.metric("Bearish Articles", sentiment_summary['bearish_count'])
                    
                    with col_news3:
                        st.metric("News Confidence", f"{sentiment_summary['confidence']:.1f}%")
                        st.metric("Total Articles", len(news_data))
                    
                    # Recent News Headlines
                    with st.expander("📋 Recent News Headlines"):
                        for _, article in news_data.head(5).iterrows():
                            sentiment_emoji = "🟢" if article['sentiment_label'] == 'positive' else "🔴" if article['sentiment_label'] == 'negative' else "⚪"
                            st.write(f"{sentiment_emoji} **{article['title']}**")
                            st.write(f"   *Sentiment: {article['sentiment_score']:.2f} | {article['published'].strftime('%H:%M %d/%m')}*")
                
                # Volume Analysis Display
                if enable_volume and volume_signals:
                    st.subheader("📊 Volume Analysis")
                    vol_col1, vol_col2, vol_col3 = st.columns(3)
                    
                    latest_volume = volume_signals['scores'].iloc[-1] if not volume_signals['scores'].empty else None
                    
                    if latest_volume is not None:
                        with vol_col1:
                            st.metric("Volume Score", f"{latest_volume.get('volume_score', 0):.2f}")
                            st.metric("Volume Confidence", f"{latest_volume.get('volume_confidence', 0):.1f}%")
                        
                        with vol_col2:
                            volume_features = volume_signals['features'].iloc[-1] if not volume_signals['features'].empty else None
                            if volume_features is not None:
                                st.metric("Volume Ratio", f"{volume_features.get('volume_ratio', 0):.2f}")
                                st.metric("High Volume Signal", "Yes" if volume_features.get('high_volume', 0) else "No")
                        
                        with vol_col3:
                            if volume_features is not None:
                                st.metric("Volume Breakout", "Yes" if volume_features.get('volume_breakout', 0) else "No")
                                st.metric("Accumulation", "Yes" if volume_features.get('accumulation', 0) else "No")
                
                # Market structure analysis
                col3, col4 = st.columns(2)
                
                with col3:
                    st.subheader("🏗️ Market Structure")
                    structure_fig = chart_plotter.create_structure_chart(ict_smc_signals)
                    st.plotly_chart(structure_fig, use_container_width=True)
                
                with col4:
                    st.subheader("📊 Signal History")
                    recent_signals = trade_signals.tail(10)
                    if not recent_signals.empty:
                        signal_summary = recent_signals.groupby('signal').agg({
                            'strength': 'mean',
                            'entry_price': 'count'
                        }).round(2)
                        signal_summary.columns = ['Avg Strength', 'Count']
                        signal_summary.index = signal_summary.index.map({
                            -1: '🔴 SELL', 0: '➖ NEUTRAL', 1: '🟢 BUY'
                        })
                        st.dataframe(signal_summary)
                    else:
                        st.info("No recent signals to display")
                
                # Performance metrics
                st.header("📊 Model Performance")
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    accuracy = ml_predictor.get_accuracy()
                    st.metric("Model Accuracy", f"{accuracy:.1f}%")
                
                with col6:
                    total_signals = len(trade_signals[trade_signals['signal'] != 0])
                    st.metric("Total Signals", total_signals)
                
                with col7:
                    avg_strength = trade_signals['strength'].mean() if not trade_signals.empty else 0
                    st.metric("Avg Signal Strength", f"{avg_strength:.1f}%")
                
                # Raw data (expandable)
                with st.expander("📋 View Raw Signal Data"):
                    st.dataframe(trade_signals.tail(20))
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.info("Please check your internet connection and try again.")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Enhanced AI Trading System! 🚀
        
        This advanced trading system now includes comprehensive analysis for cryptocurrency trading:
        
        ### 🧠 **Advanced Machine Learning**
        - Random Forest, SVM, and Gradient Boosting algorithms
        - Enhanced with volume and news sentiment features
        - Real-time prediction with multi-factor confidence scoring
        
        ### 📈 **ICT/SMC Technical Analysis**
        - Liquidity zones and Fair Value Gaps (FVG) detection
        - Order blocks and market structure analysis
        - Break of Structure (BOS) and Change of Character (CHoCH)
        - Multiple EMA timeframe alignment and trend confirmation
        
        ### 📊 **Advanced Volume Analysis**
        - Volume patterns and breakout detection
        - Accumulation/Distribution analysis
        - Volume-Price relationship confirmation
        - Money Flow Index and VWAP analysis
        
        ### 📰 **Real-time News Sentiment**
        - Cryptocurrency news aggregation from multiple sources
        - AI-powered sentiment analysis using natural language processing
        - Market impact assessment and confidence scoring
        - High-impact keyword detection for trade timing
        
        ### 🎯 **Enhanced Trade Management**
        - Multi-factor signal fusion (Technical + Volume + News + ML)
        - Dynamic entry, stop-loss, and take-profit calculation
        - Risk-based position sizing and trade strength scoring
        - Comprehensive trade probability assessment
        
        ### ₿ **TradingView Integration**
        - BTCUSDT and major cryptocurrency pairs support
        - Real-time price data with enhanced technical indicators
        - Crypto-specific volatility and momentum analysis
        
        **Select 'Cryptocurrency' and 'BTCUSDT.PS' to analyze Bitcoin with the full suite of enhanced features!**
        """)

if __name__ == "__main__":
    main()
