# AI Trading System - ICT/SMC

## Overview

This is an AI-powered trading system that combines Inner Circle Trader (ICT) and Smart Money Concepts (SMC) with machine learning for market analysis and trade signal generation. The application is built using Streamlit for the web interface and provides real-time market analysis, technical indicators, and ML-based trade predictions.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Single-page application with sidebar controls and multiple chart displays
- **Interactive Charts**: Plotly-based visualization for candlestick charts, technical indicators, and trading signals
- **Real-time Updates**: Dynamic data fetching and chart updates based on user selections
- **Responsive Design**: Dark theme optimized for trading environments

### Backend Architecture
- **Modular Python Structure**: Separated into distinct modules for data fetching, analysis, ML modeling, and visualization
- **Event-driven Processing**: User interactions trigger data fetching and analysis pipeline
- **In-memory Processing**: No persistent storage; all calculations performed in real-time

### Technology Stack
- **Python 3.11**: Core runtime environment
- **Streamlit**: Web framework for the user interface
- **Plotly**: Interactive charting and visualization
- **Pandas/NumPy**: Data manipulation and numerical computations
- **scikit-learn**: Machine learning models and preprocessing
- **yfinance**: Market data API integration

## Key Components

### Data Layer (`data/market_data.py`)
- **MarketDataFetcher**: Handles data retrieval from Yahoo Finance API
- **Supported Assets**: Forex pairs (EURUSD, GBPUSD, etc.) and stocks (AAPL, GOOGL, etc.)
- **Multiple Timeframes**: 1m to 1mo intervals with API limitation handling
- **Data Validation**: Error handling and data quality checks

### Technical Analysis Layer
- **ICT/SMC Analyzer** (`technical_analysis/ict_smc.py`): 
  - Swing point detection using scipy signal processing
  - Market structure analysis (BOS/CHOCH detection)
  - Liquidity zone identification
  - Fair Value Gap (FVG) detection
  - Order block analysis

- **EMA Analyzer** (`technical_analysis/ema_analysis.py`):
  - Multiple EMA periods (short: 8,13,21; medium: 50,89,144; long: 200,233,377)
  - EMA alignment analysis
  - Crossover detection
  - Dynamic support/resistance levels

### Machine Learning Layer (`ml_models/trade_predictor.py`)
- **Multiple Model Support**: Random Forest, SVM, Gradient Boosting
- **Feature Engineering**: Combines ICT/SMC and EMA features
- **Target Generation**: Future price movement classification (Buy/Sell/Hold)
- **Model Validation**: Cross-validation and performance metrics

### Signal Generation (`utils/trade_signals.py`)
- **Signal Fusion**: Combines ICT/SMC, EMA, and ML predictions
- **Risk Management**: Configurable risk/reward ratios and position sizing
- **Trade Level Calculation**: Entry, stop-loss, and take-profit levels
- **Signal Filtering**: Strength and confluence-based filtering

### Visualization Layer (`utils/plotting.py`)
- **Multi-panel Charts**: Price, signal strength, and volume displays
- **Signal Overlays**: Visual representation of all analysis components
- **Color-coded Themes**: Consistent bullish/bearish color scheme
- **Interactive Features**: Zoom, pan, and hover information

## Data Flow

1. **User Input**: Symbol and timeframe selection via Streamlit sidebar
2. **Data Fetching**: MarketDataFetcher retrieves OHLCV data from Yahoo Finance
3. **Technical Analysis**: Parallel processing of ICT/SMC and EMA analysis
4. **Feature Engineering**: Combination of technical analysis results into ML features
5. **ML Prediction**: Trade prediction using trained models
6. **Signal Generation**: Fusion of all analysis components into actionable signals
7. **Visualization**: Real-time chart updates with all signals and levels
8. **User Display**: Streamlit renders the complete analysis dashboard

## External Dependencies

### Data Sources
- **Yahoo Finance API**: Primary market data source via yfinance library
- **Real-time Limitations**: API rate limiting and historical data constraints

### Python Libraries
- **Core Data Processing**: pandas (2.3.0+), numpy (2.3.1+)
- **Machine Learning**: scikit-learn (1.7.0+)
- **Visualization**: plotly (6.1.2+), streamlit (1.46.0+)
- **Market Data**: yfinance (0.2.63+)
- **Signal Processing**: scipy (for peak detection)

## Deployment Strategy

### Replit Configuration
- **Runtime**: Python 3.11 with Nix package management
- **Auto-scaling**: Configured for automatic scaling deployment
- **Port Configuration**: Streamlit server runs on port 5000
- **Process Management**: Parallel workflow execution

### Environment Setup
- **Package Management**: UV lock file for reproducible dependencies
- **Configuration**: Streamlit config for dark theme and server settings
- **Resource Allocation**: Optimized for real-time data processing

### Development Workflow
- **Hot Reload**: Streamlit development server with auto-restart
- **Modular Testing**: Individual component testing capabilities
- **Performance Monitoring**: Built-in error handling and warnings suppression

## Changelog

```
Changelog:
- June 22, 2025: Enhanced AI Trading System completed
  * Added TradingView integration for BTCUSDT.PS and cryptocurrency pairs
  * Implemented advanced volume analysis with multiple indicators
  * Integrated real-time news sentiment analysis for crypto markets
  * Enhanced ML models with volume and news features
  * Added comprehensive volume pattern detection
  * Implemented multi-source news aggregation with sentiment scoring
  * Updated UI with news and volume analysis displays
- June 22, 2025: Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```