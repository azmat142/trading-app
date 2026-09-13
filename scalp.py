import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import feedparser
import gc
import nltk

# Download VADER lexicon silently for lightweight sentiment
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ML Imports
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

# Initialize Lightweight Sentiment Analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# ==========================================
# 1. PRESET TICKER DICTIONARIES
# ==========================================

ASSET_PRESETS = {
    "Forex": {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "CAD=X",
        "Gold (XAU/USD)": "GC=F",
        "Silver (XAG/USD)": "SI=F"
    },
    "Crypto": {
        "Bitcoin (BTC/USD)": "BTC-USD",
        "Ethereum (ETH/USD)": "ETH-USD",
        "Solana (SOL/USD)": "SOL-USD",
        "Ripple (XRP/USD)": "XRP-USD",
        "Cardano (ADA/USD)": "ADA-USD",
        "Dogecoin (DOGE/USD)": "DOGE-USD"
    },
    "Stocks": {
        "Apple (AAPL)": "AAPL",
        "NVIDIA (NVDA)": "NVDA",
        "Tesla (TSLA)": "TSLA",
        "Microsoft (MSFT)": "MSFT",
        "Amazon (AMZN)": "AMZN",
        "Meta (META)": "META"
    },
    "Indices": {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^IXIC",
        "Dow Jones": "^DJI"
    }
}

# ==========================================
# 2. DATA RETRIEVAL & FEATURE ENGINEERING
# ==========================================

@st.cache_data(ttl=300, max_entries=10)
def fetch_market_data(ticker: str, period: str = "30d", interval: str = "15m") -> pd.DataFrame:
    """Fetch historical OHLCV data using yfinance with strict caching."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create stationary technical indicators as ML inputs."""
    data = df.copy()

    # Relative Price Changes
    data['returns'] = data['Close'].pct_change()
    data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Moving Average Deviations
    data['sma_10'] = data['Close'].rolling(window=10).mean()
    data['sma_50'] = data['Close'].rolling(window=50).mean()
    data['dist_sma10'] = (data['Close'] - data['sma_10']) / (data['sma_10'] + 1e-9)
    data['dist_sma50'] = (data['Close'] - data['sma_50']) / (data['sma_50'] + 1e-9)
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data['rsi'] = 100 - (100 / (1 + rs))

    # MACD Histogram
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    data['macd_hist'] = (macd - signal) / (data['Close'] + 1e-9)

    # Volatility & Volume Change
    data['volatility'] = data['returns'].rolling(window=14).std()
    data['volume_change'] = data['Volume'].pct_change()

    # Target: 1 if future price in 3 bars is higher, 0 otherwise
    data['target'] = (data['Close'].shift(-3) > data['Close']).astype(int)

    # Clean missing values and infinity flags
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    return data

# ==========================================
# 3. LIGHTWEIGHT SENTIMENT ANALYSIS
# ==========================================

@st.cache_data(ttl=900, max_entries=20)
def fetch_sentiment_score(ticker: str) -> float:
    """Fetch news RSS feed and calculate sentiment using lightweight VADER."""
    clean_search = ticker.split('-')[0].split('=')[0].replace('^', '')
    rss_url = f"https://news.google.com/rss/search?q={clean_search}+market+when:1d&hl=en-US&gl=US&ceid=US:en"
    
    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries:
            return 0.0

        scores = []
        for entry in feed.entries[:5]:
            vs = vader_analyzer.polarity_scores(entry.title)
            scores.append(vs['compound'])

        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0

# ==========================================
# 4. CACHED MODEL TRAINING & PREDICTION
# ==========================================

@st.cache_resource(ttl=900, max_entries=5)
def get_calibrated_model(X_train_scaled: np.ndarray, y_train: pd.Series):
    """Global multi-thread safe model trainer."""
    base_model = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=1  # Prevent multi-threading CPU contention across sessions
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method='sigmoid',
        cv=3
    )
    calibrated_model.fit(X_train_scaled, y_train)
    return calibrated_model

def predict_asset_direction(
    df: pd.DataFrame, 
    sentiment_score: float, 
    min_confidence: float = 0.65
) -> tuple[str, float, dict]:
    """Inference engine with safety sanitization."""
    feature_cols = [
        'returns', 'log_ret', 'dist_sma10', 'dist_sma50', 
        'rsi', 'macd_hist', 'volatility', 'volume_change'
    ]
    
    X = df[feature_cols].copy()
    X['sentiment'] = sentiment_score
    y = df['target']

    # Sanitize inputs
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    calibrated_model = get_calibrated_model(X_train_scaled, y_train)

    latest_features = X_test_scaled[-1:]
    probs = calibrated_model.predict_proba(latest_features)[0]
    
    raw_confidence = float(np.max(probs))
    predicted_class = int(np.argmax(probs))

    if raw_confidence < min_confidence:
        signal = "HOLD / NEUTRAL"
    else:
        signal = "BUY" if predicted_class == 1 else "SELL"

    metrics = {
        "Prob(BUY)": round(float(probs[1]) * 100, 2),
        "Prob(SELL)": round(float(probs[0]) * 100, 2),
        "Sentiment Score": round(sentiment_score, 3)
    }

    # Free memory explicitly
    del X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    gc.collect()

    return signal, raw_confidence, metrics

# ==========================================
# 5. STREAMLIT FRONTEND DASHBOARD
# ==========================================

def main():
    st.set_page_config(page_title="Trade Ideas Pro", layout="wide")
    st.title("📈 Trade Ideas Pro (Scalp & Trend Analytics)")

    # Sidebar Controls
    st.sidebar.header("1. Select Market Asset")
    asset_class = st.sidebar.selectbox("Asset Class", options=["Forex", "Crypto", "Stocks", "Indices", "Custom Input"])

    if asset_class == "Custom Input":
        ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL").upper()
    else:
        preset_options = ASSET_PRESETS[asset_class]
        selected_name = st.sidebar.selectbox("Select Asset / Pair", options=list(preset_options.keys()))
        ticker = preset_options[selected_name]

    st.sidebar.header("2. Strategy Settings")
    timeframe = st.sidebar.selectbox("Timeframe / Interval", options=["5m", "15m", "1h", "1d"], index=1)
    period_map = {"5m": "5d", "15m": "30d", "1h": "30d", "1d": "1y"}
    
    st.sidebar.markdown("---")
    min_conf = st.sidebar.slider(
        "Min Confidence Threshold", 
        min_value=0.55, 
        max_value=0.85, 
        value=0.65, 
        step=0.01,
        help="Signals below this score default to HOLD. Recommended: 0.65 to 0.70 to avoid bad trades."
    )

    if st.sidebar.button("Run Model Prediction", type="primary"):
        with st.spinner(f"Analyzing {ticker} across indicators and news..."):
            try:
                raw_df = fetch_market_data(ticker, period=period_map[timeframe], interval=timeframe)
                if raw_df.empty:
                    st.error(f"No market data returned for ticker '{ticker}'.")
                    return

                processed_df = generate_features(raw_df)
                sentiment = fetch_sentiment_score(ticker)
                
                signal, confidence, metrics = predict_asset_direction(
                    processed_df, 
                    sentiment_score=sentiment, 
                    min_confidence=min_conf
                )

                col1, col2, col3, col4 = st.columns(4)
                
                if signal == "BUY":
                    col1.metric("Model Signal", signal, delta="BUY ENTRY", delta_color="normal")
                elif signal == "SELL":
                    col1.metric("Model Signal", signal, delta="SELL ENTRY", delta_color="inverse")
                else:
                    col1.metric("Model Signal", signal, delta="Low Confidence (HOLD)", delta_color="off")

                col2.metric("Calibrated Probability", f"{confidence * 100:.1f}%")
                col3.metric("Buy Probability", f"{metrics['Prob(BUY)']}%")
                col4.metric("Sentiment Score", f"{metrics['Sentiment Score']}")

                if signal == "HOLD / NEUTRAL":
                    st.warning(
                        f"⚠️ **Trade Skipped:** The model's win probability is **{confidence * 100:.1f}%**, "
                        f"which is below your required **{min_conf * 100:.0f}%** threshold. "
                        "Stay out of the market during low-conviction conditions."
                    )
                else:
                    st.success(f"✅ **Trade Setup Identified:** High probability signal with {confidence * 100:.1f}% confidence.")

                st.subheader(f"Price Action ({ticker})")
                fig = go.Figure(data=[go.Candlestick(
                    x=raw_df.index,
                    open=raw_df['Open'],
                    high=raw_df['High'],
                    low=raw_df['Low'],
                    close=raw_df['Close'],
                    name=ticker
                )])
                fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")

if __name__ == "__main__":
    main()
