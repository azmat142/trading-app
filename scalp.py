import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import feedparser

# ML Imports
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

# HuggingFace NLP Import with Fallback
try:
    from transformers import pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception:
    sentiment_pipeline = None

# ==========================================
# 1. DATA RETRIEVAL & FEATURE ENGINEERING
# ==========================================

@st.cache_data(ttl=300)
def fetch_market_data(ticker: str, period: str = "60d", interval: str = "15m") -> pd.DataFrame:
    """Fetch historical OHLCV data using yfinance."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create stationary technical indicators as ML inputs."""
    data = df.copy()

    # Stationarity: Relative Price Changes
    data['returns'] = data['Close'].pct_change()
    data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Moving Average Deviations (Normalized)
    data['sma_10'] = data['Close'].rolling(window=10).mean()
    data['sma_50'] = data['Close'].rolling(window=50).mean()
    data['dist_sma10'] = (data['Close'] - data['sma_10']) / data['sma_10']
    data['dist_sma50'] = (data['Close'] - data['sma_50']) / data['sma_50']
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data['rsi'] = 100 - (100 / (1 + rs))

    # MACD Histogram (Normalized)
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    data['macd_hist'] = (macd - signal) / data['Close']

    # Volatility & Volume Change
    data['volatility'] = data['returns'].rolling(window=14).std()
    data['volume_change'] = data['Volume'].pct_change()

    # Target: 1 if future price in N bars is higher, 0 otherwise
    data['target'] = (data['Close'].shift(-3) > data['Close']).astype(int)

    data.dropna(inplace=True)
    return data

# ==========================================
# 2. SENTIMENT ANALYSIS
# ==========================================

@st.cache_data(ttl=900)
def fetch_sentiment_score(ticker: str) -> float:
    """Fetch news RSS feed and calculate sentiment score (-1.0 to 1.0)."""
    rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+when:1d&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    
    if not feed.entries:
        return 0.0  # Neutral default

    headlines = [entry.title for entry in feed.entries[:5]]
    scores = []

    if sentiment_pipeline:
        try:
            results = sentiment_pipeline(headlines)
            for res in results:
                label, score = res['label'], res['score']
                if label == 'positive':
                    scores.append(score)
                elif label == 'negative':
                    scores.append(-score)
                else:
                    scores.append(0.0)
            return float(np.mean(scores))
        except Exception:
            pass

    # Simple Keyword Fallback if FinBERT is unavailable
    positive_words = {'bull', 'growth', 'surge', 'up', 'high', 'gain', 'profit', 'buy'}
    negative_words = {'bear', 'drop', 'fall', 'down', 'low', 'loss', 'sell', 'risk'}

    for headline in headlines:
        words = set(headline.lower().split())
        pos_count = len(words.intersection(positive_words))
        neg_count = len(words.intersection(negative_words))
        if pos_count + neg_count > 0:
            scores.append((pos_count - neg_count) / (pos_count + neg_count))
        else:
            scores.append(0.0)

    return float(np.mean(scores)) if scores else 0.0

# ==========================================
# 3. CALIBRATED ML MODEL PREDICTION
# ==========================================

def predict_asset_direction(
    df: pd.DataFrame, 
    sentiment_score: float, 
    min_confidence: float = 0.65
) -> tuple[str, float, dict]:
    """Train calibrated XGBoost model and generate actionable market signals."""
    feature_cols = [
        'returns', 'log_ret', 'dist_sma10', 'dist_sma50', 
        'rsi', 'macd_hist', 'volatility', 'volume_change'
    ]
    
    X = df[feature_cols].copy()
    X['sentiment'] = sentiment_score
    y = df['target']

    # Train / Test split (Time-series chronological split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Base Estimator: Un-degraded shallow XGBoost to prevent overfitting
    base_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    # Calibrate Probabilities via Sigmoid/Platt Scaling
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method='sigmoid',
        cv=3
    )
    calibrated_model.fit(X_train_scaled, y_train)

    # Predict latest bar with calibrated probabilities
    latest_features = X_test_scaled[-1:]
    probs = calibrated_model.predict_proba(latest_features)[0]  # [Prob(SELL), Prob(BUY)]
    
    raw_confidence = float(np.max(probs))
    predicted_class = int(np.argmax(probs))

    # Strict Confidence Threshold Filter
    if raw_confidence < min_confidence:
        signal = "HOLD / NEUTRAL"
    else:
        signal = "BUY" if predicted_class == 1 else "SELL"

    metrics = {
        "Prob(BUY)": round(float(probs[1]) * 100, 2),
        "Prob(SELL)": round(float(probs[0]) * 100, 2),
        "Sentiment Score": round(sentiment_score, 3)
    }

    return signal, raw_confidence, metrics

# ==========================================
# 4. STREAMLIT FRONTEND DASHBOARD
# ==========================================

def main():
    st.set_page_config(page_title="Trade Ideas Pro", layout="wide")
    st.title("📈 Trade Ideas Pro (Scalp & Trend Analytics)")

    # Sidebar Controls
    st.sidebar.header("Strategy Settings")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
    timeframe = st.sidebar.selectbox("Select Interval", options=["5m", "15m", "1h", "1d"], index=1)
    period_map = {"5m": "7d", "15m": "60d", "1h": "60d", "1d": "2y"}
    
    st.sidebar.markdown("---")
    min_conf = st.sidebar.slider(
        "Min Confidence Filter", 
        min_value=0.55, 
        max_value=0.85, 
        value=0.65, 
        step=0.05,
        help="Signals below this probability threshold default to HOLD / NEUTRAL."
    )

    if st.sidebar.button("Run Model Prediction", type="primary"):
        with st.spinner("Fetching market data and running calibrated model..."):
            try:
                # 1. Fetch & Engineer Data
                raw_df = fetch_market_data(ticker, period=period_map[timeframe], interval=timeframe)
                if raw_df.empty:
                    st.error(f"No data returned for ticker '{ticker}'. Verify the symbol.")
                    return

                processed_df = generate_features(raw_df)
                
                # 2. Fetch Sentiment
                sentiment = fetch_sentiment_score(ticker)
                
                # 3. Model Inference
                signal, confidence, metrics = predict_asset_direction(
                    processed_df, 
                    sentiment_score=sentiment, 
                    min_confidence=min_conf
                )

                # Output Metrics Header
                col1, col2, col3, col4 = st.columns(4)
                
                # Dynamic Metric Colors
                if signal == "BUY":
                    col1.metric("Model Signal", signal, delta="Bullish Edge", delta_color="normal")
                elif signal == "SELL":
                    col1.metric("Model Signal", signal, delta="-Bearish Edge", delta_color="inverse")
                else:
                    col1.metric("Model Signal", signal, delta="Low Conviction", delta_color="off")

                col2.metric("Calibrated Confidence", f"{confidence * 100:.1f}%")
                col3.metric("Buy Probability", f"{metrics['Prob(BUY)']}%")
                col4.metric("Sentiment Index", f"{metrics['Sentiment Score']}")

                # Plot Candlestick Chart
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
