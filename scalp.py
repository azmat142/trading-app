import datetime
import warnings
import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import yfinance as yf
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# 1. ASSET CATALOG
# -------------------------------------------------------------------
ASSET_CATALOG = {
    "Commodities & Forex": {
        "Gold Spot (XAU/USD)": "GC=F",
        "Silver Spot (XAG/USD)": "SI=F",
        "Crude Oil (WTI)": "CL=F",
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "USDCAD=X",
        "GBP/JPY": "GBPJPY=X",
    },
    "Crypto": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Solana": "SOL-USD",
        "Binance Coin": "BNB-USD",
        "Ripple (XRP)": "XRP-USD",
        "Cardano": "ADA-USD",
        "Dogecoin": "DOGE-USD",
        "Avalanche": "AVAX-USD",
    },
    "Stocks & Indices": {
        "S&P 500 Index": "^GSPC",
        "Nasdaq 100": "^IXIC",
        "NVIDIA": "NVDA",
        "Apple": "AAPL",
        "Tesla": "TSLA",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
    },
}


# -------------------------------------------------------------------
# 2. FINBERT SENTIMENT ANALYSIS
# -------------------------------------------------------------------
@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert"
    )
    return tokenizer, model


def fetch_and_analyze_news(symbol: str) -> float:
    clean_sym = symbol.split("-")[0].replace("=X", "").replace("=F", "")
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={clean_sym}&region=US&lang=en-US"
    feed = feedparser.parse(rss_url)

    headlines = [entry.title for entry in feed.entries[:5]]
    if not headlines:
        return 0.0

    tokenizer, model = load_sentiment_model()
    inputs = tokenizer(
        headlines, padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    pos_score = torch.mean(probs[:, 0]).item()
    neg_score = torch.mean(probs[:, 1]).item()
    return pos_score - neg_score


# -------------------------------------------------------------------
# 3. TECHNICAL FEATURE ENGINE (SCALPING & DAY TRADING SUPPORT)
# -------------------------------------------------------------------
def build_feature_matrix(symbol: str, timeframe: str) -> pd.DataFrame:
    # Set data period based on selected timeframe
    if timeframe in ["1m", "5m"]:
        period = "7d"  # 1m/5m data max intraday period supported by yfinance
    elif timeframe == "15m":
        period = "1mo"
    else:
        period = "1y"

    df = yf.download(symbol, period=period, interval=timeframe, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Scalping Indicators
    df["Return_1"] = df["Close"].pct_change(1)
    df["Return_3"] = df["Close"].pct_change(3)

    # Fast Moving Averages for Scalping Momentum
    df["EMA_8"] = df["Close"].ewm(span=8, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_Diff"] = (df["EMA_8"] - df["EMA_21"]) / df["EMA_21"]

    # Volatility (ATR)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    return df


# -------------------------------------------------------------------
# 4. PREDICTIVE SCALPING & DIRECTION MODEL
# -------------------------------------------------------------------
def predict_asset_direction(
    df: pd.DataFrame, news_sentiment: float, is_scalping: bool
):
    data = df.copy()

    # Target timeframe shift: 2 candles forward for scalping, 3 candles for day trading
    shift_len = -2 if is_scalping else -3
    volatility = data["Return_1"].std()
    target_thresh = max(0.001 if is_scalping else 0.005, float(volatility * 0.4))

    future_returns = data["Close"].shift(shift_len) / data["Close"] - 1
    data["Target"] = (future_returns > target_thresh).astype(int)

    features = ["Return_1", "Return_3", "EMA_Diff", "RSI", "ATR"]
    X = data[features].iloc[:shift_len]
    y = data["Target"].iloc[:shift_len]

    if len(X) < 50 or len(np.unique(y)) < 2:
        return None, 0.50

    model = XGBClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.04, random_state=42
    )
    model.fit(X, y)

    latest_feats = data[features].iloc[-1:].copy()
    raw_prob = model.predict_proba(latest_feats)[0][1]

    # Adjust Sentiment Weight for Scalping (Scalping relies 85% on Technicals, 15% News)
    tech_weight = 0.85 if is_scalping else 0.70
    sent_weight = 0.15 if is_scalping else 0.30

    norm_sentiment = (news_sentiment + 1.0) / 2.0
    final_prob = (tech_weight * raw_prob) + (sent_weight * norm_sentiment)

    return model, final_prob


# -------------------------------------------------------------------
# 5. STREAMLIT FRONTEND
# -------------------------------------------------------------------
st.set_page_config(page_title="AI Scalping & Direction Engine", layout="wide")

st.title("⚡ AI Scalping & Direction Predictor")

# Sidebar Configuration
category = st.sidebar.selectbox(
    "Select Sector:", list(ASSET_CATALOG.keys())
)
selected_name = st.sidebar.selectbox(
    "Select Asset:", list(ASSET_CATALOG[category].keys())
)
symbol = ASSET_CATALOG[category][selected_name]

st.sidebar.divider()
st.sidebar.subheader("⚙️ Trading Strategy Mode")
trading_mode = st.sidebar.radio(
    "Select Strategy:", ["Scalping Mode (Fast)", "Day Trading Mode"]
)

if trading_mode == "Scalping Mode (Fast)":
    is_scalping = True
    timeframe = st.sidebar.selectbox("Scalp Timeframe:", ["1m", "5m", "15m"])
else:
    is_scalping = False
    timeframe = st.sidebar.selectbox("Trading Timeframe:", ["1h", "1d"])

if st.button(f"🚀 Execute AI Scan for {selected_name}"):
    with st.spinner(f"Fetching {timeframe} market data & training model..."):
        df = build_feature_matrix(symbol, timeframe)
        sentiment = fetch_and_analyze_news(symbol)
        model, confidence = predict_asset_direction(
            df, sentiment, is_scalping
        )

        current_price = float(df["Close"].iloc[-1])
        atr = float(df["ATR"].iloc[-1])

        # Scalping Risk Management Targets
        target_multiplier = 1.2 if is_scalping else 2.0
        stop_multiplier = 0.8 if is_scalping else 1.0

        if confidence >= 0.52:
            action = "BUY 🟢"
            direction = "BULLISH SCALP / TREND"
            target_price = current_price + (target_multiplier * atr)
            stop_loss = current_price - (stop_multiplier * atr)
        elif confidence <= 0.48:
            action = "SELL / SHORT 🔴"
            direction = "BEARISH SCALP / TREND"
            target_price = current_price - (target_multiplier * atr)
            stop_loss = current_price + (stop_multiplier * atr)
        else:
            action = "NEUTRAL 🟡"
            direction = "NO CLEAR SCALP SIGNAL"
            target_price = current_price
            stop_loss = current_price

        # Display Key Information
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Recommended Action", action)
        col2.metric("Signal Confidence", f"{round(confidence * 100, 1)}%")
        col3.metric("Timeframe", timeframe)
        col4.metric(
            "Current Price",
            (
                f"${round(current_price, 4)}"
                if "USD" in symbol or "=F" in symbol
                else f"{round(current_price, 2)}"
            ),
        )

        st.divider()

        st.subheader("🎯 Scalp / Trade Execution Specifications")
        exec_df = pd.DataFrame(
            [
                {
                    "Asset": selected_name,
                    "Strategy": trading_mode,
                    "Action": action,
                    "Entry Price": round(current_price, 4),
                    "Take Profit Target": round(target_price, 4),
                    "Stop Loss Target": round(stop_loss, 4),
                    "Est. Scalp Duration": (
                        "2 - 5 Candles"
                        if is_scalping
                        else "1 - 3 Days"
                    ),
                }
            ]
        )
        st.dataframe(exec_df, use_container_width=True)

        # Plot Scalping Chart (EMA 8 & EMA 21)
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df.index[-60:],
                open=df["Open"][-60:],
                high=df["High"][-60:],
                low=df["Low"][-60:],
                close=df["Close"][-60:],
                name="Price",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index[-60:],
                y=df["EMA_8"][-60:],
                name="EMA 8 (Fast)",
                line=dict(color="lightgreen", width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index[-60:],
                y=df["EMA_21"][-60:],
                name="EMA 21 (Slow)",
                line=dict(color="red", width=1.5),
            )
        )
        fig.update_layout(
            title=f"{selected_name} ({timeframe}) Chart",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)