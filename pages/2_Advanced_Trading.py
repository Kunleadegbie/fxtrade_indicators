import streamlit as st
import pandas as pd
import numpy as np
import requests

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from shared_logic import unified_decision

st.set_page_config(page_title="Advanced Trading System", layout="wide")

# ================================
# FETCH DATA
# ================================
def fetch_data(symbol="EUR/USD", interval="1h", outputsize=200):


    import os

    api_key = os.getenv("TWELVE_DATA_KEY")

    if not api_key:
        st.error("Missing TWELVE_DATA_KEY in environment variables")
        st.stop()
    

    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}"
    r = requests.get(url)

    if r.status_code != 200:
        return None

    data = r.json()

    if "values" not in data:
        return None

    df = pd.DataFrame(data["values"])

    df = df.rename(columns={
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")


    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle Volume safely (FOREX FIX)
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    else:
        df["Volume"] = 0


    return df.dropna()


# ================================
# CORE SIGNAL ENGINE
# ================================
def generate_signal(df):

    signal = "NO TRADE"
    confidence = 0
    reason = ""

    close = df["Close"]

    # RSI
    rsi = RSIIndicator(close).rsi().iloc[-1]

    # MACD
    macd = MACD(close)
    macd_val = macd.macd().iloc[-1]
    macd_sig = macd.macd_signal().iloc[-1]

    # EMA Trend
    ema50 = EMAIndicator(close, window=50).ema_indicator().iloc[-1]

    price = close.iloc[-1]

    buy_score = 0
    sell_score = 0

    # RSI
    if rsi < 30:
        buy_score += 1
    elif rsi > 70:
        sell_score += 1

    # MACD
    if macd_val > macd_sig:
        buy_score += 1
    else:
        sell_score += 1

    # EMA trend
    if price > ema50:
        buy_score += 1
    else:
        sell_score += 1

    # Decision
    if buy_score >= 2:
        signal = "BUY"
        confidence = buy_score / 3
    elif sell_score >= 2:
        signal = "SELL"
        confidence = sell_score / 3

    reason = f"RSI={rsi:.1f}, MACD={'BUY' if macd_val>macd_sig else 'SELL'}, EMA Trend={'UP' if price>ema50 else 'DOWN'}"

    return signal, confidence, reason


# ================================
# SNIPER ENTRY
# ================================
def sniper_entry(df, direction):

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if direction == "BUY":
        if last["Close"] < prev["Close"]:
            return "WAIT - Pullback"
        if last["Close"] > prev["High"]:
            return "ENTER BUY"
        return "WAIT"

    if direction == "SELL":
        if last["Close"] > prev["Close"]:
            return "WAIT - Pullback"
        if last["Close"] < prev["Low"]:
            return "ENTER SELL"
        return "WAIT"

    return "WAIT"


# ================================
# UNIFIED DECISION (LOCAL ONLY)
# ================================
def unified_decision(kpi_decision, trading_signal, entry_signal):

    def normalize(text):
        if text is None:
            return "NEUTRAL"
        if "BUY" in text:
            return "BUY"
        elif "SELL" in text:
            return "SELL"
        return "NEUTRAL"

    kpi = normalize(kpi_decision)
    trade = normalize(trading_signal)

    # FULL ALIGNMENT REQUIRED
    if kpi == "BUY" and trade == "BUY" and "ENTER BUY" in entry_signal:
        return "EXECUTE BUY"

    if kpi == "SELL" and trade == "SELL" and "ENTER SELL" in entry_signal:
        return "EXECUTE SELL"

    return "NO TRADE"

# ================================
# RISK MANAGEMENT
# ================================
def risk_management(entry, df, signal):

    atr = df["Close"].rolling(14).std().iloc[-1]

    if signal == "BUY":
        sl = entry - (1.5 * atr)
        tp = entry + (3 * atr)

    elif signal == "SELL":
        sl = entry + (1.5 * atr)
        tp = entry - (3 * atr)

    else:
        return None, None

    return round(sl, 5), round(tp, 5)


# ================================
# UI STARTS HERE
# ================================
st.title("🚀 Advanced Trading System")

pair = st.selectbox("Select Pair", ["EUR/USD", "GBP/USD", "USD/JPY"])

df = fetch_data(pair, "1h")

if df is None:
    st.error("Failed to load data")
    st.stop()

# ================================
# SIGNAL
# ================================
signal, confidence, reason = generate_signal(df)

st.subheader("📡 Signal Engine")
st.write(f"Signal: {signal}")
st.write(f"Confidence: {confidence:.2%}")
st.write(f"Reason: {reason}")

# ================================
# SNIPER ENTRY
# ================================
entry_signal = sniper_entry(df, signal)

st.subheader("🎯 Entry Timing")
st.write(entry_signal)

# ================================
# FINAL DECISION
# ================================
st.subheader("🧠 Unified Decision Engine")

kpi_decision = st.session_state.get("kpi_decision", "NEUTRAL")

final_decision = unified_decision(kpi_decision, signal, entry_signal)

st.write(f"KPI Direction: {kpi_decision}")
st.write(f"Trading Signal: {signal}")
st.write(f"Entry Timing: {entry_signal}")

if "BUY" in final_decision:
    st.success(final_decision)
elif "SELL" in final_decision:
    st.error(final_decision)
else:
    st.warning(final_decision)

# ================================
# RISK MANAGEMENT
# ================================
if "EXECUTE" in final_decision:

    entry_price = df["Close"].iloc[-1]
    trade_signal = "BUY" if "BUY" in final_decision else "SELL"

    sl, tp = risk_management(entry_price, df, trade_signal)

    st.subheader("💰 Risk Management")
    st.write(f"Entry: {entry_price:.5f}")
    st.write(f"Stop Loss: {sl}")
    st.write(f"Take Profit: {tp}")