import streamlit as st

# 🔐 AUTH GUARD
if not st.session_state.get("logged_in", False):
    st.warning("Please login from the main page.")
    st.stop()

import pandas as pd
import requests
import os

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator


def run():

    st.set_page_config(page_title="KPI Consensus Engine", layout="wide")

    # ================================
    # FETCH DATA
    # ================================
    def fetch_data(symbol="EUR/USD", interval="1h", outputsize=200):

        api_key = os.getenv("TWELVE_DATA_KEY")

        if not api_key:
            st.error("Missing TWELVE_DATA_KEY")
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

        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce") if "Volume" in df.columns else 0

        return df.dropna()

    # ================================
    # SIGNALS
    # ================================
    def get_kpi_signals(df):

        signals = {}

        close = df["Close"]
        price = close.iloc[-1]

        rsi = RSIIndicator(close).rsi().iloc[-1]
        signals["RSI"] = "BUY" if rsi < 30 else "SELL" if rsi > 70 else "NEUTRAL"

        macd = MACD(close)
        signals["MACD"] = "BUY" if macd.macd().iloc[-1] > macd.macd_signal().iloc[-1] else "SELL"

        for p in [20, 50, 100, 200]:
            ema = EMAIndicator(close, window=p).ema_indicator().iloc[-1]
            sma = SMAIndicator(close, window=p).sma_indicator().iloc[-1]

            signals[f"EMA{p}"] = "BUY" if price > ema else "SELL"
            signals[f"SMA{p}"] = "BUY" if price > sma else "SELL"

        return signals

    def consensus(signals):

        buy = list(signals.values()).count("BUY")
        sell = list(signals.values()).count("SELL")
        total = len(signals)

        buy_ratio = buy / total
        sell_ratio = sell / total

        if buy_ratio >= 0.7:
            decision = "STRONG BUY"
        elif sell_ratio >= 0.7:
            decision = "STRONG SELL"
        else:
            decision = "NO TRADE"

        return buy, sell, total, buy_ratio, sell_ratio, decision

    def multi_timeframe_consensus(pair):

        tfs = ["30min", "1h", "4h", "1day"]
        results = {}

        for tf in tfs:
            df = fetch_data(pair, tf)
            if df is None:
                results[tf] = "NO DATA"
                continue

            signals = get_kpi_signals(df)
            _, _, _, buy_r, sell_r, decision = consensus(signals)

            if "BUY" in decision:
                results[tf] = "BUY"
            elif "SELL" in decision:
                results[tf] = "SELL"
            else:
                results[tf] = "NEUTRAL"

        buy = list(results.values()).count("BUY")
        sell = list(results.values()).count("SELL")

        if buy / len(results) >= 0.7:
            final = "STRONG BUY"
        elif sell / len(results) >= 0.7:
            final = "STRONG SELL"
        else:
            final = "NO TRADE"

        return results, final

    # ================================
    # UI
    # ================================
    st.title("📊 KPI Consensus Engine")
   
    pair = st.selectbox("Pair", [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD',
    'EUR/JPY', 'GBP/JPY', 'EUR/GBP', 'EUR/AUD', 'EUR/CAD', 'EUR/NZD',
    'GBP/AUD', 'AUD/JPY', 'CAD/JPY', 'AUD/NZD', 'CHF/JPY', 'USD/SGD',
    'USD/HKD', 'XAU/USD'
])
    

pair = st.selectbox("Select Pair", PAIRS)])
    timeframe = st.selectbox("Timeframe", ["30min", "1h", "4h", "1day"])

    df = fetch_data(pair, timeframe)

    if df is None:
        st.error("No data")
        st.stop()

    signals = get_kpi_signals(df)

    buy, sell, total, buy_ratio, sell_ratio, decision = consensus(signals)

    st.subheader("Signals")
    st.dataframe(pd.DataFrame(signals.items(), columns=["Indicator", "Signal"]))

    st.subheader("Summary")
    st.metric("BUY", buy)
    st.metric("SELL", sell)
    st.metric("TOTAL", total)

    st.write(f"BUY Strength: {buy_ratio:.2%}")
    st.write(f"SELL Strength: {sell_ratio:.2%}")

    if "BUY" in decision:
        st.success(decision)
    elif "SELL" in decision:
        st.error(decision)
    else:
        st.warning(decision)

    st.markdown("---")
    st.header("Multi-Timeframe Consensus")

    mtf_results, mtf_final = multi_timeframe_consensus(pair)

    st.dataframe(pd.DataFrame(mtf_results.items(), columns=["TF", "Signal"]))

    if "BUY" in mtf_final:
        st.success(mtf_final)
    elif "SELL" in mtf_final:
        st.error(mtf_final)
    else:
        st.warning(mtf_final)

    st.session_state["kpi_decision"] = mtf_final