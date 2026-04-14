import streamlit as st

# 🔐 AUTH GUARD
if not st.session_state.get("logged_in", False):
    st.warning("Please login from the main page.")
    st.stop()

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator

from modules.market_data import fetch_market_data


def run():

    st.set_page_config(page_title="KPI Consensus Engine", layout="wide")

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
            df = fetch_market_data(pair, tf, 200)
            if df is None or len(df) < 60:
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
        valid_count = len([v for v in results.values() if v != "NO DATA"])

        if valid_count == 0:
            return results, "NO DATA"

        if buy / valid_count >= 0.7:
            final = "STRONG BUY"
        elif sell / valid_count >= 0.7:
            final = "STRONG SELL"
        else:
            final = "NO TRADE"

        return results, final

    # ================================
    # UI
    # ================================
    st.title("📊 KPI Consensus Engine")

    PAIRS = [
        'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD',
        'EUR/JPY', 'GBP/JPY', 'EUR/GBP', 'EUR/AUD', 'EUR/CAD', 'EUR/NZD',
        'GBP/AUD', 'AUD/JPY', 'CAD/JPY', 'AUD/NZD', 'CHF/JPY', 'USD/SGD',
        'USD/HKD', 'XAU/USD'
    ]

    pair = st.selectbox("Select Pair", PAIRS)
    timeframe = st.selectbox("Timeframe", ["1h", "4h", "30min", "1day"], index=0)

    df = fetch_market_data(pair, timeframe, 200)

    fallback_used = False
    fallback_tf = "1h"

    if (df is None or len(df) < 60) and timeframe != fallback_tf:
        fallback_df = fetch_market_data(pair, fallback_tf, 200)
        if fallback_df is not None and len(fallback_df) >= 60:
            df = fallback_df
            fallback_used = True

    if fallback_used:
        st.info("Selected timeframe data was unavailable, so 1h data is being shown instead.")

    if df is None:
        st.warning("No live market data is available for this pair right now. Please try again shortly.")
        return

    if len(df) < 60:
        st.info("Market feed loaded, but there are not enough candles yet for KPI analysis on this timeframe.")
        return

    signals = get_kpi_signals(df)

    buy, sell, total, buy_ratio, sell_ratio, decision = consensus(signals)

    st.subheader("Signals")
    st.dataframe(pd.DataFrame(signals.items(), columns=["Indicator", "Signal"]))

    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("BUY", buy)
    c2.metric("SELL", sell)
    c3.metric("TOTAL", total)

    st.write(f"BUY Strength: {buy_ratio:.2%}")
    st.write(f"SELL Strength: {sell_ratio:.2%}")

    if "BUY" in decision:
        st.success(decision)
    elif "SELL" in decision:
        st.error(decision)
    else:
        st.info("Market data loaded successfully, but indicator consensus is not strong enough for a trade bias right now.")

    st.markdown("---")
    st.header("Multi-Timeframe Consensus")

    mtf_results, mtf_final = multi_timeframe_consensus(pair)

    st.dataframe(pd.DataFrame(mtf_results.items(), columns=["TF", "Signal"]))

    if mtf_final == "NO DATA":
        st.warning("Multi-timeframe market data is temporarily unavailable.")
    elif "BUY" in mtf_final:
        st.success(mtf_final)
    elif "SELL" in mtf_final:
        st.error(mtf_final)
    else:
        st.info("Multi-timeframe data is available, but there is no strong consensus bias at the moment.")

    st.session_state["kpi_decision"] = mtf_final