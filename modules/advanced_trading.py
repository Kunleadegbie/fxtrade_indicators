import streamlit as st

# 🔐 AUTH GUARD
if not st.session_state.get("logged_in", False):
    st.warning("Please login from the main page.")
    st.stop()

import pandas as pd
import os
import smtplib
from email.mime.text import MIMEText

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator

from shared_logic import unified_decision
from modules.market_data import fetch_market_data, get_quote_to_usd_rate


def run():

    st.set_page_config(page_title="Advanced Trading System", layout="wide")

    # ================================
    # EMAIL FUNCTION
    # ================================
    def send_email_alert(pair, decision, entry, sl, tp):

        sender = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASS")

        if not sender or not password:
            return

        subject = "🚨 Trade Alert - Chumcred System"

        body = f"""
Trade Signal Triggered

Pair: {pair}
Decision: {decision}

Entry: {entry}
Stop Loss: {sl}
Take Profit: {tp}
"""

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = sender

        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, sender, msg.as_string())
            server.quit()
        except Exception as e:
            print("Email error:", e)

    # ================================
    # SIGNAL ENGINE
    # ================================
    def generate_signal(df):

        signal = "NO TRADE"
        confidence = 0
        reason = "No clear setup"

        close = df["Close"]

        rsi = RSIIndicator(close).rsi().iloc[-1]
        macd = MACD(close)
        ema50 = EMAIndicator(close, window=50).ema_indicator().iloc[-1]

        price = close.iloc[-1]

        buy_score = 0
        sell_score = 0

        if rsi < 30:
            buy_score += 1
        elif rsi > 70:
            sell_score += 1

        if macd.macd().iloc[-1] > macd.macd_signal().iloc[-1]:
            buy_score += 1
        else:
            sell_score += 1

        if price > ema50:
            buy_score += 1
        else:
            sell_score += 1

        if buy_score >= 2:
            signal = "BUY"
            confidence = buy_score / 3
        elif sell_score >= 2:
            signal = "SELL"
            confidence = sell_score / 3

        reason = (
            f"RSI={rsi:.1f}, "
            f"MACD={'BUY' if macd.macd().iloc[-1] > macd.macd_signal().iloc[-1] else 'SELL'}, "
            f"EMA={'UP' if price > ema50 else 'DOWN'}"
        )

        return signal, confidence, reason

    # ================================
    # ENTRY TIMING
    # ================================
    def sniper_entry(df, direction):

        if df is None or len(df) < 2:
            return "WAIT"

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
    # ATR HELPER
    # ================================
    def get_atr_like(df, period=14):
        atr = df["Close"].rolling(period).std().iloc[-1]
        if pd.isna(atr) or atr <= 0:
            return None
        return float(atr)

    # ================================
    # STRUCTURE DETECTION
    # ================================
    def detect_structure_levels(df, lookback=20):

        if df is None or len(df) < lookback:
            return None, None

        recent = df.tail(lookback)
        support = float(recent["Low"].min())
        resistance = float(recent["High"].max())

        return support, resistance

    # ================================
    # STRUCTURE-BASED RISK MANAGEMENT
    # ================================
    def risk_management(entry, df, signal, pair):

        atr = get_atr_like(df, 14)
        if atr is None:
            return None, None, None, None

        support, resistance = detect_structure_levels(df, lookback=20)
        if support is None or resistance is None:
            return None, None, None, None

        buffer_size = 0.3 * atr

        if "XAU" in pair or "JPY" in pair:
            pip_unit = 0.01
            min_sl_pips = 20
        else:
            pip_unit = 0.0001
            min_sl_pips = 15

        min_sl_distance = min_sl_pips * pip_unit

        if signal == "BUY":
            structural_sl = support - buffer_size
            sl = min(structural_sl, entry - min_sl_distance)

            stop_distance = entry - sl
            if stop_distance <= 0:
                return None, None, None, None

            tp = entry + (2 * stop_distance)

        elif signal == "SELL":
            structural_sl = resistance + buffer_size
            sl = max(structural_sl, entry + min_sl_distance)

            stop_distance = sl - entry
            if stop_distance <= 0:
                return None, None, None, None

            tp = entry - (2 * stop_distance)

        else:
            return None, None, None, None

        return round(sl, 5), round(tp, 5), support, resistance

    # ================================
    # DYNAMIC POSITION SIZE CALCULATOR
    # ================================
    def calculate_position_size(entry, sl, risk_amount, pair):

        if entry is None or sl is None or risk_amount <= 0:
            return 0, 0, 0

        price_diff = abs(entry - sl)

        if "XAU" in pair:
            pip_unit = 0.01
        elif "JPY" in pair:
            pip_unit = 0.01
        else:
            pip_unit = 0.0001

        sl_pips = price_diff / pip_unit

        if sl_pips <= 0:
            return 0, 0, 0

        base_ccy, quote_ccy = pair.split("/")

        if pair == "XAU/USD":
            pip_value_per_standard_lot = 1.0
        elif quote_ccy == "USD":
            pip_value_per_standard_lot = 10.0
        elif base_ccy == "USD":
            pip_value_per_standard_lot = (pip_unit / entry) * 100000
        else:
            pip_value_in_quote = pip_unit * 100000
            usd_rate = get_quote_to_usd_rate(quote_ccy)

            if usd_rate is None or usd_rate == 0:
                return 0, round(sl_pips, 1), 0

            pip_value_per_standard_lot = pip_value_in_quote * usd_rate

        pip_value_per_micro = pip_value_per_standard_lot / 100
        lot_size = risk_amount / (sl_pips * pip_value_per_micro)

        return round(lot_size, 2), round(sl_pips, 1), round(pip_value_per_micro, 4)

    # ================================
    # BEST TRADE SCANNER
    # ================================
    def scan_best_trade(pairs, risk_amount):

        rows = []
        scan_pairs = pairs[:4]

        for p in scan_pairs:
            df_scan = fetch_market_data(p, "1h", 120)

            if df_scan is None or len(df_scan) < 60:
                continue

            sig, conf, rsn = generate_signal(df_scan)
            entry_sig = sniper_entry(df_scan, sig)

            if sig not in ["BUY", "SELL"]:
                continue

            if sig == "BUY" and "ENTER BUY" not in entry_sig:
                continue
            if sig == "SELL" and "ENTER SELL" not in entry_sig:
                continue

            entry = float(df_scan["Close"].iloc[-1])
            sl, tp, support, resistance = risk_management(entry, df_scan, sig, p)

            if sl is None or tp is None:
                continue

            lot_size, sl_pips, pip_val = calculate_position_size(entry, sl, risk_amount, p)

            if "XAU" in p or "JPY" in p:
                pip_unit = 0.01
            else:
                pip_unit = 0.0001

            tp_pips = abs(entry - tp) / pip_unit
            rr = round(tp_pips / sl_pips, 2) if sl_pips > 0 else 0

            rows.append({
                "Pair": p,
                "Signal": sig,
                "Confidence": conf,
                "Entry": entry,
                "SL": sl,
                "TP": tp,
                "Support": support,
                "Resistance": resistance,
                "SL Pips": sl_pips,
                "TP Pips": round(tp_pips, 1),
                "RR": rr,
                "Lot Size": lot_size,
                "Pip Value (0.01)": pip_val,
                "Reason": rsn
            })

        if not rows:
            return None, None

        df_rows = pd.DataFrame(rows)
        df_rows = df_rows.sort_values(by=["Confidence", "RR"], ascending=[False, False]).reset_index(drop=True)
        best_trade = df_rows.iloc[0].to_dict()

        return best_trade, df_rows

    # ================================
    # UI
    # ================================
    st.title("🚀 Advanced Trading System")

    PAIRS = [
        'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF',
        'EUR/JPY', 'GBP/JPY', 'EUR/GBP', 'EUR/AUD', 'EUR/CAD', 'EUR/NZD',
        'GBP/AUD', 'AUD/JPY', 'CAD/JPY', 'AUD/NZD', 'CHF/JPY', 'USD/SGD',
        'USD/HKD', 'XAU/USD'
    ]

    pair = st.selectbox("Select Pair", PAIRS)

    st.subheader("⚙️ Risk Management")

    account = st.number_input("Account ($)", value=300.0, min_value=1.0)
    risk_percent = st.slider("Risk %", 0.5, 5.0, 1.0)

    risk_amount = account * (risk_percent / 100)

    df = fetch_market_data(pair, "1h", 200)

    st.subheader("🔥 Best Trade Scanner")

    best_trade, scanner_df = scan_best_trade(PAIRS, risk_amount)

    if best_trade is not None:
        st.success(
            f"Best Pair: {best_trade['Pair']} | "
            f"{best_trade['Signal']} | "
            f"Confidence: {best_trade['Confidence']:.2%} | "
            f"RR: 1:{best_trade['RR']}"
        )

        with st.expander("View ranked scanner results"):
            display_df = scanner_df.copy()
            display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.2%}")
            st.dataframe(display_df, use_container_width=True)
    else:
        st.info("Scanner is active, but there is no strong validated trade setup across the scanned pairs right now.")

    if df is None:
        st.warning("Selected pair market feed is temporarily unavailable. Please refresh shortly.")
        return

    if len(df) < 60:
        st.info("Market feed loaded, but there are not enough candles yet for full analysis on this pair.")
        return

    signal, confidence, reason = generate_signal(df)

    st.subheader("📡 Signal Engine")
    st.write(f"Signal: {signal}")
    st.write(f"Confidence: {confidence:.2%}")
    st.write(f"Reason: {reason}")

    entry_signal = sniper_entry(df, signal)

    st.subheader("🎯 Entry Timing")
    st.write(entry_signal)

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
        st.info("Market data loaded successfully, but there is no valid executable trade setup at the moment.")

    if "EXECUTE" in final_decision:

        entry = float(df["Close"].iloc[-1])
        trade_signal = "BUY" if "BUY" in final_decision else "SELL"

        sl, tp, support, resistance = risk_management(entry, df, trade_signal, pair)

        if sl is None or tp is None:
            st.warning("Risk engine could not calculate a valid stop loss / take profit for this setup.")
            return

        lot_size, sl_pips, pip_val = calculate_position_size(entry, sl, risk_amount, pair)

        if "XAU" in pair or "JPY" in pair:
            pip_unit = 0.01
        else:
            pip_unit = 0.0001

        tp_pips = abs(entry - tp) / pip_unit if tp else 0
        rr = round(tp_pips / sl_pips, 2) if sl_pips > 0 else 0

        st.subheader("💰 Risk Management")
        st.write(f"Entry: {entry:.5f}")
        st.write(f"Stop Loss: {sl}")
        st.write(f"Take Profit: {tp}")
        st.write(f"Support: {round(support, 5) if support is not None else 'N/A'}")
        st.write(f"Resistance: {round(resistance, 5) if resistance is not None else 'N/A'}")

        st.markdown("### 💰 Trade Risk Summary")

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.metric("Lot Size", lot_size)

        with r2:
            st.metric("Risk ($)", round(risk_amount, 2))

        with r3:
            st.metric("RR Ratio", f"1:{rr}")

        with r4:
            st.metric("Pip Value (0.01 lot)", f"${pip_val}")

        st.success(f"""
EXECUTE {trade_signal} {pair}

Entry: {entry:.5f}
Stop Loss: {sl} ({round(sl_pips, 1)} pips)
Take Profit: {tp} ({round(tp_pips, 1)} pips)

Lot Size: {lot_size}
Risk: ${round(risk_amount, 2)}
RR: 1:{rr}
Pip Value (0.01 lot): ${pip_val}
""")

        if "last_signal" not in st.session_state:
            st.session_state["last_signal"] = None

        signal_key = f"{pair}_{final_decision}_{round(entry, 5)}"

        if signal_key != st.session_state["last_signal"]:
            send_email_alert(pair, final_decision, entry, sl, tp)
            st.session_state["last_signal"] = signal_key