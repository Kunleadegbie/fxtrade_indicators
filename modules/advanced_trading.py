import streamlit as st

# 🔐 AUTH GUARD
if not st.session_state.get("logged_in", False):
    st.warning("Please login from the main page.")
    st.stop()

import pandas as pd
import numpy as np
import requests
import os
import smtplib
from email.mime.text import MIMEText

from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator

from shared_logic import unified_decision


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
    # FETCH DATA
    # ================================
    def fetch_data(symbol="EUR/USD", interval="1h", outputsize=200):

        api_key = os.getenv("TWELVE_DATA_KEY")

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
            "close": "Close"
        })

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna()

    # ================================
    # SIGNAL ENGINE (UNCHANGED)
    # ================================
    def generate_signal(df):

        signal = "NO TRADE"
        confidence = 0

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

        return signal, confidence

    # ================================
    # USD CONVERSION ENGINE (NEW)
    # ================================
    def get_quote_to_usd_rate(quote_ccy):

        if quote_ccy == "USD":
            return 1.0

        direct = fetch_data(f"{quote_ccy}/USD", "1h", 5)
        if direct is not None:
            return float(direct["Close"].iloc[-1])

        inverse = fetch_data(f"USD/{quote_ccy}", "1h", 5)
        if inverse is not None:
            return 1 / float(inverse["Close"].iloc[-1])

        return None

    # ================================
    # POSITION SIZE (FULLY UPGRADED)
    # ================================
    def calculate_position_size(entry, sl, risk_amount, pair):

        if entry is None or sl is None:
            return 0, 0, 0

        base, quote = pair.split("/")

        if "JPY" in pair:
            pip_unit = 0.01
        elif "XAU" in pair:
            pip_unit = 0.01
        else:
            pip_unit = 0.0001

        sl_pips = abs(entry - sl) / pip_unit

        if quote == "USD":
            pip_value = 10
        elif base == "USD":
            pip_value = (pip_unit / entry) * 100000
        else:
            rate = get_quote_to_usd_rate(quote)
            if rate is None:
                return 0, sl_pips, 0
            pip_value = pip_unit * 100000 * rate

        pip_value_micro = pip_value / 100

        lot_size = risk_amount / (sl_pips * pip_value_micro)

        return round(lot_size, 2), round(sl_pips, 1), round(pip_value_micro, 4)

    # ================================
    # UI
    # ================================
    st.title("🚀 Advanced Trading System")

    PAIRS = [
        'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF',
        'EUR/JPY', 'GBP/JPY', 'EUR/GBP', 'XAU/USD'
    ]

    pair = st.selectbox("Select Pair", PAIRS)

    st.subheader("⚙️ Risk Management")

    account = st.number_input("Account ($)", value=300.0)
    risk_percent = st.slider("Risk %", 0.5, 5.0, 1.0)

    risk_amount = account * (risk_percent / 100)

    # ================================
    # BEST TRADE SCANNER (NEW)
    # ================================
    st.subheader("🔥 Best Trade Scanner")

    best_pair = None
    best_conf = 0

    for p in PAIRS:
        df_scan = fetch_data(p)
        if df_scan is None:
            continue

        sig, conf = generate_signal(df_scan)

        if sig in ["BUY", "SELL"] and conf > best_conf:
            best_conf = conf
            best_pair = p

    if best_pair:
        st.success(f"Best Pair: {best_pair} ({best_conf:.2%})")
    else:
        st.warning("No strong trade found")

    # ================================
    # MAIN SELECTED PAIR LOGIC
    # ================================
    df = fetch_data(pair)

    if df is None:
        st.stop()

    signal, confidence = generate_signal(df)

    st.write("Signal:", signal)
    st.write("Confidence:", f"{confidence:.2%}")

    if signal in ["BUY", "SELL"]:

        entry = df["Close"].iloc[-1]
        atr = df["Close"].rolling(14).std().iloc[-1]

        if signal == "BUY":
            sl = entry - (1.5 * atr)
            tp = entry + (3 * atr)
        else:
            sl = entry + (1.5 * atr)
            tp = entry - (3 * atr)

        lot, sl_pips, pip_val = calculate_position_size(entry, sl, risk_amount, pair)

        tp_pips = abs(entry - tp) / (0.01 if "JPY" in pair or "XAU" in pair else 0.0001)
        rr = round(tp_pips / sl_pips, 2)

        st.success(f"""
EXECUTE {signal} {pair}

Entry: {entry}
SL: {sl} ({sl_pips} pips)
TP: {tp} ({tp_pips} pips)

Lot Size: {lot}
Risk: ${risk_amount}
RR: 1:{rr}
Pip Value (0.01 lot): ${pip_val}
""")