import time
import os
import pandas as pd
import requests
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

from supabase import create_client, Client

# ================================
# CONFIG
# ================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY")

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================================
# PAIRS
# ================================
PAIRS = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD',
    'EUR/JPY', 'GBP/JPY', 'EUR/GBP', 'EUR/AUD', 'EUR/CAD', 'EUR/NZD',
    'GBP/AUD', 'AUD/JPY', 'CAD/JPY', 'AUD/NZD', 'CHF/JPY', 'USD/SGD',
    'USD/HKD', 'XAU/USD'
]

# ================================
# EMAIL
# ================================
def send_email(pair, decision, entry, sl, tp, confidence):

    subject = f"🚨 EXECUTE {decision} - {pair}"

    body = f"""
EXECUTION SIGNAL

Pair: {pair}
Decision: {decision}

Entry: {entry}
Stop Loss: {sl}
Take Profit: {tp}

Confidence: {confidence:.2%}
Time: {datetime.now()}
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_USER

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, EMAIL_USER, msg.as_string())
        server.quit()
    except Exception as e:
        print("Email error:", e)

# ================================
# FETCH DATA
# ================================
def fetch_data(symbol, interval="1h"):

    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=300&apikey={TWELVE_DATA_KEY}"
    r = requests.get(url)
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
# INDICATORS
# ================================
def add_indicators(df):

    df["MA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = df["Close"].rolling(14).mean()
    df["ATR"] = df["Close"].rolling(14).std()

    df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()

    df["BB_HIGH"] = df["Close"].rolling(20).mean() + 2 * df["Close"].rolling(20).std()
    df["BB_LOW"] = df["Close"].rolling(20).mean() - 2 * df["Close"].rolling(20).std()

    return df

# ================================
# SUPPORT / RESISTANCE
# ================================
def detect_zones(df):
    recent = df.tail(60)
    return recent["Low"].min(), recent["High"].max()

# ================================
# SIGNAL ENGINE (YOUR LOGIC)
# ================================
def generate_signal(df_1h, df_4h):

    last = df_1h.iloc[-1]

    buy_score = 0
    sell_score = 0

    # TREND
    if last["Close"] > last["MA200"]:
        buy_score += 2
    else:
        sell_score += 2

    # MACD
    if last["MACD"] > last["MACD_SIGNAL"]:
        buy_score += 1
    else:
        sell_score += 1

    # RSI
    if 50 <= last["RSI"] <= 70:
        buy_score += 1
    elif 30 <= last["RSI"] <= 50:
        sell_score += 1

    # BB
    if last["Close"] > last["BB_HIGH"]:
        buy_score += 0.5
    elif last["Close"] < last["BB_LOW"]:
        sell_score += 0.5

    signal = "NO TRADE"

    if buy_score >= 3:
        signal = "BUY"
    elif sell_score >= 3:
        signal = "SELL"

    confidence = min(0.5 + max(buy_score, sell_score)/10, 0.9)

    if confidence < 0.69:
        return "NO TRADE", confidence, None, None, None

    entry = last["Close"]
    atr = last["ATR"]

    demand, supply = detect_zones(df_1h)

    if signal == "BUY" and entry > supply:
        return "NO TRADE", confidence, None, None, None

    if signal == "SELL" and entry < demand:
        return "NO TRADE", confidence, None, None, None

    if signal == "BUY":
        sl = entry - (1.5 * atr)
        tp = entry + (3 * atr)
    else:
        sl = entry + (1.5 * atr)
        tp = entry - (3 * atr)

    return signal, confidence, entry, sl, tp

# ================================
# DEDUPLICATION
# ================================
last_sent = {}

# ================================
# MAIN LOOP
# ================================
def run():

    print("🚀 Runner Started...")

    while True:

        for pair in PAIRS:

            try:
                df_1h = fetch_data(pair, "1h")
                df_4h = fetch_data(pair, "4h")

                if df_1h is None or df_4h is None:
                    continue

                df_1h = add_indicators(df_1h)
                df_4h = add_indicators(df_4h)

                signal, confidence, entry, sl, tp = generate_signal(df_1h, df_4h)

                if signal not in ["BUY", "SELL"]:
                    continue

                key = f"{pair}_{signal}"

                if last_sent.get(pair) == key:
                    continue

                send_email(pair, signal, entry, sl, tp, confidence)

                supabase.table("ai_trade_signals").insert({
                    "pair": pair,
                    "signal": signal,
                    "confidence": float(confidence),
                    "entry_price": float(entry),
                    "stop_loss": float(sl),
                    "take_profit": float(tp),
                    "created_at": str(datetime.now())
                }).execute()

                last_sent[pair] = key

                print(f"✅ EXECUTE {signal} → {pair}")

            except Exception as e:
                print(f"Error {pair}: {e}")

        time.sleep(300)


if __name__ == "__main__":
    run()