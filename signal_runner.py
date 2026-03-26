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

    if not EMAIL_USER or not EMAIL_PASS:
        return

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

    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=300&apikey={TWELVE_DATA_KEY}"
        r = requests.get(url, timeout=20)
        data = r.json()
    except:
        return None

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
# LEARNING ENGINE
# ================================
def update_trade_outcomes():

    try:
        trades = supabase.table("ai_trade_signals")\
            .select("*")\
            .eq("outcome_checked", False)\
            .execute().data
    except:
        return

    if not trades:
        return

    for trade in trades:

        pair = trade["pair"]
        entry = trade["entry_price"]
        signal = trade["signal"]

        df = fetch_data(pair, "1h")

        if df is None:
            continue

        current_price = df["Close"].iloc[-1]

        if signal == "BUY":
            result = current_price - entry
        else:
            result = entry - current_price

        outcome = "WIN" if result > 0 else "LOSS"

        supabase.table("ai_trade_signals")\
            .update({
                "outcome": outcome,
                "exit_price": float(current_price),
                "result_pips": float(result),
                "outcome_checked": True
            })\
            .eq("id", trade["id"])\
            .execute()

        print(f"📊 Updated: {pair} → {outcome}")

def get_pair_performance(pair):

    try:
        data = supabase.table("ai_trade_signals")\
            .select("outcome")\
            .eq("pair", pair)\
            .execute().data
    except:
        return 0.5

    if not data:
        return 0.5

    df = pd.DataFrame(data)

    wins = df[df["outcome"] == "WIN"].shape[0]
    losses = df[df["outcome"] == "LOSS"].shape[0]

    total = wins + losses

    if total == 0:
        return 0.5

    return wins / total

# ================================
# SIGNAL ENGINE (WITH LEARNING)
# ================================
def generate_signal(df_1h, df_4h, pair):

    last = df_1h.iloc[-1]

    buy_score = 0
    sell_score = 0

    if last["Close"] > last["MA200"]:
        buy_score += 2
    else:
        sell_score += 2

    if last["MACD"] > last["MACD_SIGNAL"]:
        buy_score += 1
    else:
        sell_score += 1

    if 50 <= last["RSI"] <= 70:
        buy_score += 1
    elif 30 <= last["RSI"] <= 50:
        sell_score += 1

    if last["Close"] > last["BB_HIGH"]:
        buy_score += 0.5
    elif last["Close"] < last["BB_LOW"]:
        sell_score += 0.5

    signal = "NO TRADE"

    if buy_score >= 3:
        signal = "BUY"
    elif sell_score >= 3:
        signal = "SELL"

    base_conf = 0.5 + max(buy_score, sell_score)/10

    # 🔥 LEARNING
    pair_perf = get_pair_performance(pair)

    confidence = (0.7 * base_conf) + (0.3 * pair_perf)

    if pair_perf < 0.45:
        confidence -= 0.15

    if pair_perf > 0.6:
        confidence += 0.1

    confidence = min(max(confidence, 0), 0.9)

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

    print("🚀 AI Runner with Learning Started...")

    while True:

        update_trade_outcomes()  # 🔥 LEARNING STEP

        for pair in PAIRS:

            try:
                df_1h = fetch_data(pair, "1h")
                df_4h = fetch_data(pair, "4h")

                if df_1h is None or df_4h is None:
                    continue

                df_1h = add_indicators(df_1h)
                df_4h = add_indicators(df_4h)

                signal, confidence, entry, sl, tp = generate_signal(df_1h, df_4h, pair)

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