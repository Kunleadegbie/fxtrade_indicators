################################################################################
### IMPORTS + CONFIG
################################################################################
import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
import smtplib

from email.mime.text import MIMEText
from datetime import datetime

from ta.trend import macd, macd_signal
from ta.volatility import bollinger_hband, bollinger_lband, average_true_range
from ta.momentum import rsi

from supabase import create_client, Client
from modules.market_data import fetch_market_data


################################################################################
### SECRET HANDLER
################################################################################
def get_secret(key: str, default: str = "") -> str:
    v = os.getenv(key)
    if v:
        return v
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


################################################################################
### CONFIG
################################################################################
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
TWELVE_DATA_KEY = get_secret("TWELVE_DATA_KEY")

EMAIL_USER = get_secret("EMAIL_USER")
EMAIL_PASS = get_secret("EMAIL_PASS")

if not SUPABASE_URL or not SUPABASE_KEY or not TWELVE_DATA_KEY:
    st.error("Missing config. Set SUPABASE_URL, SUPABASE_KEY, TWELVE_DATA_KEY.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="AI Forex Trading Platform", layout="wide")


################################################################################
### SESSION STATE
################################################################################
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "username" not in st.session_state:
    st.session_state.username = None

if "full_name" not in st.session_state:
    st.session_state.full_name = None

if "last_signals" not in st.session_state:
    st.session_state.last_signals = {}

if "last_email_alerts" not in st.session_state:
    st.session_state.last_email_alerts = {}


################################################################################
### LOGGING
################################################################################
def log_activity(action, username):
    try:
        supabase.table("login_activity").insert({
            "username": username,
            "action": action,
            "timestamp": str(datetime.now())
        }).execute()
    except Exception:
        pass


def audit_trail(admin, action, target_user=None):
    try:
        supabase.table("audit_trail").insert({
            "admin": admin,
            "action": action,
            "target_user": target_user,
            "timestamp": str(datetime.now())
        }).execute()
    except Exception:
        pass


################################################################################
### SELF-LEARNING / SIGNAL CAPTURE FOUNDATION
################################################################################
def persist_signal_record(record: dict):
    try:
        supabase.table("ai_trade_signals").insert(record).execute()
    except Exception:
        pass


################################################################################
### AUTHENTICATION
################################################################################
def login_user(username, password):
    data = supabase.table("users_app").select("*").eq("username", username).eq("password", password).execute()

    if len(data.data) == 0:
        return None

    user = data.data[0]

    if user["status"] == "blocked":
        return "blocked"

    return user


def logout():
    log_activity("logout", st.session_state.username)
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None
    st.session_state.full_name = None
    st.rerun()


################################################################################
### LOGIN PAGE
################################################################################
def login_page():
    st.title("🔐 Login to Chumcred Limited Forex Trading Indicator Platform")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user(username, password)

        if user is None:
            st.error("Invalid username or password.")
        elif user == "blocked":
            st.error("Your account has been blocked. Contact the admin.")
        else:
            st.session_state.logged_in = True
            st.session_state.role = user["role"]
            st.session_state.username = user["username"]
            st.session_state.full_name = user["full_name"]

            log_activity("login", user["username"])
            st.rerun()


################################################################################
### ADMIN FUNCTIONS
################################################################################
def create_new_user(full_name, username, password, role):
    exists = supabase.table("users_app").select("*").eq("username", username).execute()
    if len(exists.data) > 0:
        st.error("Username already exists.")
        return

    supabase.table("users_app").insert({
        "full_name": full_name,
        "username": username,
        "password": password,
        "role": role,
        "status": "active"
    }).execute()

    audit_trail(st.session_state.username, "created user", username)
    st.success("User created successfully!")


def update_user_status(username, new_status):
    supabase.table("users_app").update({"status": new_status}).eq("username", username).execute()
    audit_trail(st.session_state.username, f"{new_status} user", username)
    st.success(f"User '{username}' is now {new_status}.")


def reset_user_password(username, new_password):
    supabase.table("users_app").update({"password": new_password}).eq("username", username).execute()
    audit_trail(st.session_state.username, "reset password", username)
    st.success("Password reset successfully!")


def view_users():
    try:
        return supabase.table("users_app").select("*").execute().data
    except Exception:
        return []


def view_login_logs():
    try:
        return supabase.table("login_activity").select("*").execute().data
    except Exception:
        return []


def view_audit_trail():
    try:
        return supabase.table("audit_trail").select("*").execute().data
    except Exception:
        return []


################################################################################
### EMAIL ALERT (FIXED)
################################################################################
def send_trade_alert_email(pair, signal, price, confidence, sl, tp, rr, trend, entry_tf) -> bool:
    if not EMAIL_USER or not EMAIL_PASS:
        return False

    subject = f"Forex Alert: {signal} {pair}"

    body = f"""
Forex Trade Signal

Pair: {pair}
Signal: {signal}

Trend Filter: {trend}
Entry Timeframe: {entry_tf}

Entry Price: {price}
Stop Loss: {sl}
Take Profit: {tp}
Risk Reward: {rr}

Confidence: {confidence}

Generated by Chumcred AI Forex Platform
""".strip()

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = "chumcred@gmail.com"

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print("Email failed:", e)
        return False


################################################################################
### DATA FETCH
################################################################################
@st.cache_data(ttl=1800)
def fetch_ohlc(symbol: str, interval: str = "1h", outputsize: int = 500):
    return fetch_market_data(symbol, interval, outputsize)


################################################################################
### INDICATORS
################################################################################
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["RSI"] = rsi(df["Close"], 14)
    df["MACD"] = macd(df["Close"])
    df["MACD_SIGNAL"] = macd_signal(df["Close"])
    df["BB_HIGH"] = bollinger_hband(df["Close"])
    df["BB_LOW"] = bollinger_lband(df["Close"])
    df["MA200"] = df["Close"].rolling(200).mean()
    df["ATR"] = average_true_range(df["High"], df["Low"], df["Close"], 14)

    return df


################################################################################
### INSTITUTIONAL SUPPORT / RESISTANCE DETECTION
################################################################################
def detect_supply_demand(df: pd.DataFrame, lookback: int = 60):
    if df is None or len(df) < lookback:
        return None, None

    recent = df.tail(lookback)
    demand = recent["Low"].min()
    supply = recent["High"].max()

    return demand, supply


def near_demand(price, demand, atr):
    if demand is None or atr is None or atr == 0:
        return False
    return abs(price - demand) <= (0.5 * atr)


def near_supply(price, supply, atr):
    if supply is None or atr is None or atr == 0:
        return False
    return abs(price - supply) <= (0.5 * atr)


################################################################################
### SMART ENTRY + LIQUIDITY SWEEP LOGIC
################################################################################
def detect_liquidity_sweep(df):
    if len(df) < 5:
        return False, False

    recent = df.tail(5)

    high_break = recent["High"].iloc[-1] > recent["High"].iloc[:-1].max()
    close_back_down = recent["Close"].iloc[-1] < recent["High"].iloc[-2]

    low_break = recent["Low"].iloc[-1] < recent["Low"].iloc[:-1].min()
    close_back_up = recent["Close"].iloc[-1] > recent["Low"].iloc[-2]

    sweep_sell = high_break and close_back_down
    sweep_buy = low_break and close_back_up

    return sweep_buy, sweep_sell


def smart_entry_confirmation(df, signal):
    if len(df) < 3:
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if signal == "BUY":
        return last["Close"] > prev["High"]

    if signal == "SELL":
        return last["Close"] < prev["Low"]

    return False


################################################################################
### NEWS FILTER
################################################################################
def news_filter():
    now = datetime.utcnow().hour

    if 0 <= now <= 6:
        return True

    if 12 <= now <= 14:
        return True

    return False


################################################################################
### HELPERS
################################################################################
def classify_macd(last_row) -> str:
    if last_row["MACD"] > last_row["MACD_SIGNAL"]:
        return "Bullish"
    if last_row["MACD"] < last_row["MACD_SIGNAL"]:
        return "Bearish"
    return "Neutral"


def classify_trend(close_price: float, ma200: float) -> str:
    if pd.isna(ma200):
        return "Unknown"
    if close_price > ma200:
        return "Uptrend"
    if close_price < ma200:
        return "Downtrend"
    return "Sideways"


def higher_timeframe_trend(df_4h: pd.DataFrame) -> str:
    if df_4h is None or len(df_4h) < 220:
        return "Unknown"

    last = df_4h.iloc[-1]
    if pd.isna(last["MA200"]):
        return "Unknown"

    if last["Close"] > last["MA200"]:
        return "Uptrend"
    if last["Close"] < last["MA200"]:
        return "Downtrend"
    return "Sideways"


def get_pair_performance(pair):
    try:
        data = supabase.table("ai_trade_signals")\
            .select("pair, outcome")\
            .eq("pair", pair)\
            .execute().data
    except Exception:
        return 0.5

    if not data:
        return 0.5

    df = pd.DataFrame(data)

    wins = df[df["outcome"] == "WIN"].shape[0]
    losses = df[df["outcome"] == "LOSS"].shape[0]

    total = wins + losses

    if total == 0:
        return 0.5

    win_rate = wins / total

    return win_rate


################################################################################
### SIGNAL ENGINE
################################################################################
def generate_signal(df_1h, df_4h):
    signal = "NO TRADE"
    confidence = 0.0
    reason = "No clear setup"
    entry = None
    sl = None
    tp = None
    rr = None
    demand = None
    supply = None

    trend_4h = "Unknown"
    macd_state = "Neutral"
    last = None

    if df_1h is None or len(df_1h) < 220:
        return {
            "signal": signal,
            "confidence": confidence,
            "trend": trend_4h,
            "macd_state": macd_state,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "reason": reason,
            "support": demand,
            "resistance": supply,
            "last_row": last
        }

    last = df_1h.iloc[-1]

    trend_1h = classify_trend(last["Close"], last["MA200"])
    trend_4h = higher_timeframe_trend(df_4h)
    macd_state = classify_macd(last)

    buy_score = 0.0
    sell_score = 0.0

    if trend_4h == "Uptrend":
        buy_score += 2
    elif trend_4h == "Downtrend":
        sell_score += 2

    if trend_1h == "Uptrend":
        buy_score += 1
    elif trend_1h == "Downtrend":
        sell_score += 1

    if last["MACD"] > last["MACD_SIGNAL"]:
        buy_score += 1
    elif last["MACD"] < last["MACD_SIGNAL"]:
        sell_score += 1

    if 50 <= last["RSI"] <= 70:
        buy_score += 0.8
    elif 30 <= last["RSI"] <= 50:
        sell_score += 0.8

    if last["Close"] > last["BB_HIGH"]:
        buy_score += 0.5
    elif last["Close"] < last["BB_LOW"]:
        sell_score += 0.5

    if not pd.isna(last["ATR"]) and last["ATR"] > 0:
        buy_score += 0.2
        sell_score += 0.2

    raw_score = max(buy_score, sell_score)

    if buy_score >= 3 and buy_score > sell_score:
        signal = "BUY"
    elif sell_score >= 3 and sell_score > buy_score:
        signal = "SELL"

    pair = "UNKNOWN"
    if "Pair" in df_1h.columns:
        pair = df_1h["Pair"].iloc[-1]

    historical_win_rate = get_pair_performance(pair)

    base_conf = 0.5 + raw_score / 10
    confidence = (0.7 * base_conf) + (0.3 * historical_win_rate)
    confidence = min(confidence, 0.9)

    if signal in ["BUY", "SELL"]:
        if historical_win_rate < 0.45:
            confidence -= 0.15
            reason += " | Weak historical performance"
        elif historical_win_rate > 0.6:
            confidence += 0.1
            reason += " | Strong historical edge"

    if signal == "NO TRADE":
        confidence = min(raw_score / 10, 0.5)

    entry = float(last["Close"])
    atr = float(last["ATR"]) if not pd.isna(last["ATR"]) else 0

    demand, supply = detect_supply_demand(df_1h)

    if signal == "SELL" and near_demand(entry, demand, atr):
        signal = "NO TRADE"
        reason += " | Blocked: near demand"

    if signal == "BUY" and near_supply(entry, supply, atr):
        signal = "NO TRADE"
        reason += " | Blocked: near resistance"

    sweep_buy, sweep_sell = detect_liquidity_sweep(df_1h)

    if signal == "BUY" and not sweep_buy:
        confidence -= 0.1
        reason += " | Weak liquidity confirmation"

    if signal == "SELL" and not sweep_sell:
        confidence -= 0.1
        reason += " | Weak liquidity confirmation"

    if signal in ["BUY", "SELL"]:
        confirmed = smart_entry_confirmation(df_1h, signal)

        if not confirmed:
            confidence -= 0.1
            reason += " | Early entry"

    if signal in ["BUY", "SELL"] and news_filter():
        confidence -= 0.15
        reason += " | Risky session"

    if signal == "BUY" and atr > 0:
        sl = entry - (1.5 * atr)
        tp = entry + (3 * atr)
        rr = round((tp - entry) / (entry - sl), 2)

    elif signal == "SELL" and atr > 0:
        sl = entry + (1.5 * atr)
        tp = entry - (3 * atr)
        rr = round((entry - tp) / (sl - entry), 2)

    confidence = max(min(confidence, 0.9), 0)

    if confidence < 0.69:
        signal = "NO TRADE"
        reason += " | Blocked: confidence below 50%"

    reason = f"{reason} | 4H={trend_4h}, 1H={trend_1h}, MACD={macd_state}, RSI={last['RSI']:.1f}"

    return {
        "signal": signal,
        "confidence": confidence,
        "trend": trend_4h,
        "macd_state": macd_state,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "reason": reason,
        "support": demand,
        "resistance": supply,
        "last_row": last
    }


################################################################################
### BACKTEST ENGINE
################################################################################
def run_backtest(df_1h: pd.DataFrame, df_4h: pd.DataFrame):
    if df_1h is None or df_4h is None or len(df_1h) < 260 or len(df_4h) < 260:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0
        }

    wins = 0
    losses = 0

    for i in range(220, len(df_1h) - 6):
        sub_1h = df_1h.iloc[:i].copy()
        sub_4h = df_4h[df_4h["Date"] <= sub_1h.iloc[-1]["Date"]].copy()

        if len(sub_4h) < 220:
            continue

        sig_pkg = generate_signal(sub_1h, sub_4h)
        signal = sig_pkg["signal"]
        entry = sig_pkg["entry"]

        if signal not in ["BUY", "SELL"] or entry is None:
            continue

        future = df_1h.iloc[i + 6]["Close"]

        if signal == "BUY" and future > entry:
            wins += 1
        elif signal == "SELL" and future < entry:
            wins += 1
        else:
            losses += 1

    total = wins + losses
    win_rate = (wins / total) if total > 0 else 0.0

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate
    }


################################################################################
### REAL TRADE OUTCOME TRACKING
################################################################################
def update_trade_outcomes():
    try:
        data = supabase.table("ai_trade_signals")\
            .select("*")\
            .eq("outcome_checked", False)\
            .execute().data
    except Exception:
        return

    if not data:
        return

    for trade in data:
        pair = trade["pair"]
        entry = trade["entry_price"]
        signal = trade["signal"]

        if entry is None or signal not in ["BUY", "SELL"]:
            continue

        df = fetch_ohlc(pair, "1h", 50)

        if df is None or len(df) < 5:
            continue

        current_price = float(df.iloc[-1]["Close"])

        if signal == "BUY":
            result = current_price - entry
        else:
            result = entry - current_price

        outcome = "WIN" if result > 0 else "LOSS"

        supabase.table("ai_trade_signals")\
            .update({
                "outcome": outcome,
                "exit_price": current_price,
                "result_pips": result,
                "outcome_checked": True
            })\
            .eq("id", trade["id"])\
            .execute()


################################################################################
### PERFORMANCE DASHBOARD
################################################################################
def get_performance_stats():
    try:
        data = supabase.table("ai_trade_signals").select("*").execute().data
    except Exception:
        return None

    if not data or len(data) == 0:
        return None

    df = pd.DataFrame(data)
    df = df.dropna(subset=["signal", "entry_price"])

    total_trades = len(df)
    wins = df[df["outcome"] == "WIN"].shape[0]
    losses = df[df["outcome"] == "LOSS"].shape[0]
    win_rate = wins / total_trades if total_trades > 0 else 0
    avg_conf = df["confidence"].mean()
    best_pairs = df.groupby("pair")["confidence"].mean().sort_values(ascending=False).head(5)

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_confidence": avg_conf,
        "best_pairs": best_pairs,
        "raw": df.tail(20)
    }


##############################################################################
### FOREX DASHBOARD
##############################################################################
update_trade_outcomes()

def forex_dashboard():
    st.title("AI Forex Trade Indicator & Market Scanner")
    st.write("Institutional confirmation: 4H trend + 1H entry, ATR trade plan, multi-pair scanner, ranking, backtest.")

    PAIRS = [
        'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD',
        'EUR/JPY', 'GBP/JPY', 'EUR/GBP', 'EUR/AUD', 'EUR/CAD', 'EUR/NZD',
        'GBP/AUD', 'AUD/JPY', 'CAD/JPY', 'AUD/NZD', 'CHF/JPY', 'USD/SGD',
        'USD/HKD', 'XAU/USD'
    ]

    st.sidebar.title("Forex Scanner - Real Data")
    selected_pair = st.sidebar.selectbox("Deep Analysis Pair", PAIRS, index=0)
    show_rawdata = st.sidebar.checkbox("Show Raw Data", False)

    scanner_rows = []
    active_signals = []

    for pair in PAIRS:
        df_1h = fetch_ohlc(pair, "1h", 500)
        df_4h = fetch_ohlc(pair, "4h", 500)

        if df_1h is None or df_4h is None:
            continue

        if len(df_1h) < 220 or len(df_4h) < 220:
            continue

        df_1h = compute_indicators(df_1h)
        df_4h = compute_indicators(df_4h)

        sig = generate_signal(df_1h, df_4h)
        last = sig["last_row"]

        if last is None:
            continue

        scanner_rows.append({
            "Pair": pair,
            "Price": round(float(last["Close"]), 5),
            "Trend": sig["trend"],
            "RSI": round(float(last["RSI"]), 1),
            "MACD": sig.get("macd_state", ""),
            "Signal": sig["signal"],
            "Confidence": f"{sig['confidence']:.1%}",
            "Entry": round(sig["entry"], 5) if sig["entry"] is not None else None,
            "Stop Loss": round(sig["sl"], 5) if sig["sl"] is not None else None,
            "Take Profit": round(sig["tp"], 5) if sig["tp"] is not None else None,
            "RR": sig["rr"],
            "Support": round(sig.get("support"), 5) if sig.get("support") is not None else None,
            "Resistance": round(sig.get("resistance"), 5) if sig.get("resistance") is not None else None
        })

        if sig["signal"] in ["BUY", "SELL"]:
            active_signals.append({
                "Pair": pair,
                "Signal": sig["signal"],
                "Confidence": sig["confidence"],
                "Entry": sig["entry"],
                "SL": sig["sl"],
                "TP": sig["tp"],
                "RR": sig["rr"],
                "Trend": sig["trend"],
                "Reason": sig["reason"]
            })

            signal_key = f"{pair}|{sig['signal']}"
            if st.session_state.last_email_alerts.get(pair) != signal_key:
                send_trade_alert_email(
                    pair=pair,
                    signal=sig["signal"],
                    price=round(sig["entry"], 5) if sig["entry"] is not None else None,
                    confidence=f"{sig['confidence']:.1%}",
                    sl=round(sig["sl"], 5) if sig["sl"] else None,
                    tp=round(sig["tp"], 5) if sig["tp"] else None,
                    rr=sig["rr"],
                    trend=sig["trend"],
                    entry_tf="1H"
                )
                st.session_state.last_email_alerts[pair] = signal_key

            persist_signal_record({
                "pair": pair,
                "signal": sig["signal"],
                "confidence": float(sig["confidence"]),
                "entry_price": float(sig["entry"]) if sig["entry"] is not None else None,
                "stop_loss": float(sig["sl"]) if sig["sl"] is not None else None,
                "take_profit": float(sig["tp"]) if sig["tp"] is not None else None,
                "trend": sig["trend"],
                "reason": sig["reason"],
                "captured_at": str(datetime.now())
            })

    scanner_df = pd.DataFrame(scanner_rows)

    st.subheader("Market Scanner")
    if len(scanner_df) > 0:
        st.dataframe(scanner_df, height=420, use_container_width=True)
    else:
        st.warning("No market data available at the moment.")
        return

    st.markdown("---")

    st.subheader("Signal Ranking")
    ranked = scanner_df[scanner_df["Signal"] != "NO TRADE"].copy()

    if len(ranked) > 0:
        ranked["ConfidenceSort"] = ranked["Confidence"].str.rstrip("%").astype(float)
        ranked = ranked.sort_values("ConfidenceSort", ascending=False).drop(columns=["ConfidenceSort"])
        st.table(ranked.head(10))
    else:
        st.info("No active BUY/SELL signals right now.")

    st.markdown("---")

    st.subheader("Multiple Live Signals")
    if len(active_signals) > 0:
        multi_df = pd.DataFrame(active_signals)
        multi_df["Confidence"] = multi_df["Confidence"].apply(lambda x: f"{x:.1%}")
        st.dataframe(multi_df, use_container_width=True)
    else:
        st.caption("No simultaneous actionable signals at this time.")

    st.markdown("---")

    df_1h = fetch_ohlc(selected_pair, "1h", 500)
    df_4h = fetch_ohlc(selected_pair, "4h", 500)

    if df_1h is None or df_4h is None or len(df_1h) < 220 or len(df_4h) < 220:
        st.error("Could not fetch selected pair data.")
        return

    df_1h = compute_indicators(df_1h)
    df_4h = compute_indicators(df_4h)

    sig = generate_signal(df_1h, df_4h)

    st.header(f"Deep Technical Analytics — {selected_pair}")

    color_map = {"BUY": "green", "SELL": "red", "NO TRADE": "gray"}
    st.markdown(
        f"""
        <div style="background-color:{color_map[sig['signal']]};padding:1rem;border-radius:1rem;">
            <h2 style="color:white;margin:0;">{sig['signal']}</h2>
            <p style="color:white;">Confidence: <b>{sig['confidence']:.1%}</b></p>
            <p style="color:white;">Trend: <b>{sig['trend']}</b></p>
            <p style="color:white;">Reason: <b>{sig['reason']}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Trade Setup")
    st.write(f"Entry Price: {sig['entry']:.5f}" if sig["entry"] is not None else "Entry Price: N/A")
    st.write(f"Stop Loss: {sig['sl']:.5f}" if sig["sl"] is not None else "Stop Loss: N/A")
    st.write(f"Take Profit: {sig['tp']:.5f}" if sig["tp"] is not None else "Take Profit: N/A")
    st.write(f"Risk Reward Ratio: 1:{sig['rr']}" if sig["rr"] is not None else "Risk Reward Ratio: N/A")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_1h["Date"],
        open=df_1h["Open"],
        high=df_1h["High"],
        low=df_1h["Low"],
        close=df_1h["Close"],
        name=selected_pair
    ))
    fig.add_trace(go.Scatter(x=df_1h["Date"], y=df_1h["BB_HIGH"], name="Bollinger High"))
    fig.add_trace(go.Scatter(x=df_1h["Date"], y=df_1h["BB_LOW"], name="Bollinger Low"))
    fig.add_trace(go.Scatter(x=df_1h["Date"], y=df_1h["MA200"], name="MA200"))
    st.plotly_chart(fig, use_container_width=True)

    st.write("### RSI & MACD")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_1h["Date"], y=df_1h["RSI"], name="RSI"))
    fig2.add_trace(go.Scatter(x=df_1h["Date"], y=df_1h["MACD"], name="MACD"))
    fig2.add_trace(go.Scatter(x=df_1h["Date"], y=df_1h["MACD_SIGNAL"], name="MACD Signal"))
    st.plotly_chart(fig2, use_container_width=True)

    if show_rawdata:
        st.dataframe(df_1h.tail(50), use_container_width=True)

    st.markdown("---")

    st.header("Strategy Backtest")
    bt = run_backtest(df_1h, df_4h)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", bt["trades"])
    c2.metric("Wins", bt["wins"])
    c3.metric("Losses", bt["losses"])
    c4.metric("Win Rate", f"{bt['win_rate']:.2%}")

    st.markdown("---")
    st.header("Institutional Indicators")
    st.write("""
- 4H trend filter using MA200
- 1H MACD confirmation
- 1H RSI momentum filter
- Bollinger breakout validation
- ATR-based stop-loss / take-profit
- Multi-pair scanner with simultaneous signal detection
""")

    st.markdown("---")
    st.header("📊 Performance Dashboard")

    perf = get_performance_stats()

    if perf is None:
        st.info("No performance data available yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Signals", perf["total_trades"])
        c2.metric("Win Rate", f"{perf['win_rate']:.2%}")
        c3.metric("Wins", perf["wins"])
        c4.metric("Losses", perf["losses"])

        st.write(f"Average Confidence: {perf['avg_confidence']:.2%}")

        st.subheader("Top Performing Pairs")
        st.table(perf["best_pairs"])

        st.subheader("Recent Signals")
        st.dataframe(perf["raw"], use_container_width=True)


################################################################################
### ADMIN DASHBOARD
################################################################################
def admin_home():
    st.sidebar.header(f"Admin Panel — {st.session_state.full_name}")

    choice = st.sidebar.radio("Menu:", [
        "Dashboard",
        "Advanced Trading",
        "KPI Consensus",
        "Create User",
        "Manage Users",
        "Reset Password",
        "Login Activity Logs",
        "Audit Trail",
        "Logout"
    ])

    if choice == "Dashboard":
        forex_dashboard()

    elif choice == "Advanced Trading":
        from modules import advanced_trading
        advanced_trading.run()

    elif choice == "KPI Consensus":
        from modules import kpi_consensus
        kpi_consensus.run()

    elif choice == "Create User":
        st.title("➕ Create New User")
        full_name = st.text_input("Full Name")
        username = st.text_input("Username")
        password = st.text_input("Password")
        role = st.selectbox("Role", ["admin", "user"])

        if st.button("Create User"):
            create_new_user(full_name, username, password, role)

    elif choice == "Manage Users":
        st.title("🛠 Manage Users")
        users = view_users()
        st.table(users)

        username_change = st.text_input("Username to modify:")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Block"):
                update_user_status(username_change, "blocked")
        with col2:
            if st.button("Unblock"):
                update_user_status(username_change, "active")

    elif choice == "Reset Password":
        st.title("🔑 Reset User Password")
        username = st.text_input("Username")
        new_pw = st.text_input("New Password")

        if st.button("Reset"):
            reset_user_password(username, new_pw)

    elif choice == "Login Activity Logs":
        st.title("📘 Login Activity Logs")
        logs = view_login_logs()
        st.table(logs)

    elif choice == "Audit Trail":
        st.title("📙 Admin Audit Trail")
        logs = view_audit_trail()
        st.table(logs)

    elif choice == "Logout":
        logout()


################################################################################
### USER DASHBOARD
################################################################################
def user_home():
    st.sidebar.header(f"Welcome {st.session_state.full_name}")

    page = st.sidebar.radio(
        "Navigation",
        ["Advanced Trading", "KPI Consensus", "Logout"]
    )

    if page == "Advanced Trading":
        from modules import advanced_trading
        advanced_trading.run()

    elif page == "KPI Consensus":
        from modules import kpi_consensus
        kpi_consensus.run()

    elif page == "Logout":
        logout()


################################################################################
### ROUTING CONTROL
################################################################################
if not st.session_state.logged_in:
    login_page()
else:
    if st.session_state.role == "admin":
        admin_home()
    else:
        user_home()

st.write("**Contact customer support via email: chumcred@gmail.com or ‪+2348025420200‬ to set your username and password. Terms and conditions apply.**")
st.write("**Powered by Chumcred Limited**")