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
### DATA FETCH
################################################################################

@st.cache_data(ttl=1800)
def fetch_ohlc(symbol: str, interval: str = "1h", outputsize: int = 500):

    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVE_DATA_KEY}"
    )

    r = requests.get(url, timeout=30)
    j = r.json()

    if "values" not in j:
        return None

    df = pd.DataFrame(j["values"])

    df = df.rename(columns={
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    })

    df["Date"] = pd.to_datetime(df["Date"])

    for c in ["Open", "High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("Date").dropna()

    return df


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
### INSTITUTIONAL SUPPLY / DEMAND DETECTION
################################################################################

def detect_supply_demand(df: pd.DataFrame, lookback=60):

    recent = df.tail(lookback)

    demand = recent["Low"].min()
    supply = recent["High"].max()

    return demand, supply


def near_demand(price, demand, atr):

    if atr == 0 or demand is None:
        return False

    return abs(price - demand) <= atr * 0.5


def near_supply(price, supply, atr):

    if atr == 0 or supply is None:
        return False

    return abs(price - supply) <= atr * 0.5


################################################################################
### SIGNAL ENGINE
################################################################################

def generate_signal(df_1h, df_4h):

    last = df_1h.iloc[-1]

    trend_1h = "Uptrend" if last["Close"] > last["MA200"] else "Downtrend"

    last4 = df_4h.iloc[-1]
    trend_4h = "Uptrend" if last4["Close"] > last4["MA200"] else "Downtrend"

    macd_state = "Bullish" if last["MACD"] > last["MACD_SIGNAL"] else "Bearish"

    buy_score = 0
    sell_score = 0

    if trend_4h == "Uptrend":
        buy_score += 2
    else:
        sell_score += 2

    if trend_1h == "Uptrend":
        buy_score += 1
    else:
        sell_score += 1

    if macd_state == "Bullish":
        buy_score += 1
    else:
        sell_score += 1

    if 52 <= last["RSI"] <= 68:
        buy_score += 1
    elif 32 <= last["RSI"] <= 48:
        sell_score += 1

    if last["Close"] > last["BB_HIGH"]:
        buy_score += 0.75
    elif last["Close"] < last["BB_LOW"]:
        sell_score += 0.75

    signal = "NO TRADE"

    if buy_score >= 4:
        signal = "BUY"

    if sell_score >= 4:
        signal = "SELL"

    entry = float(last["Close"])
    atr = float(last["ATR"])

    demand, supply = detect_supply_demand(df_1h)

    reason = f"Trend4H={trend_4h}"

    if signal == "SELL" and near_demand(entry, demand, atr):
        signal = "NO TRADE"
        reason += " | blocked near demand"

    if signal == "BUY" and near_supply(entry, supply, atr):
        signal = "NO TRADE"
        reason += " | blocked near supply"

    sl = None
    tp = None
    rr = None

    if signal == "BUY":
        sl = entry - atr * 1.5
        tp = entry + atr * 3
        rr = 2

    if signal == "SELL":
        sl = entry + atr * 1.5
        tp = entry - atr * 3
        rr = 2

    confidence = min(0.50 + max(buy_score, sell_score) / 10, 0.80)

    return {
        "signal": signal,
        "confidence": confidence,
        "trend": trend_4h,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "reason": reason,
        "support": demand,
        "resistance": supply,
        "last_row": last
    }