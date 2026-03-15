################################################################################
###  IMPORTS + CONFIG
################################################################################
import os
import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objs as go
import smtplib

from email.mime.text import MIMEText

from ta.trend import macd, macd_signal
from ta.volatility import bollinger_hband, bollinger_lband, average_true_range
from ta.momentum import rsi

from supabase import create_client, Client
from datetime import datetime


################################################################################
###  SECRET LOADER
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
###  SUPABASE
################################################################################

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
TWELVE_DATA_KEY = get_secret("TWELVE_DATA_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not TWELVE_DATA_KEY:
    st.error("Missing config. Set SUPABASE_URL, SUPABASE_KEY, TWELVE_DATA_KEY.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

EMAIL_USER = get_secret("EMAIL_USER")
EMAIL_PASS = get_secret("EMAIL_PASS")


################################################################################
### STREAMLIT PAGE
################################################################################

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


################################################################################
### LOGGING
################################################################################

def log_activity(action, username):
    supabase.table("login_activity").insert({
        "username": username,
        "action": action,
        "timestamp": str(datetime.now())
    }).execute()


def audit_trail(admin, action, target_user=None):
    supabase.table("audit_trail").insert({
        "admin": admin,
        "action": action,
        "target_user": target_user,
        "timestamp": str(datetime.now())
    }).execute()


################################################################################
### AUTH
################################################################################

def login_user(username, password):

    data = supabase.table("users_app").select("*") \
        .eq("username", username) \
        .eq("password", password) \
        .execute()

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

    st.title("🔐 Login to Chumcred Forex Trading Platform")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        user = login_user(username, password)

        if user is None:
            st.error("Invalid username or password")

        elif user == "blocked":
            st.error("Your account is blocked")

        else:

            st.session_state.logged_in = True
            st.session_state.role = user["role"]
            st.session_state.username = user["username"]
            st.session_state.full_name = user["full_name"]

            log_activity("login", user["username"])

            st.rerun()


################################################################################
### EMAIL ALERT
################################################################################

def send_trade_email(signal, pair, price, confidence, sl, tp):

    if not EMAIL_USER or not EMAIL_PASS:
        return

    subject = f"Forex Signal: {signal} {pair}"

    body = f"""
Forex Trade Signal

Pair: {pair}
Signal: {signal}

Entry Price: {price}

Stop Loss: {sl}
Take Profit: {tp}

Confidence: {confidence}

Generated by Chumcred AI Forex Platform
"""

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

    except Exception as e:
        print("Email failed:", e)


################################################################################
### FOREX DASHBOARD
################################################################################

def forex_dashboard():

    st.title("Forex Trading Indicators Platform")

    MAJOR_PAIRS = [
        'EUR/USD','GBP/USD','USD/JPY','AUD/USD','USD/CAD','USD/CHF','NZD/USD',
        'EUR/JPY','GBP/JPY','EUR/GBP','XAU/USD'
    ]

    PAIR_SYMBOL_MAP = {p:p for p in MAJOR_PAIRS}

    st.sidebar.title("Scanner")

    main_pair = st.sidebar.selectbox("Pair", MAJOR_PAIRS)

    report_days = st.sidebar.slider("History", 5, 30, 10)

    ############################################################
    ### FETCH DATA
    ############################################################

    def fetch_data(pair):

        url = f"https://api.twelvedata.com/time_series?symbol={pair}&interval=1h&outputsize=500&apikey={TWELVE_DATA_KEY}"

        r = requests.get(url)

        j = r.json()

        if "values" not in j:
            return None

        df = pd.DataFrame(j["values"])

        df = df.rename(columns={
            "datetime":"Date",
            "open":"Open",
            "high":"High",
            "low":"Low",
            "close":"Close"
        })

        df["Date"] = pd.to_datetime(df["Date"])

        for c in ["Open","High","Low","Close"]:
            df[c] = pd.to_numeric(df[c])

        df = df.sort_values("Date")

        return df


    df = fetch_data(main_pair)

    if df is None:
        st.error("Could not fetch data")
        return


    ############################################################
    ### INDICATORS
    ############################################################

    df["rsi"] = rsi(df["Close"], 14)

    df["macd"] = macd(df["Close"])
    df["macd_signal"] = macd_signal(df["Close"])

    df["bb_high"] = bollinger_hband(df["Close"])
    df["bb_low"] = bollinger_lband(df["Close"])

    df["ma200"] = df["Close"].rolling(200).mean()

    df["atr"] = average_true_range(df["High"], df["Low"], df["Close"])


    ############################################################
    ### SIGNAL ENGINE
    ############################################################

    last = df.iloc[-1]

    buy_score = 0
    sell_score = 0

    if last["Close"] > last["ma200"]:
        buy_score += 1

    if last["Close"] < last["ma200"]:
        sell_score += 1

    if last["macd"] > last["macd_signal"]:
        buy_score += 1

    if last["macd"] < last["macd_signal"]:
        sell_score += 1

    if last["rsi"] > 55:
        buy_score += 1

    if last["rsi"] < 45:
        sell_score += 1

    if last["Close"] > last["bb_high"]:
        buy_score += 1

    if last["Close"] < last["bb_low"]:
        sell_score += 1


    signal = "NO TRADE"

    if buy_score >= 3:
        signal = "BUY"

    if sell_score >= 3:
        signal = "SELL"


    confidence = max(buy_score, sell_score) / 5


    ############################################################
    ### TRADE PLAN
    ############################################################

    entry = last["Close"]

    atr = last["atr"]

    sl = None
    tp = None

    if signal == "BUY":

        sl = entry - 1.5 * atr
        tp = entry + 3 * atr

    if signal == "SELL":

        sl = entry + 1.5 * atr
        tp = entry - 3 * atr


    ############################################################
    ### EMAIL ALERT
    ############################################################

    if "last_signal" not in st.session_state:
        st.session_state.last_signal = None

    sig_id = f"{main_pair}-{signal}"

    if signal in ["BUY","SELL"] and sig_id != st.session_state.last_signal:

        send_trade_email(signal, main_pair, entry, f"{confidence:.1%}", sl, tp)

        st.session_state.last_signal = sig_id


    ############################################################
    ### DISPLAY
    ############################################################

    color = {"BUY":"green","SELL":"red","NO TRADE":"gray"}

    st.markdown(
        f"""
        <div style='background:{color[signal]};
        padding:20px;border-radius:10px'>
        <h1 style='color:white'>{signal}</h1>
        <h3 style='color:white'>Confidence {confidence:.1%}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.write("Entry:",entry)

    if sl:
        st.write("Stop Loss:",sl)

    if tp:
        st.write("Take Profit:",tp)


    ############################################################
    ### CHART
    ############################################################

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["bb_high"],
        name="BB High"
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["bb_low"],
        name="BB Low"
    ))

    st.plotly_chart(fig,use_container_width=True)


################################################################################
### ADMIN
################################################################################

def admin_home():

    st.sidebar.header(f"Admin — {st.session_state.full_name}")

    choice = st.sidebar.radio("Menu",[
        "Dashboard",
        "Logout"
    ])

    if choice == "Dashboard":
        forex_dashboard()

    if choice == "Logout":
        logout()


################################################################################
### USER
################################################################################

def user_home():

    st.sidebar.header(f"Welcome {st.session_state.full_name}")

    choice = st.sidebar.radio("Menu",[
        "Dashboard",
        "Logout"
    ])

    if choice == "Dashboard":
        forex_dashboard()

    if choice == "Logout":
        logout()


################################################################################
### ROUTER
################################################################################

if not st.session_state.logged_in:

    login_page()

else:

    if st.session_state.role == "admin":

        admin_home()

    else:

        user_home()


st.write("Support: chumcred@gmail.com")
st.write("Powered by Chumcred Limited")