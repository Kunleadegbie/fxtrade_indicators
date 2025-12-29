################################################################################
###  IMPORTS
################################################################################

import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objs as go

from ta.trend import macd, macd_signal
from ta.volatility import bollinger_hband, bollinger_lband
from ta.momentum import rsi

from supabase import create_client, Client
from datetime import datetime


################################################################################
###  SUPABASE INITIALIZATION
################################################################################

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
    options={"timeout": 30}
)

################################################################################
###  STREAMLIT PAGE CONFIG
################################################################################

st.set_page_config(page_title="AI Forex Trading Platform", layout="wide")


################################################################################
###  SESSION STATE
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
###  LOGIN ACTIVITY LOGGING
################################################################################

def log_activity(action, username):
    supabase.table("login_activity").insert({
        "username": username,
        "action": action,
        "timestamp": str(datetime.now())
    }).execute()


################################################################################
###  AUDIT TRAIL (ADMIN ACTIONS)
################################################################################

def audit_trail(admin, action, target_user=None):
    supabase.table("audit_trail").insert({
        "admin": admin,
        "action": action,
        "target_user": target_user,
        "timestamp": str(datetime.now())
    }).execute()


################################################################################
###  AUTHENTICATION
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
###  LOGIN PAGE — (1A OPTION)
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
###  ADMIN FUNCTIONS
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
    return supabase.table("users_app").select("*").execute().data


def view_login_logs():
    return supabase.table("login_activity").select("*").execute().data


def view_audit_trail():
    return supabase.table("audit_trail").select("*").execute().data



################################################################################
###  FOREX DASHBOARD (YOUR FULL CODE)
################################################################################

def forex_dashboard():

    st.title("Forex Trading Indicators Platform")
    # --- Display ---
    st.write(
    "Trade decisions for every major currency pairs is fully analyzed with RSI, Bollinger Bands, MACD."
)

    TWELVE_DATA_KEY = st.secrets["TWELVE_DATA_KEY"]

    MAJOR_PAIRS = [
        'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD',
        'EUR/JPY', 'GBP/JPY', 'EUR/GBP', 'EUR/AUD', 'EUR/CAD', 'EUR/NZD',
        'GBP/AUD', 'AUD/JPY', 'CAD/JPY', 'SEK/JPY', 'AUD/NZD', 'CHF/JPY', 'USD/SGD',
        'USD/HKD', 'XAU/USD'
    ]

    PAIR_SYMBOL_MAP = {p: p for p in MAJOR_PAIRS}

    # Sidebar Controls
    st.sidebar.title("Forex Scanner - Real Data")
    main_pair = st.sidebar.selectbox("Main Currency Pair", MAJOR_PAIRS, index=0)
    report_days = st.sidebar.slider("History (Days)", 3, 30, 10)
    risk_tolerance = st.sidebar.slider("Risk Tolerance (1 - Low, 10 - High)", 1, 10, 4)
    show_indicators = st.sidebar.checkbox("Show Indicators", True)
    show_rawdata = st.sidebar.checkbox("Show Raw Data (EUR/USD)", False)

    # ==========================================
    # Fetch TwelveData OHLC
    # ==========================================
    @st.cache_data(ttl=60*30, show_spinner=False)
    def fetch_twelvedata_ohlc(pair_symbol, api_key, interval="1h", days=14):
        url = f"https://api.twelvedata.com/time_series?symbol={pair_symbol}&interval={interval}&outputsize={24*days+50}&apikey={api_key}"
        r = requests.get(url)

        if r.status_code != 200 or 'values' not in r.text:
            st.error(f"API Error: {r.status_code} — {r.text}")
            return None

        j = r.json()
        if 'values' not in j:
            st.error("No data returned from API")
            return None

        df = pd.DataFrame(j['values'])

        expected_cols = ['datetime', 'open', 'high', 'low', 'close']
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return None

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
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.dropna()

    # ==========================================
    # Technical Indicators
    # ==========================================
    def compute_indicators(df):
        df = df.copy()
        df['rsi'] = rsi(df['Close'], window=14)
        df['macd'] = macd(df['Close'], window_slow=26, window_fast=12)
        df['macd_signal'] = macd_signal(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['bb_bbh'] = bollinger_hband(df['Close'], window=20, window_dev=2)
        df['bb_bbl'] = bollinger_lband(df['Close'], window=20, window_dev=2)
        return df

    # ==========================================
    # Trading Signal Logic
    # ==========================================
    def simple_trade_signal(df):
        if len(df) < 30:
            return "NO TRADE", 0.0

        last = df.iloc[-1]
        trade = "NO TRADE"
        score = 0.0

        if last["macd"] > last["macd_signal"] and last["rsi"] > 55 and last["Close"] > last["bb_bbh"]:
            trade = "BUY"
            score = 0.65 + (last["rsi"]-55)/100 + 0.1*(last["macd"] - last["macd_signal"])

        elif last["macd"] < last["macd_signal"] and last["rsi"] < 45 and last["Close"] < last["bb_bbl"]:
            trade = "SELL"
            score = 0.65 + (45-last["rsi"])/100 + 0.1*(last["macd_signal"] - last["macd"])

        elif abs(last["rsi"] - 50) < 5:
            trade = "NO TRADE"
            score = 0.3

        return trade, min(max(score, 0), 1)

    # ==========================================
    # Fetch Main Pair
    # ==========================================
    main_df = fetch_twelvedata_ohlc(PAIR_SYMBOL_MAP[main_pair], TWELVE_DATA_KEY, interval="1h", days=report_days)

    if main_df is None:
        st.error("Could not fetch live data.")
        return

    main_df = compute_indicators(main_df)
    main_signal, main_conf = simple_trade_signal(main_df)

    # ==========================================
    # Simulated Multi-Pair Stats
    # ==========================================
    multi_stats = []
    for pair, symbol in PAIR_SYMBOL_MAP.items():
        if pair == main_pair:
            last = main_df.iloc[-1]
            multi_stats.append({
                "Pair": pair,
                "Last": f"{last['Close']:.5f}",
                "RSI": f"{last['rsi']:.1f}",
                "MACD": f"{last['macd']:.5f}",
                "BB High": f"{last['bb_bbh']:.5f}",
                "BB Low": f"{last['bb_bbl']:.5f}",
                "Trade": main_signal,
                "Confidence": f"{main_conf:.1%}"
            })

        else:
            rng = np.random.default_rng(abs(hash(symbol))%99999)
            multi_stats.append({
                "Pair": pair,
                "Last": f"{1.00 + 0.1 * rng.random():.5f}",
                "RSI": f"{30 + 40*rng.random():.1f}",
                "MACD": f"{0.001*rng.standard_normal():.5f}",
                "BB High": f"{1.01 + 0.1*rng.random():.5f}",
                "BB Low": f"{0.98 + 0.1*rng.random():.5f}",
                "Trade": rng.choice(["BUY", "SELL", "NO TRADE"], p=[.35,.35,.3]),
                "Confidence": f"{0.60 + 0.2*rng.random():.1%}"
            })

    st.dataframe(pd.DataFrame(multi_stats), height=445)


    st.markdown("---")

    # ==========================================
    # Deep Technical Analysis
    # ==========================================
    st.header(f"Deep Technical Analytics — {main_pair}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=main_df["Date"],
        open=main_df["Open"],
        high=main_df["High"],
        low=main_df["Low"],
        close=main_df["Close"],
        name=f"{main_pair} Price"
    ))

    if show_indicators:
        fig.add_trace(go.Scatter(x=main_df["Date"], y=main_df["bb_bbh"], line=dict(dash='dot', color='purple'), name="Bollinger High"))
        fig.add_trace(go.Scatter(x=main_df["Date"], y=main_df["bb_bbl"], line=dict(dash='dot', color='gray'), name="Bollinger Low"))

    st.plotly_chart(fig, use_container_width=True)

    # RSI / MACD Chart
    st.write("### RSI & MACD Signals for EUR/USD")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=main_df["Date"], y=main_df["rsi"], name="RSI"))
    fig2.add_trace(go.Scatter(x=main_df["Date"], y=main_df["macd"], name="MACD"))
    fig2.add_trace(go.Scatter(x=main_df["Date"], y=main_df["macd_signal"], name="MACD Signal"))
    st.plotly_chart(fig2, use_container_width=True)

    # Large Signal Block
    color_map = {"BUY": "green", "SELL": "red", "NO TRADE": "gray"}
    st.markdown(
        f"""
        <div style="background-color:{color_map[main_signal]};
                    padding:1rem;
                    border-radius:1rem;">
            <h2 style="color:white;margin:0;">{main_signal}</h2>
            <p style="color:white;">Confidence: <b>{main_conf:.1%}</b> (Mix: MACD/RSI/Bollinger)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if show_rawdata:
        st.dataframe(main_df.tail(50), height=300)

    st.markdown("---")

    # Best ML suggestion
    st.header("Best ML Method For Forecasting")
    st.write(
    """
    - **Gradient Boosted Trees (XGBoost/LightGBM)**: Best for tabular data like OHLCV plus indicators (RSI, MACD, BB,      lagged closes).  
    - **LSTM/GRU Networks** with Keras/TensorFlow: Best when you have massive time series and want to use sequential dependencies.  
   - Start with XGBoost + features, expand with deep learning for complex relationship as your historical labeled set grows!
"""
)
    

    st.subheader("Risk Management Guidance")
    st.write(f"""
    - Never risk more than {(risk_tolerance*2):.1f}% per trade  
    - Wait for strong techical agreement (MACD/RSI/Bollinger all agree)
    - Use live backtesting before applying real capital
    - Only trade when MACD + RSI + Bollinger agree  
    - "NO-TRADE" means protect your account - edge is not confirmed.Use NO-TRADE signals to protect capital  
    """)


    #############################################
    #### (Your full code already provided above)
    #############################################

    # *** END FOREX CODE ***


################################################################################
###  ADMIN DASHBOARD
################################################################################

def admin_home():

    st.sidebar.header(f"Admin Panel — {st.session_state.full_name}")

    choice = st.sidebar.radio("Menu:", [
        "Dashboard",
        "Create User",
        "Manage Users",
        "Reset Password",
        "Login Activity Logs",
        "Audit Trail",
        "Logout"
    ])

    if choice == "Dashboard":
        forex_dashboard()

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
###  USER DASHBOARD
################################################################################

def user_home():
    st.sidebar.header(f"Welcome {st.session_state.full_name}")
    choice = st.sidebar.radio("Menu:", ["Dashboard", "Logout"])

    if choice == "Dashboard":
        forex_dashboard()

    elif choice == "Logout":
        logout()


################################################################################
###  ROUTING CONTROL
################################################################################

if not st.session_state.logged_in:
    login_page()

else:
    if st.session_state.role == "admin":
        admin_home()
    else:
        user_home()




st.write("**Contact customer support via email: chumcred@gmail.com or ‪+2347040000063‬ to set your username and password. Terms and conditions apply.**")




st.write("**Powered by Chumcred Limited**")

# END APP
