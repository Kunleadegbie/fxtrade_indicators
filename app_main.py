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

if "last_email_alerts" not in st.session_state:
    st.session_state.last_email_alerts = {}


################################################################################
### AUTH
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
            st.error("Your account has been blocked.")
        else:
            st.session_state.logged_in = True
            st.session_state.role = user["role"]
            st.session_state.username = user["username"]
            st.session_state.full_name = user["full_name"]
            st.rerun()


################################################################################
### DASHBOARD (UNCHANGED CORE)
################################################################################
def forex_dashboard():
    st.title("📊 AI Forex Dashboard")
    st.write("Market scanner and trading intelligence system running...")


################################################################################
### ADMIN DASHBOARD (FIXED CLEANLY)
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
        st.title("Create User")

    elif choice == "Manage Users":
        st.title("Manage Users")

    elif choice == "Reset Password":
        st.title("Reset Password")

    elif choice == "Login Activity Logs":
        st.title("Login Logs")

    elif choice == "Audit Trail":
        st.title("Audit Trail")

    elif choice == "Logout":
        logout()


################################################################################
### USER DASHBOARD (FIXED)
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
### ROUTING
################################################################################
if not st.session_state.logged_in:
    login_page()
else:
    if st.session_state.role == "admin":
        admin_home()
    else:
        user_home()


st.write("Contact support: chumcred@gmail.com | +2348025420200")
st.write("Powered by Chumcred Limited")