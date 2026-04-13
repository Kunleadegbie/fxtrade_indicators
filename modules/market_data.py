import os
import time
import requests
import pandas as pd
import streamlit as st


@st.cache_data(ttl=90, show_spinner=False)
def fetch_market_data(symbol: str = "EUR/USD", interval: str = "1h", outputsize: int = 200):
    api_key = os.getenv("TWELVE_DATA_KEY")

    if not api_key:
        return None

    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}"
    )

    for _ in range(2):
        try:
            r = requests.get(url, timeout=15)

            if r.status_code != 200:
                time.sleep(1)
                continue

            content_type = r.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                time.sleep(1)
                continue

            data = r.json()

            if "values" not in data:
                time.sleep(1)
                continue

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

            df = df.dropna()

            if len(df) == 0:
                return None

            return df

        except Exception:
            time.sleep(1)

    return None


@st.cache_data(ttl=300, show_spinner=False)
def get_quote_to_usd_rate(quote_ccy: str):
    if quote_ccy == "USD":
        return 1.0

    direct_pair = f"{quote_ccy}/USD"
    df_direct = fetch_market_data(direct_pair, "1h", 10)

    if df_direct is not None and len(df_direct) > 0:
        return float(df_direct["Close"].iloc[-1])

    inverse_pair = f"USD/{quote_ccy}"
    df_inverse = fetch_market_data(inverse_pair, "1h", 10)

    if df_inverse is not None and len(df_inverse) > 0:
        rate = float(df_inverse["Close"].iloc[-1])
        if rate != 0:
            return 1 / rate

    return None