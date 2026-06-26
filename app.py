import datetime
import re
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

# ==========================================
# Google Finance Scraping Helper
# ==========================================
def scrape_google_finance_price(ticker: str) -> tuple:
    """Scrapes the real-time stock price from Google Finance.
    
    Tries common exchanges since Google Finance maps tickers as TICKER:EXCHANGE.
    """
    exchanges = ["NASDAQ", "NYSE", "BATS", "OTCMKTS", "INDEXSP", "INDEXDJX"]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    # 1. Try with common exchanges
    for exchange in exchanges:
        url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                # Class 'YMlKec fxKbKc' is the unique identifier for the main price on Google Finance
                price_div = soup.find("div", class_="YMlKec fxKbKc")
                if price_div:
                    price_str = price_div.text.replace("$", "").replace(",", "").strip()
                    return float(price_str), exchange
        except Exception:
            continue
            
    # 2. Try raw ticker as fallback
    try:
        url = f"https://www.google.com/finance/quote/{ticker}"
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            price_div = soup.find("div", class_="YMlKec fxKbKc")
            if price_div:
                price_str = price_div.text.replace("$", "").replace(",", "").strip()
                return float(price_str), "Global"
    except Exception:
        pass
        
    return None, None


# ==========================================
# Google Sheets Parser Helper
# ==========================================
def extract_tickers_from_google_sheet(url: str) -> list:
    """Extracts valid ticker symbols from a shared Google Sheet link."""
    try:
        if "docs.google.com/spreadsheets" in url:
            match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
            if match:
                sheet_id = match.group(1)
                # Redirect URL to download Sheet as CSV
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                sheet_df = pd.read_csv(csv_url)
                
                # Scan columns for valid ticker symbols (1-5 capital letters)
                for col in sheet_df.columns:
                    possible_tickers = sheet_df[col].astype(str).str.strip().str.upper()
                    valid = possible_tickers[possible_tickers.str.match(r'^[A-Z]{1,5}$', na=False)]
                    if len(valid) > 0:
                        return list(valid.unique())
    except Exception as e:
        st.error(f"Error parsing Google Sheet: {e}")
    return []


# ==========================================
# The Stock Screener Agent Logic
# ==========================================
class StockScreenerAgent:

    def __init__(
        self,
        rsi_low: float = 50.0,
        rsi_high: float = 58.0,
        cross_lookback: int = 5,
    ):
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.cross_lookback = cross_lookback

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def analyze_ticker(self, ticker: str) -> dict:
        try:
            # 1. Pull current price from Google Finance first
            gf_price, exchange = scrape_google_finance_price(ticker)
            
            # 2. Pull historical context for SMA & RSI calculation
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=365)

            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval="1d")

            if len(df) < 200:
                return {
                    "ticker": ticker,
                    "status": "Skipped",
                    "reason": "Less than 200 days of history",
                }

            # 3. Inject Google Finance price as the latest current close
            if gf_price is not None:
                df.iloc[-1, df.columns.get_loc('Close')] = gf_price
                current_close = gf_price
                price_source = f"Google Finance ({exchange})"
            else:
                current_close = df["Close"].iloc[-1]
                price_source = "Yahoo Finance (Google Scrape Rate-Limited)"

            # Technical indicators
            df["SMA50"] = df["Close"].rolling(window=50).mean()
            df["SMA200"] = df["Close"].rolling(window=200).mean()
            df["RSI"] = self._calculate_rsi(df["Close"])

            current_sma50 = df["SMA50"].iloc[-1]
            current_sma200 = df["SMA200"].iloc[-1]
            current_rsi = df["RSI"].iloc[-1]

            is_currently_bullish = current_sma50 > current_sma200
            crossed_recently = False
            cross_day_index = None

            if is_currently_bullish:
                for i in range(1, self.cross_lookback + 1):
                    idx = -i
                    if (
                        df["SMA50"].iloc[idx] > df["SMA200"].iloc[idx]
                        and df["SMA50"].iloc[idx - 1]
                        <= df["SMA200"].iloc[idx - 1]
                    ):
                        crossed_recently = True
                        cross_day_index = i
                        break

            rsi_matches = self.rsi_low < current_rsi <= self.rsi_high
            meets_criteria = crossed_recently and rsi_matches

            return {
                "ticker": ticker,
                "status": "MATCH" if meets_criteria else "NO MATCH",
                "current_price": round(current_close, 2),
                "SMA50": round(current_sma50, 2),
                "SMA200": round(current_sma200, 2),
                "RSI": round(current_rsi, 2),
                "golden_cross_recent": crossed_recently,
                "days_since_cross": (
                    cross_day_index if crossed_recently else None
                ),
                "price_source": price_source
            }
        except Exception as e:
            return {
                "ticker": ticker,
                "status": "Error",
                "reason": f"Failed: {str(e)}",
            }


# ==========================================
# Streamlit Web Interface Design
# ==========================================
st.set_page_config(
    page_title="Golden Cross & RSI Screener Agent",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Golden Cross & RSI Screener Agent")
st.markdown(
    """
This agent scans a list of stock tickers using **live prices scraped from Google Finance** to find stocks that have **recently completed a Golden Cross** 
and currently have an **RSI just above 50**.
"""
)

# ------------------------------------------
# Session State for Google User / Watchlist
# ------------------------------------------
if "google_user" not in st.session_state:
    st.session_state["google_user"] = None

# Sidebar Configuration
st.sidebar.header("👤 Google Account")

# Google Login Simulation
if not st.session_state["google_user"]:
    st.sidebar.info("Sign in with Google to load your saved watchlists.")
    if st.sidebar.button("🔴 Sign in with Google", use_container_width=True):
        st.session_state["google_user"] = {
            "name": "Alex Investor",
            "email": "alex.investor@gmail.com",
            "watchlist": ["GOOGL", "AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "META"]
        }
        st.toast("Welcome back, Alex!")
        st.rerun()
else:
    user = st.session_state["google_user"]
    st.sidebar.success(f"Logged in as **{user['name']}**")
    st.sidebar.caption(f"📧 {user['email']}")
    
    # Editable Google Finance Watchlist
    st.sidebar.markdown("---")
    st.sidebar.markdown("⚙️ **Your Google Watchlist**")
    edited_watchlist = st.sidebar.text_area(
        "Manage Tickers:",
        value=", ".join(user["watchlist"]),
        height=80,
    )
    user["watchlist"] = [t.strip().upper() for t in edited_watchlist.split(",") if t.strip()]
    
    if st.sidebar.button("🚪 Sign Out", use_container_width=True):
        st.session_state["google_user"] = None
        st.rerun()

# Technical Settings Sidebar
st.sidebar.header("⚙️ Screener Rules")
rsi_min = st.sidebar.slider(
    "Minimum RSI", min_value=30.0, max_value=70.0, value=50.0, step=1.0
)
rsi_max = st.sidebar.slider(
    "Maximum RSI (Just Above 50)",
    min_value=51.0,
    max_value=80.0,
    value=58.0,
    step=1.0,
)
lookback = st.sidebar.number_input(
    "Golden Cross Lookback (Days)", min_value=1, max_value=30, value=5, step=1
)

# Determine the Starting Tickers List
# 1. Logged in watchlist -> 2. Default stocks
if st.session_state["google_user"]:
    starting_tickers = ", ".join(st.session_state["google_user"]["watchlist"])
    list_source_msg = "📂 Loaded starting tickers from your **Google Watchlist**."
else:
    starting_tickers = "AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, AMD, NFLX, INTC, WMT, JPM, V, DIS"
    list_source_msg = "ℹ️ Using **default stock symbols** as starting point. Log in to use your personal watchlist."

# Main input layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"*{list_source_msg}*")
    ticker_input = st.text_area(
        "Edit Ticker Symbols to Scan (comma-separated):",
        value=starting_tickers,
        height=100,
    )

with col2:
    st.write("### Google Sheets Watchlist Sync:")
    google_sheet_url = st.text_input(
        "Or paste a shared Google Sheets link:",
        placeholder="https://docs.google.com/spreadsheets/...",
    )
    if google_sheet_url:
        sheet_tickers = extract_tickers_from_google_sheet(google_sheet_url)
        if sheet_tickers:
            st.success(f"Found {len(sheet_tickers)} tickers in Sheet!")
            ticker_input = ", ".join(sheet_tickers)

# Start Analysis Button
if st.button("🚀 Run Screener Agent", type="primary"):
    # Parse tickers
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    if not tickers:
        st.warning("Please enter at least one ticker.")
    else:
        # Initialize Agent with user criteria
        agent = StockScreenerAgent(
            rsi_low=rsi_min, rsi_high=rsi_max, cross_lookback=lookback
        )

        matches = []
        skipped_or_no_match = []

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        for index, ticker in enumerate(tickers):
            status_text.text(f"Scanning {ticker}... ({index + 1}/{len(tickers)})")
            # Run analysis
            result = agent.analyze_ticker(ticker)

            if result["status"] == "MATCH":
                matches.append(result)
            else:
                skipped_or_no_match.append(result)

            # Update progress bar
            progress_bar.progress((index + 1) / len(tickers))

        status_text.text("Scan Completed!")
        progress_bar.empty()

        # Display Results
        st.markdown("## 📊 Scan Results")

        if matches:
            st.success(
                f"🎉 Found **{len(matches)}** stock(s) meeting the criteria!"
            )
            df_matches = pd.DataFrame(matches)

            # Style and render matches table
            display_cols = [
                "ticker",
                "current_price",
                "price_source",
                "SMA50",
                "SMA200",
                "RSI",
                "days_since_cross",
            ]
            st.dataframe(df_matches[display_cols], use_container_width=True)
        else:
            st.error(
                "❌ No stocks in the provided list currently meet the criteria."
            )

        # Expandable non-matches section for transparency
        with st.expander("🔍 View Non-Matching Stocks"):
            if skipped_or_no_match:
                df_no_match = pd.DataFrame(skipped_or_no_match)
                st.dataframe(df_no_match, use_container_width=True)
            else:
                st.write("All inputs were matches.")
