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
    """Scrapes real-time stock price from Google Finance DOM[<vertex-ai-rich-citation-chip>2</vertex-ai-rich-citation-chip>]."""
    exchanges = ["NASDAQ", "NYSE", "BATS", "OTCMKTS", "INDEXSP", "INDEXDJX"]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    # Try with common exchange variations first
    for exchange in exchanges:
        url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                # Class 'YMlKec fxKbKc' is Google's active market price class[<vertex-ai-rich-citation-chip>2</vertex-ai-rich-citation-chip>]
                price_div = soup.find("div", class_="YMlKec fxKbKc")
                if price_div:
                    price_str = price_div.text.replace("$", "").replace(",", "").strip()
                    return float(price_str), exchange
        except Exception:
            continue
            
    # Try raw ticker as final fallback
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
# Public Google Sheets Parser Helper
# ==========================================
def extract_tickers_from_google_sheet(url: str) -> list:
    """Extracts valid stock tickers from a public shared Google Sheet."""
    try:
        if "docs.google.com/spreadsheets" in url:
            match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
            if match:
                sheet_id = match.group(1)
                # Appends a direct CSV export endpoint to read the sheet via pandas
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                sheet_df = pd.read_csv(csv_url)
                
                # Look for columns that contain 1 to 5-letter uppercase strings
                for col in sheet_df.columns:
                    possible_tickers = sheet_df[col].astype(str).str.strip().str.upper()
                    valid = possible_tickers[possible_tickers.str.match(r'^[A-Z]{1,5}$', na=False)]
                    if len(valid) > 0:
                        return list(valid.unique())
    except Exception as e:
        st.error(
            f"Unable to read Sheet: {e}. "
            "Please make sure your Google Sheet is shared with: 'Anyone with the link can view'."
        )
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
            
            # 2. Pull historical data
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=365)

            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval="1d")

            if len(df) < 200:
                return {
                    "ticker": ticker,
                    "status": "Skipped",
                    "reason": "Requires 200+ historical trading days",
                }

            # 3. Inject Google Finance price as the latest current close BEFORE calculations
            if gf_price is not None:
                df.iloc[-1, df.columns.get_loc('Close')] = gf_price
                current_close = gf_price
                price_source = f"Google Finance ({exchange})"
            else:
                current_close = df["Close"].iloc[-1]
                price_source = "Yahoo Finance (Google Scrape Rate-Limited)"

            # Technical indicator computation
            df["SMA50"] = df["Close"].rolling(window=50).mean()
            df["SMA200"] = df["Close"].rolling(window=200).mean()
            df["RSI"] = self._calculate_rsi(df["Close"])

            current_sma50 = df["SMA50"].iloc[-1]
            current_sma200 = df["SMA200"].iloc[-1]
            current_rsi = df["RSI"].iloc[-1]

            is_currently_bullish = current_sma50 > current_sma200
            crossed_recently = False
            cross_day_index = None
            days_since_cross_actual = None

            if is_currently_bullish:
                # We search up to 30 trading days back to see when the cross actually occurred
                # (This powers the recommendation engine for "near-matches")
                for i in range(1, 31):
                    idx = -i
                    if len(df) + idx - 1 < 0:
                        break
                    if (
                        df["SMA50"].iloc[idx] > df["SMA200"].iloc[idx]
                        and df["SMA50"].iloc[idx - 1] <= df["SMA200"].iloc[idx - 1]
                    ):
                        days_since_cross_actual = i
                        if i <= self.cross_lookback:
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
                "is_currently_bullish": is_currently_bullish,
                "golden_cross_recent": crossed_recently,
                "days_since_cross": cross_day_index,
                "days_since_cross_actual": days_since_cross_actual,
                "price_source": price_source
            }
        except Exception as e:
            return {
                "ticker": ticker,
                "status": "Error",
                "reason": f"Failed: {str(e)}",
            }


# ==========================================
# Automated Parameter Recommendation Engine
# ==========================================
def generate_recommendations(skipped_or_no_match: list, rsi_min: float, rsi_max: float, lookback: int) -> list:
    """Analyzes non-matching stocks and suggests settings changes to find candidates."""
    recommendations = []
    lookback_near_matches = []
    rsi_near_matches = []
    bearish_count = 0
    total_processed = 0

    for item in skipped_or_no_match:
        if item.get("status") in ["Skipped", "Error"]:
            continue
        total_processed += 1

        # Track if it's completely bearish (no golden cross active)
        if not item.get("is_currently_bullish", False):
            bearish_count += 1
            continue

        actual_cross = item.get("days_since_cross_actual")
        current_rsi = item.get("RSI")
        ticker = item.get("ticker")

        # Case 1: Has a Golden Cross, but it happened outside our lookback limit
        if actual_cross is not None and actual_cross > lookback:
            lookback_near_matches.append({
                "ticker": ticker,
                "actual_cross": actual_cross,
                "rsi": current_rsi
            })

        # Case 2: Crossed within our lookback limit, but RSI was slightly out of bounds
        elif actual_cross is not None and actual_cross <= lookback:
            if current_rsi < rsi_min or current_rsi > rsi_max:
                rsi_near_matches.append({
                    "ticker": ticker,
                    "actual_cross": actual_cross,
                    "rsi": current_rsi
                })

    # Scenario A: All stocks are completely bearish
    if bearish_count == total_processed and total_processed > 0:
        recommendations.append(
            "⚠️ **Bearish Market Trend:** All scanned stocks are in a bearish phase (SMA50 < SMA200). "
            "No adjusting of settings can find a Golden Cross here. Consider adding other sectors, index ETFs, or waiting for a cycle shift."
        )
        return recommendations

    # Scenario B: Suggest Lookback Adjustment
    if lookback_near_matches:
        lookback_near_matches.sort(key=lambda x: x["actual_cross"])
        best_cands = lookback_near_matches[:3]
        cand_str = ", ".join([f"**{c['ticker']}** (crossed {c['actual_cross']} days ago, RSI: {c['rsi']})" for c in best_cands])
        max_needed_lookback = max([c['actual_cross'] for c in best_cands])
        recommendations.append(
            f"📅 **Adjust Golden Cross Lookback:** We detected active Golden Cross patterns slightly older than your {lookback}-day setting. "
            f"If you increase your **Lookback Days to {max_needed_lookback}**, you would capture: {cand_str}."
        )

    # Scenario C: Suggest RSI Bounds Adjustment
    if rsi_near_matches:
        cand_details = []
        suggest_min, suggest_max = rsi_min, rsi_max
        for c in rsi_near_matches:
            suggest_min = min(suggest_min, c['rsi'] - 1)
            suggest_max = max(suggest_max, c['rsi'] + 1)
            cand_details.append(f"**{c['ticker']}** (crossed {c['actual_cross']} days ago, RSI: {c['rsi']})")
        
        cand_str = ", ".join(cand_details[:3])
        recommendations.append(
            f"⚖️ **Adjust RSI Bounds:** Several stocks completed their Golden Cross, but their RSI fell outside your {rsi_min} - {rsi_max} range. "
            f"If you adjust your **RSI Bounds to {round(suggest_min, 1)} - {round(suggest_max, 1)}**, you would capture: {cand_str}."
        )

    if not recommendations:
        recommendations.append(
            "💡 **Expand Your Search:** No near-matches were found within a 30-day window. "
            "Try adding a wider variety of symbols or linking a larger public Google Sheets watchlist."
        )

    return recommendations


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
This agent scans stocks using **real-time prices pulled directly from Google Finance**[<vertex-ai-rich-citation-chip>1</vertex-ai-rich-citation-chip>] to locate setups where a **Golden Cross** 
recently formed, and the **RSI (14)** is sitting just above the 50 line (representing emerging bullish momentum).
"""
)

# Sidebar Configuration
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

st.sidebar.markdown("---")
st.sidebar.markdown(
    "💡 *A shorter lookback isolates exact crossovers. A wider lookback allows more time for the pattern to develop.*"
)

# Set Default starting tickers
default_tickers = "AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, AMD, INTC, JPM, MU, QCOM"

if "ticker_list" not in st.session_state:
    st.session_state["ticker_list"] = default_tickers

# Main Page Layout (Two Columns)
col1, col2 = st.columns([2, 1])

with col1:
    ticker_input = st.text_area(
        "Enter Stock Ticker Symbols (comma-separated):",
        value=st.session_state["ticker_list"],
        height=100,
    )

with col2:
    st.write("### Sync Public Google Sheet:")
    google_sheet_url = st.text_input(
        "Paste a public Google Sheets link to sync tickers:",
        placeholder="https://docs.google.com/spreadsheets/...",
    )
    if google_sheet_url:
        sheet_tickers = extract_tickers_from_google_sheet(google_sheet_url)
        if sheet_tickers:
            ticker_str = ", ".join(sheet_tickers)
            if st.session_state["ticker_list"] != ticker_str:
                st.session_state["ticker_list"] = ticker_str
                st.rerun()

# Start Analysis Button
if st.button("🚀 Run Screener Agent", type="primary"):
    # Parse inputs
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    if not tickers:
        st.warning("Please enter at least one ticker.")
    else:
        agent = StockScreenerAgent(
            rsi_low=rsi_min, rsi_high=rsi_max, cross_lookback=lookback
        )

        matches = []
        skipped_or_no_match = []

        # Streamlit Progress trackers
        progress_bar = st.progress(0)
        status_text = st.empty()

        for index, ticker in enumerate(tickers):
            status_text.text(f"Scanning {ticker}... ({index + 1}/{len(tickers)})")
            result = agent.analyze_ticker(ticker)

            if result["status"] == "MATCH":
                matches.append(result)
            else:
                skipped_or_no_match.append(result)

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
                "❌ No stocks in your list currently meet the strict Golden Cross & RSI criteria."
            )
            
            # ==========================================
            # RENDER AUTOMATED RECOMMENDATIONS
            # ==========================================
            st.markdown("### 💡 Recommended Adjustments")
            recs = generate_recommendations(skipped_or_no_match, rsi_min, rsi_max, lookback)
            
            with st.container():
                for rec in recs:
                    st.info(rec)

        # Expandable non-matches section for transparency
        with st.expander("🔍 View All Evaluated Stocks"):
            if skipped_or_no_match:
                df_no_match = pd.DataFrame(skipped_or_no_match)
                st.dataframe(df_no_match, use_container_width=True)
            else:
                st.write("All inputs were matches.")
