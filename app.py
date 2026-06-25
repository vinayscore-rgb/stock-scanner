import datetime
import pandas as pd
import streamlit as st
import yfinance as yf


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
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=365)

            stock = yf.Ticker(ticker)
            # FIX: Removed progress=False parameter which caused the Streamlit error
            df = stock.history(
                start=start_date, end=end_date, interval="1d"
            )

            if len(df) < 200:
                return {
                    "ticker": ticker,
                    "status": "Skipped",
                    "reason": "Less than 200 days of history",
                }

            # Tech indicators
            df["SMA50"] = df["Close"].rolling(window=50).mean()
            df["SMA200"] = df["Close"].rolling(window=200).mean()
            df["RSI"] = self._calculate_rsi(df["Close"])

            current_close = df["Close"].iloc[-1]
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
This agent scans a list of stock tickers to find candidates that have **recently completed a Golden Cross** 
(50-day SMA crossing above 200-day SMA) and currently have an **RSI just above 50** (indicating emerging bullish momentum).
"""
)

# Sidebar Configuration
st.sidebar.header("⚙️ Agent Settings")
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
    "💡 *A smaller lookback finds stocks that crossed very recently. A wider lookback gives a broader window.*"
)

# Main input layout
col1, col2 = st.columns([2, 1])

with col1:
    default_tickers = "AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, AMD, NFLX, INTC, WMT, JPM, V, DIS, UBER, COIN, PYPL, SQ, HOOD"
    ticker_input = st.text_area(
        "Enter Stock Ticker Symbols (comma-separated):",
        value=default_tickers,
        height=100,
    )

with col2:
    st.write("### Criteria Rules:")
    st.info(
        f"""
    1. **Golden Cross** occurred within the last **{lookback} trading days**.
    2. **RSI (14)** is currently between **{rsi_min}** and **{rsi_max}**.
    """
)

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
