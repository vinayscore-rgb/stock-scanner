"""
Golden Cross & RSI Screener
===========================

Scans a watchlist for two independent technical setups:

1. **Golden Cross + RSI band** - SMA50 crossed above SMA200 within the last
   N trading days, while RSI(14) currently sits inside a configurable band.
2. **Regular Bullish Divergence** - price printed a lower trough while RSI
   printed a higher trough, resolved within the last 30 trading days.

Price data comes from Yahoo Finance via ``yfinance``, split- and
dividend-adjusted (``auto_adjust=True``).
"""

import datetime as dt
import io
import re
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

HISTORY_YEARS = 2          # 2y keeps SMA200 well-defined across the cross window
MIN_TRADING_DAYS = 200     # SMA200 needs 200 observations before it exists
RSI_PERIOD = 14
SMA_FAST = 50
SMA_SLOW = 200
CROSS_SEARCH_DAYS = 30     # how far back we look to date an existing cross

# Divergence detection (fixed - deliberately not exposed in the sidebar)
DIV_WINDOW = 3             # bars on each side that define a local trough
DIV_MIN_SEPARATION = 5     # troughs closer than this are the same swing
DIV_MAX_SEPARATION = 40    # troughs further apart than this aren't related
DIV_RECENCY = 30           # the second trough must land within this many bars
DIV_SEARCH_BARS = 90       # only hunt for troughs in this recent slice

CACHE_TTL_SECONDS = 900    # 15 minutes
MAX_TICKERS = 150          # guard against a paste that would hang the app

NO_DIVERGENCE = {
    "has_divergence": False,
    "div_t1_date": "N/A",
    "div_t1_price": None,
    "div_t1_rsi": None,
    "div_t2_date": "N/A",
    "div_t2_price": None,
    "div_t2_rsi": None,
    "div_message": "No bullish divergence",
}


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI. ``ewm(com=period - 1)`` reproduces Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    # A zero average loss makes the ratio inf (or NaN if gains are zero too).
    # Both mean "no downside pressure in the window", which is RSI 100.
    return rsi.where(avg_loss != 0, 100.0)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with SMA50, SMA200 and RSI columns attached."""
    out = df.copy()
    close = out["Close"]
    out["SMA50"] = close.rolling(SMA_FAST).mean()
    out["SMA200"] = close.rolling(SMA_SLOW).mean()
    out["RSI"] = compute_rsi(close)
    return out


# ---------------------------------------------------------------------------
# Golden cross
# ---------------------------------------------------------------------------

def days_since_golden_cross(df: pd.DataFrame, max_lookback: int) -> Optional[int]:
    """
    Trading days since SMA50 last crossed *above* SMA200.

    Returns ``None`` if no crossing is found inside ``max_lookback`` bars.
    A return of 1 means the cross printed on the most recent bar.
    """
    fast = df["SMA50"]
    slow = df["SMA200"]

    for age in range(1, max_lookback + 1):
        current, previous = -age, -age - 1
        if len(df) + previous < 0:
            break

        values = (fast.iloc[current], slow.iloc[current],
                  fast.iloc[previous], slow.iloc[previous])
        if any(pd.isna(v) for v in values):
            break

        crossed = values[0] > values[1] and values[2] <= values[3]
        if crossed:
            return age

    return None


# ---------------------------------------------------------------------------
# Bullish divergence
# ---------------------------------------------------------------------------

def _format_date(stamp) -> str:
    try:
        return stamp.strftime("%Y-%m-%d")
    except AttributeError:
        return str(stamp)


def find_local_troughs(df: pd.DataFrame) -> List[dict]:
    """Bars that are strictly lower than every neighbour within DIV_WINDOW."""
    close = df["Close"]
    rsi = df["RSI"]
    total = len(df)

    troughs: List[dict] = []
    first = max(DIV_WINDOW, total - DIV_SEARCH_BARS)

    for i in range(first, total - DIV_WINDOW):
        price = close.iloc[i]
        strength = rsi.iloc[i]
        if pd.isna(price) or pd.isna(strength):
            continue

        neighbourhood = close.iloc[i - DIV_WINDOW:i + DIV_WINDOW + 1]
        # Every other bar in the window must be strictly higher.
        if int((neighbourhood > price).sum()) != len(neighbourhood) - 1:
            continue

        troughs.append({
            "idx": i,
            "date": _format_date(close.index[i]),
            "price": float(price),
            "rsi": float(strength),
        })

    return troughs


def detect_bullish_divergence(df: pd.DataFrame) -> dict:
    """
    Find a regular bullish divergence: a lower price trough paired with a
    higher RSI trough, with the second trough resolving recently.
    """
    troughs = find_local_troughs(df)
    total = len(df)
    pairs: List[Tuple[dict, dict]] = []

    for a in range(len(troughs)):
        for b in range(a + 1, len(troughs)):
            first, second = troughs[a], troughs[b]

            separation = second["idx"] - first["idx"]
            if not DIV_MIN_SEPARATION <= separation <= DIV_MAX_SEPARATION:
                continue
            if second["idx"] < total - DIV_RECENCY:
                continue

            lower_low = second["price"] < first["price"]
            higher_rsi = second["rsi"] > first["rsi"]
            if lower_low and higher_rsi:
                pairs.append((first, second))

    if not pairs:
        return dict(NO_DIVERGENCE)

    # Report the most recently resolved divergence.
    pairs.sort(key=lambda pair: pair[1]["idx"], reverse=True)
    first, second = pairs[0]

    return {
        "has_divergence": True,
        "div_t1_date": first["date"],
        "div_t1_price": round(first["price"], 2),
        "div_t1_rsi": round(first["rsi"], 2),
        "div_t2_date": second["date"],
        "div_t2_price": round(second["price"], 2),
        "div_t2_rsi": round(second["rsi"], 2),
        "div_message": "Valid bullish divergence detected",
    }


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

def screen_ticker(
    ticker: str,
    history: Optional[pd.DataFrame],
    rsi_low: float,
    rsi_high: float,
    lookback: int,
) -> dict:
    """Run both setups against one ticker's history and return a flat record."""
    available = 0 if history is None else len(history)
    if available < MIN_TRADING_DAYS:
        return {
            "ticker": ticker,
            "status": "Skipped",
            "reason": "Only {} trading days available; {} needed for SMA200.".format(
                available, MIN_TRADING_DAYS
            ),
        }

    df = add_indicators(history)

    close_now = df["Close"].iloc[-1]
    sma_fast = df["SMA50"].iloc[-1]
    sma_slow = df["SMA200"].iloc[-1]
    rsi_now = df["RSI"].iloc[-1]

    if any(pd.isna(v) for v in (close_now, sma_fast, sma_slow, rsi_now)):
        return {
            "ticker": ticker,
            "status": "Skipped",
            "reason": "Indicator values incomplete for the latest bar.",
        }

    is_bullish = bool(sma_fast > sma_slow)
    cross_age = days_since_golden_cross(df, CROSS_SEARCH_DAYS) if is_bullish else None
    crossed_recently = cross_age is not None and cross_age <= lookback
    rsi_matches = rsi_low < rsi_now <= rsi_high

    record = {
        "ticker": ticker,
        "status": "MATCH" if (crossed_recently and rsi_matches) else "NO MATCH",
        "current_price": round(float(close_now), 2),
        "SMA50": round(float(sma_fast), 2),
        "SMA200": round(float(sma_slow), 2),
        "RSI": round(float(rsi_now), 2),
        "is_currently_bullish": is_bullish,
        "golden_cross_recent": crossed_recently,
        "days_since_cross": cross_age if crossed_recently else None,
        "days_since_cross_actual": cross_age,
        "as_of": _format_date(df.index[-1]),
    }
    record.update(detect_bullish_divergence(df))
    return record


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_history(tickers: Tuple[str, ...], years: int) -> Dict[str, pd.DataFrame]:
    """
    Download daily history for every ticker in one batched request.

    One batched call is far friendlier to Yahoo's rate limits than a request
    per ticker, which matters on shared cloud IPs. Tickers that Yahoo can't
    resolve are simply absent from the returned mapping.
    """
    if not tickers:
        return {}

    end = dt.date.today() + dt.timedelta(days=1)  # end is exclusive
    start = end - dt.timedelta(days=365 * years + 10)

    raw = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
        timeout=30,
    )

    if raw is None or raw.empty:
        return {}

    histories: Dict[str, pd.DataFrame] = {}
    multi = isinstance(raw.columns, pd.MultiIndex)

    for ticker in tickers:
        try:
            if multi:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                frame = raw[ticker].copy()
            else:
                frame = raw.copy()
        except (KeyError, IndexError):
            continue

        if "Close" not in frame.columns:
            continue

        frame = frame.dropna(subset=["Close"])
        if not frame.empty:
            histories[ticker] = frame

    return histories


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def tickers_from_google_sheet(url: str) -> Tuple[List[str], Optional[str]]:
    """
    Pull tickers out of a published Google Sheet.

    Returns ``(tickers, error)``. The error is returned rather than rendered
    here because this function is cached - a cached call would skip any
    Streamlit side effect on subsequent runs.
    """
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        return [], "That doesn't look like a Google Sheets link."

    csv_url = "https://docs.google.com/spreadsheets/d/{}/export?format=csv".format(
        match.group(1)
    )
    gid = re.search(r"[#&?]gid=([0-9]+)", url)
    if gid:
        csv_url += "&gid={}".format(gid.group(1))

    try:
        response = requests.get(csv_url, timeout=15)
        response.raise_for_status()
        sheet = pd.read_csv(io.StringIO(response.text))
    except Exception as exc:  # network, HTTP, or parse failure
        return [], (
            "Couldn't read that sheet ({}). Make sure it is shared as "
            "'Anyone with the link can view'.".format(exc)
        )

    for column in sheet.columns:
        values = sheet[column].astype(str).str.strip().str.upper()
        valid = values[values.str.fullmatch(r"[A-Z]{1,5}([.\-][A-Z])?", na=False)]
        if not valid.empty:
            return list(dict.fromkeys(valid)), None

    return [], "No column in that sheet looked like a list of ticker symbols."


def parse_tickers(raw: str) -> List[str]:
    """Split user input into a deduplicated, upper-cased ticker list."""
    candidates = re.split(r"[,\s]+", raw.upper())
    cleaned = [c.strip() for c in candidates if c.strip()]
    valid = [c for c in cleaned if re.fullmatch(r"[A-Z0-9]{1,6}([.\-][A-Z]{1,2})?", c)]
    return list(dict.fromkeys(valid))


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

def generate_recommendations(
    non_matches: Sequence[dict],
    rsi_min: float,
    rsi_max: float,
    lookback: int,
) -> List[str]:
    """Suggest setting changes that would surface near-miss candidates."""
    lookback_near: List[dict] = []
    rsi_near: List[dict] = []
    bearish = 0
    considered = 0

    for item in non_matches:
        if item.get("status") in ("Skipped", "Error"):
            continue
        considered += 1

        if not item.get("is_currently_bullish", False):
            bearish += 1
            continue

        cross_age = item.get("days_since_cross_actual")
        if cross_age is None:
            continue

        entry = {
            "ticker": item.get("ticker"),
            "cross_age": cross_age,
            "rsi": item.get("RSI"),
        }

        if cross_age > lookback:
            # Golden cross is real, just older than the current window.
            lookback_near.append(entry)
        elif entry["rsi"] is not None and not (rsi_min < entry["rsi"] <= rsi_max):
            # Crossed in time, but RSI sat outside the band.
            rsi_near.append(entry)

    recommendations: List[str] = []

    if considered > 0 and bearish == considered:
        return [
            "⚠️ **Bearish across the board:** every scanned stock has SMA50 below "
            "SMA200, so no setting will surface a Golden Cross here. Consider adding "
            "other sectors or index ETFs, or waiting for the cycle to turn."
        ]

    if lookback_near:
        lookback_near.sort(key=lambda e: e["cross_age"])
        best = lookback_near[:3]
        needed = max(e["cross_age"] for e in best)
        names = ", ".join(
            "**{}** (crossed {} days ago, RSI {})".format(e["ticker"], e["cross_age"], e["rsi"])
            for e in best
        )
        recommendations.append(
            "📅 **Widen the cross lookback:** some active Golden Crosses are just older "
            "than your {}-day window. Raising **Lookback Days to {}** would capture: "
            "{}.".format(lookback, needed, names)
        )

    if rsi_near:
        suggested_min, suggested_max = rsi_min, rsi_max
        for entry in rsi_near:
            suggested_min = min(suggested_min, entry["rsi"] - 1)
            suggested_max = max(suggested_max, entry["rsi"] + 1)
        names = ", ".join(
            "**{}** (crossed {} days ago, RSI {})".format(e["ticker"], e["cross_age"], e["rsi"])
            for e in rsi_near[:3]
        )
        recommendations.append(
            "⚖️ **Widen the RSI band:** these completed a Golden Cross but their RSI fell "
            "outside {} - {}. Adjusting **RSI Bounds to {} - {}** would capture: {}.".format(
                rsi_min, rsi_max, round(suggested_min, 1), round(suggested_max, 1), names
            )
        )

    if not recommendations:
        recommendations.append(
            "💡 **Widen the search:** no near-misses turned up in a 30-day window. Try a "
            "broader list of symbols, or sync a larger public Google Sheets watchlist."
        )

    return recommendations


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Golden Cross & RSI Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Technical Screener Agent")
st.markdown(
    "Scans a watchlist for **Golden Cross + RSI** setups and **Regular Bullish "
    "Divergences** in parallel. Prices are split- and dividend-adjusted daily "
    "closes from Yahoo Finance."
)

st.sidebar.header("⚙️ Screening Rules")
rsi_min = st.sidebar.slider("Minimum RSI", 30.0, 70.0, 50.0, step=1.0)
rsi_max = st.sidebar.slider("Maximum RSI", 31.0, 80.0, 58.0, step=1.0)
lookback = st.sidebar.number_input(
    "Golden Cross lookback (trading days)", min_value=1, max_value=CROSS_SEARCH_DAYS,
    value=5, step=1,
)

if rsi_min >= rsi_max:
    st.sidebar.error("Minimum RSI must be below maximum RSI - nothing can match.")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Clear cached prices"):
    fetch_history.clear()
    tickers_from_google_sheet.clear()
    st.sidebar.success("Cache cleared - the next scan refetches.")
st.sidebar.caption(
    "Prices are cached for {} minutes and fetched in a single batched request.".format(
        CACHE_TTL_SECONDS // 60
    )
)

DEFAULT_TICKERS = "AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, AMD, INTC, JPM, MU, QCOM"
if "ticker_list" not in st.session_state:
    st.session_state["ticker_list"] = DEFAULT_TICKERS

left, right = st.columns([2, 1])

with left:
    ticker_input = st.text_area(
        "Ticker symbols (comma or space separated):",
        value=st.session_state["ticker_list"],
        height=100,
    )

with right:
    st.markdown("**Sync a public Google Sheet**")
    sheet_url = st.text_input(
        "Paste a shared Google Sheets link:",
        placeholder="https://docs.google.com/spreadsheets/...",
    )
    if sheet_url:
        sheet_tickers, sheet_error = tickers_from_google_sheet(sheet_url)
        if sheet_error:
            st.error(sheet_error)
        elif sheet_tickers:
            joined = ", ".join(sheet_tickers)
            if st.session_state["ticker_list"] != joined:
                st.session_state["ticker_list"] = joined
                st.rerun()
            st.success("Synced {} tickers.".format(len(sheet_tickers)))

if st.button("🚀 Run screener", type="primary"):
    tickers = parse_tickers(ticker_input)

    if not tickers:
        st.warning("Enter at least one ticker symbol.")
        st.stop()

    if len(tickers) > MAX_TICKERS:
        st.warning(
            "Scanning the first {} of {} tickers.".format(MAX_TICKERS, len(tickers))
        )
        tickers = tickers[:MAX_TICKERS]

    with st.spinner("Fetching {} tickers from Yahoo Finance...".format(len(tickers))):
        try:
            histories = fetch_history(tuple(tickers), HISTORY_YEARS)
        except Exception as exc:
            st.error("Price download failed: {}".format(exc))
            st.stop()

    if not histories:
        st.error(
            "No price data came back. Yahoo may be rate-limiting this deployment, "
            "or none of those symbols resolved. Try again shortly."
        )
        st.stop()

    progress = st.progress(0.0)
    status = st.empty()
    results: List[dict] = []

    for position, ticker in enumerate(tickers, start=1):
        status.text("Analyzing {} ({}/{})".format(ticker, position, len(tickers)))
        try:
            results.append(
                screen_ticker(ticker, histories.get(ticker), rsi_min, rsi_max, lookback)
            )
        except Exception as exc:
            results.append({
                "ticker": ticker,
                "status": "Error",
                "reason": "Analysis failed: {}".format(exc),
            })
        progress.progress(position / len(tickers))

    progress.empty()
    status.empty()

    matches = [r for r in results if r.get("status") == "MATCH"]
    non_matches = [r for r in results if r.get("status") != "MATCH"]
    divergences = [r for r in results if r.get("has_divergence")]

    st.markdown("## 📊 Results")
    tab_cross, tab_div, tab_all = st.tabs([
        "🎯 Golden Cross & RSI",
        "🐂 Bullish Divergences",
        "🔍 All Scanned",
    ])

    with tab_cross:
        if matches:
            st.success("Found {} stock(s) meeting the Golden Cross criteria.".format(len(matches)))
            st.dataframe(
                pd.DataFrame(matches)[[
                    "ticker", "current_price", "SMA50", "SMA200",
                    "RSI", "days_since_cross", "as_of",
                ]],
                hide_index=True,
            )
        else:
            st.error("No stocks meet the current Golden Cross and RSI parameters.")
            st.markdown("### 💡 Suggested adjustments")
            for note in generate_recommendations(non_matches, rsi_min, rsi_max, lookback):
                st.info(note)

    with tab_div:
        st.markdown(
            "A **regular bullish divergence** is a lower price trough paired with a "
            "higher RSI trough - a possible reversal signal. Only divergences "
            "resolving in the last {} trading days are shown.".format(DIV_RECENCY)
        )
        if divergences:
            st.success("Detected {} stock(s) with a bullish divergence.".format(len(divergences)))
            table = pd.DataFrame(divergences)[[
                "ticker", "current_price",
                "div_t1_date", "div_t1_price", "div_t1_rsi",
                "div_t2_date", "div_t2_price", "div_t2_rsi",
            ]].rename(columns={
                "div_t1_date": "Trough 1 date",
                "div_t1_price": "Trough 1 price",
                "div_t1_rsi": "Trough 1 RSI",
                "div_t2_date": "Trough 2 date",
                "div_t2_price": "Trough 2 price",
                "div_t2_rsi": "Trough 2 RSI",
            })
            st.dataframe(table, hide_index=True)
        else:
            st.info("No bullish divergences in the last {} trading days.".format(DIV_RECENCY))

    with tab_all:
        st.markdown("Everything returned by the scan, including skips and errors.")
        all_results = pd.DataFrame(results)
        st.dataframe(all_results, hide_index=True)
        st.download_button(
            "⬇️ Download results as CSV",
            data=all_results.to_csv(index=False).encode("utf-8"),
            file_name="screener-results-{}.csv".format(dt.date.today().isoformat()),
            mime="text/csv",
        )
