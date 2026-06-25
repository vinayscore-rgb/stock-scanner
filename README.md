How the Logic Works:
Golden Cross: A Golden Cross occurs when the 50-day Simple Moving Average (SMA) crosses above the 200-day SMA. Because catching a crossover on the exact current day is rare, the agent uses a lookback window (default: 5 days) to identify stocks that have recently crossed into a Golden Cross.
RSI (Relative Strength Index): Standard 14-day RSI. "Just above 50" is mathematically defined as a configurable range (default: 50<RSI≤58), which indicates emerging bullish momentum without being overbought.
