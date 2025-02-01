# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 10:03:33 2025

@author: kushp
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------------------
# Data Acquisition Module
# ------------------------------

class DataFetcher:
    """
    Fetch historical stock price data using yfinance.
    """
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch(self) -> pd.DataFrame:
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if data.empty:
            raise ValueError("No data fetched. Please check ticker and date range.")
        data.dropna(inplace=True)
        return data

# ------------------------------
# Strategy Builder Module (Stock Strategies)
# ------------------------------

class Strategy:
    """
    Base Strategy class. Subclasses should implement generate_signals (for stock strategies)
    or simulate (for options strategies).
    """
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement generate_signals()")

class SMACrossoverStrategy(Strategy):
    """
    Simple Moving Average (SMA) Crossover Strategy.
    """
    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['short_mavg'] = data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        signals['long_mavg'] = data['Close'].rolling(window=self.long_window, min_periods=1).mean()
        signals['signal'] = np.where(signals['short_mavg'] > signals['long_mavg'], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()  # +1: buy, -1: sell
        return signals

class RSITradingStrategy(Strategy):
    """
    RSI Trading Strategy.
    When RSI is below oversold threshold, enter long (signal 1);
    when above overbought threshold, exit (signal 0).
    """
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def compute_rsi(self, data: pd.DataFrame) -> pd.Series:
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=self.period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['rsi'] = self.compute_rsi(data)
        signal_series = []
        current_signal = 0
        for rsi_val in signals['rsi']:
            if rsi_val < self.oversold:
                current_signal = 1
            elif rsi_val > self.overbought:
                current_signal = 0
            signal_series.append(current_signal)
        signals['signal'] = signal_series
        signals['positions'] = pd.Series(signal_series, index=signals.index).diff().fillna(0.0)
        return signals

class BollingerBandsStrategy(Strategy):
    """
    Bollinger Bands Mean Reversion Strategy.
    Buy when price falls below the lower band; sell when price rises above the upper band.
    """
    def __init__(self, window: int = 20, std_multiplier: float = 2.0):
        self.window = window
        self.std_multiplier = std_multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        rolling_mean = data['Close'].rolling(window=self.window, min_periods=1).mean()
        rolling_std = data['Close'].rolling(window=self.window, min_periods=1).std()
        upper_band = rolling_mean + self.std_multiplier * rolling_std
        lower_band = rolling_mean - self.std_multiplier * rolling_std

        signal_series = []
        current_signal = 0
        for price, lb, ub in zip(data['Close'], lower_band, upper_band):
            if price < lb:
                current_signal = 1
            elif price > ub:
                current_signal = 0
            signal_series.append(current_signal)
        signals['signal'] = signal_series
        signals['positions'] = pd.Series(signal_series, index=signals.index).diff().fillna(0.0)
        signals['rolling_mean'] = rolling_mean
        signals['upper_band'] = upper_band
        signals['lower_band'] = lower_band
        return signals

class SecondDerivativeMAStrategy(Strategy):
    """
    Second Derivative Moving Average Strategy.
    
    Computes a moving average (MA) over a specified window and then calculates its
    second derivative (i.e. acceleration). A buy signal (1) is generated when the
    second derivative exceeds a positive threshold; a sell signal (0) is generated when
    it falls below the negative threshold; otherwise, the previous signal is maintained.
    
    Parameters:
      - ma_window: period for the moving average.
      - threshold: magnitude required for the second derivative to trigger a change.
    """
    def __init__(self, ma_window: int = 50, threshold: float = 0.1):
        self.ma_window = ma_window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        # Compute moving average
        signals['ma'] = data['Close'].rolling(window=self.ma_window, min_periods=1).mean()
        # Compute second derivative using two successive differences
        signals['second_deriv'] = signals['ma'].diff().diff()
        
        # Generate signals: if second derivative > threshold then buy (1),
        # if < -threshold then sell (0); else hold previous signal.
        signal = []
        prev_signal = 0
        for val in signals['second_deriv']:
            if pd.isna(val):
                signal.append(prev_signal)
            elif val > self.threshold:
                prev_signal = 1
                signal.append(1)
            elif val < -self.threshold:
                prev_signal = 0
                signal.append(0)
            else:
                signal.append(prev_signal)
        signals['signal'] = signal
        signals['positions'] = pd.Series(signal, index=signals.index).diff().fillna(0.0)
        return signals

# ------------------------------
# Options Strategies (Simplified Simulation)
# ------------------------------

class WheelStrategyOptions(Strategy):
    """
    A very simplified simulation of the Wheel strategy.
    When not holding stock, sell a cash‑secured put; if assigned, switch to stock-owned
    and then sell a covered call.
    The simulation uses a “holding period” (in days) to model option expiration outcomes.
    """
    def __init__(self, put_offset=0.05, call_offset=0.05, put_premium_rate=0.02,
                 call_premium_rate=0.02, holding_period=5, shares=100):
        self.put_offset = put_offset
        self.call_offset = call_offset
        self.put_premium_rate = put_premium_rate
        self.call_premium_rate = call_premium_rate
        self.holding_period = holding_period
        self.shares = shares

    def simulate(self, data: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
        cash = initial_capital
        state = "no_stock"  # possible states: "no_stock" or "stock_owned"
        entry_price = None
        portfolio_values = []
        dates = data.index
        i = 0
        while i < len(dates):
            price = data['Close'].iloc[i]
            if state == "no_stock":
                # Sell put option
                strike = price * (1 - self.put_offset)
                premium = price * self.put_premium_rate
                cash += premium * self.shares
                if i + self.holding_period < len(dates):
                    exp_price = data['Close'].iloc[i + self.holding_period]
                else:
                    exp_price = price
                if exp_price < strike:
                    cost = strike * self.shares
                    if cash >= cost:
                        cash -= cost
                        state = "stock_owned"
                        entry_price = strike
            elif state == "stock_owned":
                # Sell covered call
                strike = entry_price * (1 + self.call_offset)
                premium = price * self.call_premium_rate
                if i + self.holding_period < len(dates):
                    exp_price = data['Close'].iloc[i + self.holding_period]
                else:
                    exp_price = price
                cash += premium * self.shares
                if exp_price > strike:
                    cash += strike * self.shares
                    state = "no_stock"
                    entry_price = None
            portfolio_value = cash if state == "no_stock" else cash + data['Close'].iloc[i] * self.shares
            portfolio_values.append(portfolio_value)
            i += 1
        portfolio = pd.DataFrame(index=dates, data={'total': portfolio_values})
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

class CreditSpreadsStrategyOptions(Strategy):
    """
    A simplified simulation of a credit spreads strategy.
    Here we assume a put credit spread is sold on a weekly basis.
    Profit is the premium if the option expires worthless; otherwise, a loss equal to
    (spread_width - premium) is assumed.
    """
    def __init__(self, spread_width=5.0, spread_offset=0.05, premium_rate=0.02, holding_period=5):
        self.spread_width = spread_width
        self.spread_offset = spread_offset
        self.premium_rate = premium_rate
        self.holding_period = holding_period

    def simulate(self, data: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
        cash = initial_capital
        portfolio_values = []
        dates = data.index
        i = 0
        while i < len(dates):
            price = data['Close'].iloc[i]
            strike = price * (1 - self.spread_offset)
            premium = price * self.premium_rate
            if i + self.holding_period < len(dates):
                exp_price = data['Close'].iloc[i + self.holding_period]
            else:
                exp_price = price
            if exp_price > strike:
                profit = premium
            else:
                profit = - (self.spread_width - premium)
            cash += profit
            for j in range(self.holding_period):
                portfolio_values.append(cash)
            i += self.holding_period
        portfolio_values = portfolio_values[:len(dates)]
        portfolio = pd.DataFrame(index=dates, data={'total': portfolio_values})
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

class IronCondorsStrategyOptions(Strategy):
    """
    A simplified simulation of an Iron Condor strategy.
    Two credit spreads (one put and one call) are sold.
    If the underlying remains between the inner strikes at expiration,
    the full premium is kept; otherwise, a maximum loss is assumed.
    """
    def __init__(self, lower_put_offset=0.05, upper_call_offset=0.05,
                 inner_put_offset=0.02, inner_call_offset=0.02,
                 premium_rate=0.02, holding_period=5):
        self.lower_put_offset = lower_put_offset
        self.upper_call_offset = upper_call_offset
        self.inner_put_offset = inner_put_offset
        self.inner_call_offset = inner_call_offset
        self.premium_rate = premium_rate
        self.holding_period = holding_period

    def simulate(self, data: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
        cash = initial_capital
        portfolio_values = []
        dates = data.index
        i = 0
        while i < len(dates):
            price = data['Close'].iloc[i]
            lower_inner = price * (1 - self.inner_put_offset)
            upper_inner = price * (1 + self.inner_call_offset)
            lower_outer = price * (1 - self.lower_put_offset)
            upper_outer = price * (1 + self.upper_call_offset)
            premium = price * self.premium_rate
            if i + self.holding_period < len(dates):
                exp_price = data['Close'].iloc[i + self.holding_period]
            else:
                exp_price = price
            if lower_inner <= exp_price <= upper_inner:
                profit = premium
            else:
                if exp_price < lower_inner:
                    loss = (lower_inner - lower_outer) - premium
                    profit = -loss
                elif exp_price > upper_inner:
                    loss = (upper_outer - upper_inner) - premium
                    profit = -loss
            cash += profit
            for j in range(self.holding_period):
                portfolio_values.append(cash)
            i += self.holding_period
        portfolio_values = portfolio_values[:len(dates)]
        portfolio = pd.DataFrame(index=dates, data={'total': portfolio_values})
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

# ------------------------------
# Stock Backtesting Engine with Custom Profit-Taking / Stop-Loss
# ------------------------------

class Backtester:
    """
    Simulates trades based on strategy-generated signals.
    Two simulation modes are provided:
      • run_backtest: vectorized simulation (no extra exit conditions)
      • run_backtest_custom: day-by-day simulation that checks for a profit target or stop-loss
    """
    def __init__(self, data: pd.DataFrame, signals: pd.DataFrame,
                 initial_capital: float = 100000.0, shares: int = 100):
        self.data = data
        self.signals = signals
        self.initial_capital = initial_capital
        self.shares = shares

    def run_backtest(self) -> pd.DataFrame:
        # Build positions DataFrame: number of shares held
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions['positions'] = self.shares * self.signals['signal']

        # Ensure that "Close" is a Series. If it is a DataFrame, squeeze it.
        close_prices = self.data['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.squeeze()

        # Calculate portfolio value over time
        portfolio = pd.DataFrame(index=self.signals.index)
        portfolio['holdings'] = positions['positions'] * close_prices
        pos_diff = positions['positions'].diff().fillna(0.0)
        portfolio['cash'] = self.initial_capital - (pos_diff * close_prices).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

    def run_backtest_custom(self, profit_target: float, stop_loss: float) -> pd.DataFrame:
        position = 0
        entry_price = None
        cash = self.initial_capital
        total_values = []
        positions_series = []
        for date, price in self.data['Close'].items():
            signal = self.signals['signal'].loc[date]
            if position == 0:
                if signal == 1:
                    position = self.shares
                    entry_price = price
                    cash -= price * self.shares
            else:
                if price >= entry_price * (1 + profit_target) or price <= entry_price * (1 - stop_loss):
                    cash += price * self.shares
                    position = 0
                    entry_price = None
                elif signal == 0:
                    cash += price * self.shares
                    position = 0
                    entry_price = None
            total = cash + position * price
            total_values.append(total)
            positions_series.append(position)
        portfolio = pd.DataFrame(index=self.data.index)
        portfolio['total'] = total_values
        portfolio['positions'] = positions_series
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

# ------------------------------
# Visualization
# ------------------------------

def plot_results(portfolio: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio.index, portfolio['total'], label="Portfolio Total Value")
    ax.set_title("Portfolio Performance")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Value")
    ax.legend()
    ax.grid(True)
    return fig

# ------------------------------
# Streamlit Web App
# ------------------------------

def main():
    st.title("QuantBacktest Web App")
    st.markdown("A quantitative backtesting platform for stock and options trading strategies.")

    # Sidebar – Backtest settings
    st.sidebar.header("Backtest Settings")
    ticker = st.sidebar.text_input("Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime(2021, 1, 1))
    initial_capital = st.sidebar.number_input("Initial Capital", value=100000.0)
    shares = st.sidebar.number_input("Number of Shares", value=100, step=1)

    strategy_options = ["SMA Crossover", "RSI Trading", "Bollinger Bands",
                        "Custom Profit/Stop", "Second Derivative MA",
                        "Wheel Strategy", "Credit Spreads", "Iron Condors"]
    selected_strategy = st.sidebar.selectbox("Select Strategy", strategy_options)

    portfolio = None

    try:
        fetcher = DataFetcher(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        data = fetcher.fetch()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    # ------------------------------
    # Stock Strategies
    # ------------------------------
    if selected_strategy in ["SMA Crossover", "RSI Trading", "Bollinger Bands", "Custom Profit/Stop", "Second Derivative MA"]:
        if selected_strategy in ["SMA Crossover", "Custom Profit/Stop"]:
            st.sidebar.subheader("SMA Parameters")
            sma_short = st.sidebar.slider("Short Window", min_value=5, max_value=100, value=50)
            sma_long = st.sidebar.slider("Long Window", min_value=20, max_value=300, value=200)
            strategy = SMACrossoverStrategy(short_window=sma_short, long_window=sma_long)
        elif selected_strategy == "RSI Trading":
            st.sidebar.subheader("RSI Parameters")
            rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=30, value=14)
            oversold = st.sidebar.slider("Oversold Threshold", min_value=10, max_value=50, value=30)
            overbought = st.sidebar.slider("Overbought Threshold", min_value=50, max_value=90, value=70)
            strategy = RSITradingStrategy(period=rsi_period, oversold=oversold, overbought=overbought)
        elif selected_strategy == "Bollinger Bands":
            st.sidebar.subheader("Bollinger Bands Parameters")
            bb_window = st.sidebar.slider("Window", min_value=10, max_value=100, value=20)
            bb_std_multiplier = st.sidebar.slider("Std Dev Multiplier", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
            strategy = BollingerBandsStrategy(window=bb_window, std_multiplier=bb_std_multiplier)
        elif selected_strategy == "Second Derivative MA":
            st.sidebar.subheader("Second Derivative MA Parameters")
            sd_ma_window = st.sidebar.slider("MA Window", min_value=5, max_value=100, value=50)
            sd_threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=10.0, value=0.1, step=0.1)
            strategy = SecondDerivativeMAStrategy(ma_window=sd_ma_window, threshold=sd_threshold)

        signals = strategy.generate_signals(data)

        use_custom = False
        if selected_strategy == "Custom Profit/Stop":
            use_custom = True
            st.sidebar.subheader("Profit-taking / Stop-loss Settings")
            profit_target = st.sidebar.slider("Profit Target (%)", min_value=1, max_value=50, value=10) / 100.0
            stop_loss = st.sidebar.slider("Stop Loss (%)", min_value=1, max_value=50, value=5) / 100.0

        if st.sidebar.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                backtester = Backtester(data, signals, initial_capital, shares)
                if use_custom:
                    portfolio = backtester.run_backtest_custom(profit_target, stop_loss)
                else:
                    portfolio = backtester.run_backtest()

    # ------------------------------
    # Options Strategies
    # ------------------------------
    elif selected_strategy == "Wheel Strategy":
        st.sidebar.subheader("Wheel Strategy Parameters")
        put_offset = st.sidebar.slider("Put Offset (%)", min_value=1, max_value=10, value=5) / 100.0
        call_offset = st.sidebar.slider("Call Offset (%)", min_value=1, max_value=10, value=5) / 100.0
        put_premium_rate = st.sidebar.slider("Put Premium Rate (%)", min_value=1, max_value=10, value=2) / 100.0
        call_premium_rate = st.sidebar.slider("Call Premium Rate (%)", min_value=1, max_value=10, value=2) / 100.0
        holding_period = st.sidebar.slider("Holding Period (days)", min_value=1, max_value=30, value=5)
        strategy = WheelStrategyOptions(put_offset, call_offset, put_premium_rate,
                                        call_premium_rate, holding_period, shares)
        if st.sidebar.button("Run Backtest"):
            with st.spinner("Running Wheel Strategy simulation..."):
                portfolio = strategy.simulate(data, initial_capital)
    elif selected_strategy == "Credit Spreads":
        st.sidebar.subheader("Credit Spreads Parameters")
        spread_width = st.sidebar.number_input("Spread Width", value=5.0)
        spread_offset = st.sidebar.slider("Spread Offset (%)", min_value=1, max_value=10, value=5) / 100.0
        premium_rate = st.sidebar.slider("Premium Rate (%)", min_value=1, max_value=10, value=2) / 100.0
        holding_period = st.sidebar.slider("Holding Period (days)", min_value=1, max_value=30, value=5)
        strategy = CreditSpreadsStrategyOptions(spread_width, spread_offset, premium_rate, holding_period)
        if st.sidebar.button("Run Backtest"):
            with st.spinner("Running Credit Spreads simulation..."):
                portfolio = strategy.simulate(data, initial_capital)
    elif selected_strategy == "Iron Condors":
        st.sidebar.subheader("Iron Condors Parameters")
        lower_put_offset = st.sidebar.slider("Lower Put Offset (%)", min_value=1, max_value=10, value=5) / 100.0
        upper_call_offset = st.sidebar.slider("Upper Call Offset (%)", min_value=1, max_value=10, value=5) / 100.0
        inner_put_offset = st.sidebar.slider("Inner Put Offset (%)", min_value=1, max_value=10, value=2) / 100.0
        inner_call_offset = st.sidebar.slider("Inner Call Offset (%)", min_value=1, max_value=10, value=2) / 100.0
        premium_rate = st.sidebar.slider("Premium Rate (%)", min_value=1, max_value=10, value=2) / 100.0
        holding_period = st.sidebar.slider("Holding Period (days)", min_value=1, max_value=30, value=5)
        strategy = IronCondorsStrategyOptions(lower_put_offset, upper_call_offset, inner_put_offset,
                                               inner_call_offset, premium_rate, holding_period)
        if st.sidebar.button("Run Backtest"):
            with st.spinner("Running Iron Condors simulation..."):
                portfolio = strategy.simulate(data, initial_capital)

    # ------------------------------
    # Results & Visualization
    # ------------------------------
    if portfolio is not None:
        st.subheader("Performance Summary")
        total_return = (portfolio['total'].iloc[-1] - portfolio['total'].iloc[0]) / portfolio['total'].iloc[0]
        days = (portfolio.index[-1] - portfolio.index[0]).days
        annual_return = (1 + total_return) ** (365.0/days) - 1 if days > 0 else np.nan
        volatility = portfolio['returns'].std() * np.sqrt(252)
        max_drawdown = (portfolio['total'].cummax() - portfolio['total']).max()
        sharpe = (np.sqrt(252) * portfolio['returns'].mean() /
                  (portfolio['returns'].std() + 1e-9))
        st.write({
            "Total Return (%)": f"{total_return * 100:.2f}%",
            "Annualized Return (%)": f"{annual_return * 100:.2f}%",
            "Volatility (%)": f"{volatility * 100:.2f}%",
            "Max Drawdown (%)": f"{max_drawdown * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}"
        })
        st.subheader("Portfolio Performance")
        fig = plot_results(portfolio)
        st.pyplot(fig)

if __name__ == "__main__":
    main()

