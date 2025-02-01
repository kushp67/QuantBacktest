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
# Options Strategies (Simplified Simulation with Actual Options Data)
# ------------------------------

class WheelStrategyOptions(Strategy):
    """
    Wheel Strategy (Options) – now using actual options chain data.
    
    When not holding stock, the strategy sells a cash‑secured put.
    It fetches the current options chain for the chosen ticker and expiration date,
    selects a put whose strike is close to a target (e.g. price*(1 - put_offset)),
    and uses its actual last price as the premium.
    
    If assigned (i.e. if at expiration the underlying is below the option's strike),
    the strategy buys stock (at the strike) and then sells a covered call using the
    current call options chain data.
    
    Note: This implementation uses current options data from yfinance.
    """
    def __init__(self, put_offset=0.05, call_offset=0.05, holding_period=5, shares=100):
        self.put_offset = put_offset
        self.call_offset = call_offset
        self.holding_period = holding_period
        self.shares = shares

    def simulate(self, data: pd.DataFrame, ticker: str, initial_capital: float) -> pd.DataFrame:
        # Retrieve current options chain for a chosen expiration date
        tkr = yf.Ticker(ticker)
        option_dates = tkr.options
        if not option_dates:
            raise ValueError("No options data available for ticker.")
        exp_date = option_dates[0]  # For demonstration, choose the earliest expiration
        chain = tkr.option_chain(exp_date)
        puts = chain.puts
        calls = chain.calls

        cash = initial_capital
        state = "no_stock"  # "no_stock" or "stock_owned"
        entry_price = None
        portfolio_values = []
        dates = data.index
        i = 0
        while i < len(dates):
            price = float(data['Close'].iloc[i])
            if state == "no_stock":
                # Sell put: target strike is price*(1 - put_offset)
                desired_strike = price * (1 - self.put_offset)
                idx = (puts['strike'] - desired_strike).abs().idxmin()
                chosen_put = puts.loc[idx]
                premium = float(chosen_put['lastPrice'])
                cash += premium * self.shares
                # Determine expiration outcome using the underlying's price after holding_period days
                if i + self.holding_period < len(dates):
                    exp_price = float(data['Close'].iloc[i + self.holding_period])
                else:
                    exp_price = price
                # If the underlying is below the option's strike, assignment occurs
                if exp_price < chosen_put['strike']:
                    cost = chosen_put['strike'] * self.shares
                    if cash >= cost:
                        cash -= cost
                        state = "stock_owned"
                        entry_price = chosen_put['strike']
            elif state == "stock_owned":
                # Sell covered call: target strike is entry_price*(1 + call_offset)
                desired_strike = entry_price * (1 + self.call_offset)
                idx = (calls['strike'] - desired_strike).abs().idxmin()
                chosen_call = calls.loc[idx]
                premium_call = float(chosen_call['lastPrice'])
                if i + self.holding_period < len(dates):
                    exp_price = float(data['Close'].iloc[i + self.holding_period])
                else:
                    exp_price = price
                cash += premium_call * self.shares
                # If the underlying is above the call's strike at expiration, assignment occurs
                if exp_price > chosen_call['strike']:
                    cash += chosen_call['strike'] * self.shares
                    state = "no_stock"
                    entry_price = None
            portfolio_value = cash if state == "no_stock" else cash + float(data['Close'].iloc[i]) * self.shares
            portfolio_values.append(portfolio_value)
            i += 1
        portfolio = pd.DataFrame(index=dates, data={'total': portfolio_values})
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

class CreditSpreadsStrategyOptions(Strategy):
    """
    Credit Spreads Strategy (Options) – using actual options data.
    
    This simplified simulation assumes a put credit spread is sold.
    It fetches the current puts chain, selects a put near the target strike,
    and uses its actual last price as the premium.
    
    The profit is the premium if the option expires worthless; otherwise,
    a loss equal to (spread_width - premium) is assumed.
    """
    def __init__(self, spread_width=5.0, spread_offset=0.05, holding_period=5):
        self.spread_width = spread_width
        self.spread_offset = spread_offset
        self.holding_period = holding_period

    def simulate(self, data: pd.DataFrame, ticker: str, initial_capital: float) -> pd.DataFrame:
        tkr = yf.Ticker(ticker)
        option_dates = tkr.options
        if not option_dates:
            raise ValueError("No options data available for ticker.")
        exp_date = option_dates[0]
        chain = tkr.option_chain(exp_date)
        puts = chain.puts

        cash = initial_capital
        portfolio_values = []
        dates = data.index
        i = 0
        while i < len(dates):
            price = float(data['Close'].iloc[i])
            desired_strike = price * (1 - self.spread_offset)
            idx = (puts['strike'] - desired_strike).abs().idxmin()
            chosen_put = puts.loc[idx]
            premium = float(chosen_put['lastPrice'])
            if i + self.holding_period < len(dates):
                exp_price = float(data['Close'].iloc[i + self.holding_period])
            else:
                exp_price = price
            if exp_price > chosen_put['strike']:
                profit = premium
            else:
                profit = - (self.spread_width - premium)
            cash += profit
            # Fill in the portfolio for holding_period days
            for j in range(self.holding_period):
                portfolio_values.append(cash)
            i += self.holding_period
        portfolio_values = portfolio_values[:len(dates)]
        portfolio = pd.DataFrame(index=dates, data={'total': portfolio_values})
        portfolio['returns'] = portfolio['total'].pct_change().fillna(0.0)
        return portfolio

class IronCondorsStrategyOptions(Strategy):
    """
    Iron Condors Strategy (Options) – using actual options data.
    
    This simplified simulation sells both a put spread and a call spread.
    It fetches the current puts and calls chains, selects options near the target inner strikes,
    and uses their actual last prices as premiums.
    
    If the underlying remains between the inner strikes at expiration, full premium is kept;
    otherwise, a loss is incurred approximated by the difference between outer and inner strikes minus premium.
    """
    def __init__(self, lower_put_offset=0.05, upper_call_offset=0.05,
                 inner_put_offset=0.02, inner_call_offset=0.02, holding_period=5):
        self.lower_put_offset = lower_put_offset
        self.upper_call_offset = upper_call_offset
        self.inner_put_offset = inner_put_offset
        self.inner_call_offset = inner_call_offset
        self.holding_period = holding_period

    def simulate(self, data: pd.DataFrame, ticker: str, initial_capital: float) -> pd.DataFrame:
        tkr = yf.Ticker(ticker)
        option_dates = tkr.options
        if not option_dates:
            raise ValueError("No options data available for ticker.")
        exp_date = option_dates[0]
        chain = tkr.option_chain(exp_date)
        puts = chain.puts
        calls = chain.calls

        cash = initial_capital
        portfolio_values = []
        dates = data.index
        i = 0
        while i < len(dates):
            price = float(data['Close'].iloc[i])
            inner_put_target = price * (1 - self.inner_put_offset)
            inner_call_target = price * (1 + self.inner_call_offset)
            outer_put_target = price * (1 - self.lower_put_offset)
            outer_call_target = price * (1 + self.upper_call_offset)
            
            idx_put = (puts['strike'] - inner_put_target).abs().idxmin()
            chosen_put = puts.loc[idx_put]
            idx_call = (calls['strike'] - inner_call_target).abs().idxmin()
            chosen_call = calls.loc[idx_call]
            
            premium_put = float(chosen_put['lastPrice'])
            premium_call = float(chosen_call['lastPrice'])
            total_premium = premium_put + premium_call
            
            if i + self.holding_period < len(dates):
                exp_price = float(data['Close'].iloc[i + self.holding_period])
            else:
                exp_price = price
            
            if (exp_price >= chosen_put['strike']) and (exp_price <= chosen_call['strike']):
                profit = total_premium
            else:
                loss_put = (chosen_put['strike'] - outer_put_target) if exp_price < chosen_put['strike'] else 0
                loss_call = (outer_call_target - chosen_call['strike']) if exp_price > chosen_call['strike'] else 0
                profit = total_premium - (loss_put + loss_call)
            
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
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions['positions'] = self.shares * self.signals['signal']
        close_prices = self.data['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.squeeze()
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
# Visualization Functions
# ------------------------------

def plot_results(portfolio: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio.index, portfolio['total'], label="Strategy Portfolio")
    ax.set_title("Portfolio Performance")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Value")
    ax.legend()
    ax.grid(True)
    return fig

def plot_buy_hold_comparison(portfolio: pd.DataFrame, data: pd.DataFrame, initial_capital: float):
    """
    Computes a buy and hold strategy and plots its portfolio value alongside the strategy portfolio.
    Buy and hold: invest the entire initial capital at the first close.
    """
    # Calculate buy and hold portfolio value
    bh_shares = initial_capital / data['Close'].iloc[0]
    buy_hold = bh_shares * data['Close']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio.index, portfolio['total'], label="Strategy Portfolio")
    ax.plot(data.index, buy_hold, label="Buy & Hold", linestyle='--')
    ax.set_title("Strategy vs. Buy & Hold Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
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
        holding_period = st.sidebar.slider("Holding Period (days)", min_value=1, max_value=30, value=5)
        strategy = WheelStrategyOptions(put_offset, call_offset, holding_period=holding_period, shares=shares)
        if st.sidebar.button("Run Backtest"):
            with st.spinner("Running Wheel Strategy simulation..."):
                portfolio = strategy.simulate(data, ticker, initial_capital)
    elif selected_strategy == "Credit Spreads":
        st.sidebar.subheader("Credit Spreads Parameters")
        spread_width = st.sidebar.number_input("Spread Width", value=5.0)
        spread_offset = st.sidebar.slider("Spread Offset (%)", min_value=1, max_value=10, value=5) / 100.0
        holding_period = st.sidebar.slider("Holding Period (days)", min_value=1, max_value=30, value=5)
        strategy = CreditSpreadsStrategyOptions(spread_width, spread_offset, holding_period=holding_period)
        if st.sidebar.button("Run Backtest"):
            with st.spinner("Running Credit Spreads simulation..."):
                portfolio = strategy.simulate(data, ticker, initial_capital)
    elif selected_strategy == "Iron Condors":
        st.sidebar.subheader("Iron Condors Parameters")
        lower_put_offset = st.sidebar.slider("Lower Put Offset (%)", min_value=1, max_value=10, value=5) / 100.0
        upper_call_offset = st.sidebar.slider("Upper Call Offset (%)", min_value=1, max_value=10, value=5) / 100.0
        inner_put_offset = st.sidebar.slider("Inner Put Offset (%)", min_value=1, max_value=10, value=2) / 100.0
        inner_call_offset = st.sidebar.slider("Inner Call Offset (%)", min_value=1, max_value=10, value=2) / 100.0
        holding_period = st.sidebar.slider("Holding Period (days)", min_value=1, max_value=30, value=5)
        strategy = IronCondorsStrategyOptions(lower_put_offset, upper_call_offset, inner_put_offset,
                                               inner_call_offset, holding_period=holding_period)
        if st.sidebar.button("Run Backtest"):
            with st.spinner("Running Iron Condors simulation..."):
                portfolio = strategy.simulate(data, ticker, initial_capital)

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
        fig1 = plot_results(portfolio)
        st.pyplot(fig1)
        st.subheader("Strategy vs. Buy & Hold Comparison")
        fig2 = plot_buy_hold_comparison(portfolio, data, initial_capital)
        st.pyplot(fig2)

if __name__ == "__main__":
    main()
