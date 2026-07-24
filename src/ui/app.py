import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import DEFAULT_TICKERS, DEFAULT_RISK_FREE_RATE, MIN_STOCKS, LOOKBACK_YEARS
from src.services.portfolio_service import run_portfolio_analysis

# --- Configuration & Defaults ---
st.set_page_config(page_title="StockAlpha Portfolio Optimizer", page_icon="📈", layout="wide")
if "ticker_options" not in st.session_state:
    st.session_state.ticker_options = DEFAULT_TICKERS.copy()

if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = DEFAULT_TICKERS[:5]

# --- UI Sidebar ---
def add_ticker_callback():
    ticker = st.session_state.custom_ticker_input.strip().upper()

    if not ticker:
        st.session_state.ticker_add_message = ("warning", "Please enter a ticker.")
    elif ticker in st.session_state.ticker_options:
        st.session_state.ticker_add_message = ("info", f"{ticker} already exists.")
    else:
        st.session_state.ticker_options.append(ticker)
        st.session_state.selected_tickers.append(ticker)
        st.session_state.ticker_add_message = ("success", f"{ticker} added.")
    
    # Clear the input box after adding
    st.session_state.custom_ticker_input = ""


if "ticker_options" not in st.session_state:
    st.session_state.ticker_options = DEFAULT_TICKERS.copy()

if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = DEFAULT_TICKERS[:5]

st.sidebar.title("StockAlpha Configuration")
st.sidebar.markdown("Configure your portfolio parameters.")

selected_tickers = st.sidebar.multiselect(
    "Select Stocks (NSE Tickers)",
    options=st.session_state.ticker_options,
    key="selected_tickers",
    help="Add or remove stock tickers to include in the optimization."
)

st.sidebar.text_input(
    "Add Custom Ticker (e.g. TATAMOTORS.NS)",
    key="custom_ticker_input"
)

st.sidebar.button("Add Ticker", on_click=add_ticker_callback)

# Show the message AFTER rerun, since toast/warning calls before rerun get wiped
if "ticker_add_message" in st.session_state:
    level, msg = st.session_state.ticker_add_message
    getattr(st.sidebar, level)(msg)
    del st.session_state.ticker_add_message


end_date = st.sidebar.date_input("End Date", datetime.today())
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=LOOKBACK_YEARS*365))

run_btn = st.sidebar.button("Run Optimization", type="primary")

# --- Main App ---
st.title("📈 StockAlpha Portfolio Optimizer")
st.markdown("""
This tool uses Modern Portfolio Theory (MPT) to find the mathematically optimal allocation 
for your selected stocks, maximizing the **Sharpe Ratio** (Return vs. Risk).
""")

if run_btn:
    if len(selected_tickers) < MIN_STOCKS:
        st.error(f"Please select at least {MIN_STOCKS} stocks to run the optimization.")
    else:
        with st.spinner(f"Fetching data and calculating optimal allocation for {len(selected_tickers)} stocks..."):
            
            # Run entire pipeline via service layer
            analysis = run_portfolio_analysis(
                tickers=selected_tickers, 
                start_date=start_date.strftime('%Y-%m-%d'), 
                end_date=end_date.strftime('%Y-%m-%d'),
                risk_free_rate=DEFAULT_RISK_FREE_RATE
            )
            
        if not analysis:
            st.error("Failed to fetch enough valid data. Please check your tickers.")
        else:
            if not analysis.opt_result.success:
                st.warning(f"Optimizer did not converge perfectly, results may be sub-optimal. Reason: {analysis.opt_result.error_message}")
                
            # Verify weights sum to 100%
            if not np.isclose(analysis.weight_sum, 1.0, atol=1e-4):
                st.warning(f"Warning: Optimal weights sum to {analysis.weight_sum*100:.2f}% instead of 100%.")
                
            # --- Display Results ---
            st.divider()
            
            # --- Row 1: Benchmarks ---
            st.subheader("Performance Comparison")
            
            # Format the dataframe for display
            display_df = analysis.benchmark_results.comparison_df.copy()
            display_df = display_df.astype(object)

            display_df.loc["Return"] = (display_df.loc["Return"] * 100).map("{:.2f}%".format)
            display_df.loc["Volatility"] = (display_df.loc["Volatility"] * 100).map("{:.2f}%".format)
            display_df.loc["Max Drawdown"] = (display_df.loc["Max Drawdown"] * 100).map("{:.2f}%".format)
            display_df.loc["Sharpe Ratio"] = display_df.loc["Sharpe Ratio"].map("{:.3f}".format)
            
            st.dataframe(display_df, use_container_width=True)
            
            # --- Row 2: Charts ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Optimal Allocation")
                # Format weights for pie chart
                weights_df = pd.DataFrame({
                    "Ticker": list(analysis.opt_result.weights.keys()),
                    "Weight": list(analysis.opt_result.weights.values())
                })
                # Filter out near-zero weights for cleaner display
                weights_df = weights_df[weights_df["Weight"] > 0.005]
                
                # Altair Donut Chart
                donut_chart = alt.Chart(weights_df).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="Weight", type="quantitative"),
                    color=alt.Color(field="Ticker", type="nominal"),
                    tooltip=['Ticker', alt.Tooltip('Weight', format='.1%')]
                ).properties(height=350)
                
                st.altair_chart(donut_chart, use_container_width=True)
                st.caption(f"Total Weight Sum: {analysis.weight_sum * 100:.2f}%")

            with col2:
                st.subheader("Efficient Frontier")
                
                # Base line for efficient frontier
                frontier_line = alt.Chart(analysis.frontier_df).mark_line(color="gray", strokeDash=[5,5]).encode(
                    x=alt.X("volatility:Q", title="Volatility (Risk)", axis=alt.Axis(format='%')),
                    y=alt.Y("return:Q", title="Expected Return", axis=alt.Axis(format='%')),
                    tooltip=[alt.Tooltip('return:Q', format='.2%'), alt.Tooltip('volatility:Q', format='.2%')]
                )
                
                # Point for Optimal Portfolio
                opt_point = pd.DataFrame([{
                    "volatility": analysis.opt_result.metrics.volatility if analysis.opt_result.metrics else 0,
                    "return": analysis.opt_result.metrics.expected_return if analysis.opt_result.metrics else 0,
                    "Label": "Max Sharpe (Optimal)"
                }])
                
                opt_chart = alt.Chart(opt_point).mark_circle(size=150, color="green").encode(
                    x="volatility:Q",
                    y="return:Q",
                    tooltip=["Label", alt.Tooltip('return:Q', format='.2%'), alt.Tooltip('volatility:Q', format='.2%')]
                )
                
                # Point for Equal Weight Portfolio
                eq_point = pd.DataFrame([{
                    "volatility": analysis.benchmark_results.equal_weight.metrics.volatility if analysis.benchmark_results.equal_weight.metrics else 0,
                    "return": analysis.benchmark_results.equal_weight.metrics.expected_return if analysis.benchmark_results.equal_weight.metrics else 0,
                    "Label": "Equal Weight (Naive)"
                }])
                
                eq_chart = alt.Chart(eq_point).mark_circle(size=150, color="red").encode(
                    x="volatility:Q",
                    y="return:Q",
                    tooltip=["Label", alt.Tooltip('return:Q', format='.2%'), alt.Tooltip('volatility:Q', format='.2%')]
                )
                
                combined_chart = (frontier_line + opt_chart + eq_chart).properties(height=350)
                st.altair_chart(combined_chart, use_container_width=True)
