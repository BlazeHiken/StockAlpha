import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import os
import sys
import plotly.graph_objects as go

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
            
            st.markdown("### 📊 Portfolio Backtest & Analytics")
            st.markdown("""
            **Methodology Note:** To prevent lookahead bias, the optimizer was strictly trained on the **first 80%** of the historical data (In-Sample). 
            Those learned weights were then frozen and applied to the **final 20%** of the data (Out-Of-Sample) to evaluate true performance.
            """)
            
            # --- Row 1: Benchmarks ---
            st.subheader("Performance Comparison")
            
            # Format the dataframe for display
            display_df = analysis.benchmark_results.comparison_df.copy()
            display_df = display_df.astype(object)

            percentage_rows = [
                "IS Return",
                "OOS Return",
                "IS Volatility",
                "OOS Volatility",
                "IS Max Drawdown",
                "OOS Max Drawdown",
            ]

            sharpe_rows = [
                "IS Sharpe",
                "OOS Sharpe",
            ]

            for row in percentage_rows:
                display_df.loc[row] = display_df.loc[row].map(
                    lambda x: f"{float(x) * 100:.2f}%"
                )

            for row in sharpe_rows:
                display_df.loc[row] = display_df.loc[row].map(
                    lambda x: f"{float(x):.3f}"
                )
            
            st.dataframe(display_df, use_container_width=True)
            
            # --- Row 2: Charts ---
            st.divider()
            st.subheader("Historical Backtest")
            
            import plotly.express as px
            from src.quant.metrics import calculate_cumulative_returns, calculate_annualized_returns, calculate_annualized_volatility
            
            full_returns = pd.concat([analysis.train_returns, analysis.test_returns])
            tickers = list(full_returns.columns)
            
            opt_weights_arr = np.array([analysis.opt_result.weights.get(t, 0) for t in tickers])
            eq_weights_arr = np.array([analysis.benchmark_results.equal_weight.weights.get(t, 0) for t in tickers])
            
            opt_returns = (full_returns * opt_weights_arr).sum(axis=1)
            eq_returns = (full_returns * eq_weights_arr).sum(axis=1)
            
            chart_df = pd.DataFrame({
                "Optimal (Max Sharpe)": calculate_cumulative_returns(opt_returns),
                "Naive (Equal Weight)": calculate_cumulative_returns(eq_returns)
            })
            
            if analysis.market_train is not None and analysis.market_test is not None:
                market_returns_full = pd.concat([analysis.market_train, analysis.market_test])
                chart_df["Market (NIFTY 50)"] = calculate_cumulative_returns(market_returns_full)
            
            fig_bt = px.line(chart_df, labels={"value": "Cumulative Wealth Index", "index": "Date"}, title="Cumulative Returns")
            split_date = analysis.test_returns.index[0]
            fig_bt.add_vline(x=split_date, line_dash="dash", line_color="red", annotation_text="Out-of-Sample Test →", annotation_position="top right")
            st.plotly_chart(fig_bt, use_container_width=True)
            
            # --- Row 3: Heatmap and Allocation ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Optimal Allocation")
                # Format weights for pie chart
                weights_df = pd.DataFrame({
                    "Ticker": list(analysis.opt_result.weights.keys()),
                    "Weight": list(analysis.opt_result.weights.values())
                })
                weights_df = weights_df[weights_df["Weight"] > 0.005]
                
                donut_chart = alt.Chart(weights_df).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="Weight", type="quantitative"),
                    color=alt.Color(field="Ticker", type="nominal"),
                    tooltip=['Ticker', alt.Tooltip('Weight', format='.1%')]
                ).properties(height=350)
                
                st.altair_chart(donut_chart, use_container_width=True)
                st.caption(f"Total Weight Sum: {analysis.weight_sum * 100:.2f}%")
                
            with col2:
                st.subheader("Asset Correlation")
                fig_corr = px.imshow(analysis.correlation_df, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
                st.plotly_chart(fig_corr, use_container_width=True)

            # --- Row 4: Risk Return Scatter ---
            st.divider()
            st.subheader("Risk-Return Profile (In-Sample)")
            
            # Scatter for individual assets
            ind_ann_ret = calculate_annualized_returns(analysis.train_returns)
            ind_ann_vol = calculate_annualized_volatility(analysis.train_returns)
            
            scatter_data = []
            for t in tickers:
                scatter_data.append({"Asset": t, "Return": ind_ann_ret[t], "Volatility": ind_ann_vol[t], "Type": "Individual Asset"})
            
            # Add Portfolios
            optimal_metrics = analysis.opt_result.in_sample_metrics
            equal_metrics = analysis.benchmark_results.equal_weight.in_sample_metrics

            if optimal_metrics is not None:
                scatter_data.append({
                    "Asset": "Optimal",
                    "Return": optimal_metrics.expected_return,
                    "Volatility": optimal_metrics.volatility,
                    "Type": "Portfolio"
                })

            if equal_metrics is not None:
                scatter_data.append({
                    "Asset": "Equal Weight",
                    "Return": equal_metrics.expected_return,
                    "Volatility": equal_metrics.volatility,
                    "Type": "Portfolio"
                })

            market_result = analysis.benchmark_results.market_benchmark

            if market_result is not None and market_result.in_sample_metrics is not None:
                market_metrics = market_result.in_sample_metrics

                scatter_data.append({
                    "Asset": "NIFTY 50",
                    "Return": market_metrics.expected_return,
                    "Volatility": market_metrics.volatility,
                    "Type": "Market"
                })  
            scatter_df = pd.DataFrame(scatter_data)
            
            fig_scatter = px.scatter(
                scatter_df, x="Volatility", y="Return", color="Type", text="Asset", 
                title="Risk vs Expected Return (Train Period)",
                labels={"Return": "Expected Return", "Volatility": "Risk (Volatility)"}
            )
            fig_scatter.update_traces(textposition="top center")
            
            # Add frontier line
            fig_scatter.add_trace(
                go.Scatter(
                    x=analysis.frontier_df["volatility"],
                    y=analysis.frontier_df["return"],
                    mode="lines",
                    name="Efficient Frontier",
                    line=dict(
                        dash="dash",
                        width=2
                    )
                )
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
