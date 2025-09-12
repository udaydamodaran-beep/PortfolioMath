import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from datetime import datetime
import plotly.express as px
from io import BytesIO

# -------------------------------
# Config & Setup
# -------------------------------
st.set_page_config(page_title="NIFTY50 Explorer", page_icon="📈", layout="wide")

st.title("📊 NIFTY 50 Stock Metrics Dashboard")
st.markdown("Explore historical returns, volatility, beta and R² of NIFTY 50 stocks vs the NIFTY index.")

# Subtle credit line
st.markdown("<p style='font-size:0.8em; color:gray;'>Developed by Uday Damodaran for Pedagogical Purposes</p>", unsafe_allow_html=True)

# Metric explanations
with st.expander("ℹ️ What do these metrics mean?"):
    st.markdown("""
    - **Return (1Y, 2Y, 5Y %)**: The annualized percentage return over the past 1, 2, or 5 years.
    - **Volatility %**: The standard deviation of daily returns annualized (higher = more risk/uncertainty).
    - **Beta**: Measures how sensitive the stock is to NIFTY movements. 
        - Beta > 1 → stock moves more than the index.
        - Beta < 1 → stock moves less than the index.
    - **R²**: The goodness-of-fit of the regression with NIFTY. Closer to 1 means the index explains most of the stock’s moves.
    """)

# List of tickers (NSE Yahoo Finance format)
tickers = [
 "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS",
 "BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS","BEL.NS","BHARTIARTL.NS",
 "BRITANNIA.NS","CIPLA.NS","COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS",
 "EICHERMOT.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS",
 "HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","INDUSINDBK.NS",
 "INFY.NS","ITC.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS",
 "M&M.NS","MARUTI.NS","NESTLEIND.NS","NTPC.NS","ONGC.NS",
 "POWERGRID.NS","RELIANCE.NS","SBIN.NS","SHREECEM.NS","SUNPHARMA.NS",
 "TCS.NS","TATAMOTORS.NS","TATASTEEL.NS","TECHM.NS","TITAN.NS",
 "ULTRATECH.NS","UPL.NS","WIPRO.NS"
]

index_ticker = "^NSEI"  # NIFTY index

# -------------------------------
# Helper functions
# -------------------------------
def _first_numeric_series(df_or_series):
    if isinstance(df_or_series, pd.Series):
        return df_or_series
    if isinstance(df_or_series, pd.DataFrame):
        cols = df_or_series.columns
        if isinstance(cols, pd.MultiIndex):
            for col in cols:
                if 'close' in str(col).lower():
                    return df_or_series[col]
            for col in cols:
                if pd.api.types.is_numeric_dtype(df_or_series[col]):
                    return df_or_series[col]
        else:
            if 'Close' in df_or_series.columns:
                return df_or_series['Close']
            for col in df_or_series.columns:
                if pd.api.types.is_numeric_dtype(df_or_series[col]):
                    return df_or_series[col]
    return pd.Series(dtype=float)


def get_adj_close(ticker, start):
    data = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if data is None or (hasattr(data, 'empty') and data.empty):
        return pd.Series(dtype=float)
    return _first_numeric_series(data)


def to_excel_bytes(df):
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

@st.cache_data(show_spinner=True)
def compute_metrics():
    today = pd.Timestamp.today().normalize()
    start_5y = today - pd.DateOffset(years=6)

    index_series = get_adj_close(index_ticker, start=start_5y).dropna()
    results = []

    for t in tickers:
        try:
            s = get_adj_close(t, start=start_5y)
            s = _first_numeric_series(s).dropna()
            if s.empty:
                continue

            r = np.log(s / s.shift(1)).dropna()
            idx_r = np.log(index_series / index_series.shift(1)).dropna()

            joined = pd.concat([r, idx_r], axis=1, join='inner').dropna()
            if joined.shape[1] < 2:
                raise ValueError("Insufficient overlapping returns with index")

            stock_col = joined.columns[0]
            index_col = joined.columns[1]
            joined = joined[[stock_col, index_col]]
            joined.columns = ['stock', 'index']

            ANN = 252
            vol_ann = joined['stock'].std() * np.sqrt(ANN)

            def ann_return(series, years):
                end = series.index.max()
                start = end - pd.DateOffset(years=years)
                series_window = series[series.index >= start]
                if series_window.empty:
                    return np.nan
                start_price = series_window.iloc[0]
                end_price = series.iloc[-1]
                if isinstance(start_price, (pd.Series, np.ndarray)):
                    start_price = np.asarray(start_price).flatten()[0]
                if isinstance(end_price, (pd.Series, np.ndarray)):
                    end_price = np.asarray(end_price).flatten()[0]
                if start_price == 0:
                    return np.nan
                return float((end_price / start_price)**(1/years) - 1)

            r_1y = ann_return(s, 1)
            r_2y = ann_return(s, 2)
            r_5y = ann_return(s, 5)

            Y = joined['stock']
            X = add_constant(joined['index'])
            model = OLS(Y, X).fit()
            beta = model.params.get('index', np.nan)
            r2 = model.rsquared

            results.append({
                "Ticker": t,
                "Return 1Y %": r_1y * 100 if not pd.isna(r_1y) else np.nan,
                "Return 2Y %": r_2y * 100 if not pd.isna(r_2y) else np.nan,
                "Return 5Y %": r_5y * 100 if not pd.isna(r_5y) else np.nan,
                "Volatility %": float(vol_ann * 100) if not pd.isna(vol_ann) else np.nan,
                "Beta": float(beta) if not pd.isna(beta) else np.nan,
                "R²": float(r2) if not pd.isna(r2) else np.nan
            })
        except Exception as e:
            st.warning(f"⚠️ Error processing {t}: {e}")
            continue

    df = pd.DataFrame(results)
    numeric_cols = ["Return 1Y %","Return 2Y %","Return 5Y %","Volatility %","Beta","R²"]
    if not df.empty:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df

# -------------------------------
# Main App
# -------------------------------
with st.spinner("Fetching data & computing metrics... This may take a minute."):
    df = compute_metrics()

st.success("Data loaded!")

st.sidebar.header("🔍 Filters")
search = st.sidebar.text_input("Search ticker (e.g. INFY)")
if search:
    df = df[df['Ticker'].str.contains(search.upper(), na=False)]

if df.empty:
    st.warning("No data available — try rerunning or check your internet connection to Yahoo Finance.")
else:
    st.dataframe(df.set_index("Ticker"), use_container_width=True)

# Download Buttons
st.subheader("⬇️ Download Data")
col1, col2 = st.columns(2)
if not df.empty:
    csv = df.to_csv(index=False).encode('utf-8')
    excel_bytes = to_excel_bytes(df)
    with col1:
        st.download_button("📥 Download as CSV", data=csv, file_name="nifty50_metrics.csv", mime="text/csv", use_container_width=True)
    with col2:
        st.download_button("📊 Download as Excel", data=excel_bytes, file_name="nifty50_metrics.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

# Plotting section
st.subheader("📈 Visualize Metrics")
numeric_cols = ["Return 1Y %","Return 2Y %","Return 5Y %","Volatility %
