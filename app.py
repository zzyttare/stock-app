import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from FinMind.data import DataLoader

# ========= 頁面設定 =========
st.set_page_config(page_title="台股技術分析 App", layout="wide")
st.title("台股技術分析 App")

# ========= Token =========
MY_TOKEN = os.getenv("FINMIND_TOKEN")

if not MY_TOKEN:
    st.error("請先設定環境變數 FINMIND_TOKEN")
    st.stop()

# ========= 初始化 =========
@st.cache_resource
def init_system():
    api = DataLoader()
    api.token = MY_TOKEN
    df = api.taiwan_stock_info()
    df = df[df["stock_id"].str.fullmatch(r"\d{4}")]
    market_map = {sid: ("上櫃" if int(sid) >= 4000 else "上市") for sid in df["stock_id"]}
    names = dict(zip(df["stock_id"], df["stock_name"]))
    return api, names, market_map

dl, stock_names, market_type_map = init_system()

# ========= 技術指標 =========
def get_indicators(df):
    if df is None or df.empty or len(df) < 40:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={
        "adj close": "close",
        "volume": "vol",
        "trading_volume": "vol"
    })

    required = {"open", "high", "low", "close", "vol"}
    if not required.issubset(df.columns):
        return None

    close = df["close"]
    vol = df["vol"]

    df["sma5"] = close.rolling(5).mean()
    df["sma10"] = close.rolling(10).mean()
    df["sma20"] = close.rolling(20).mean()
    df["sma60"] = close.rolling(60).mean()

    std20 = close.rolling(20).std()
    df["up"] = df["sma20"] + 2 * std20
    df["dn"] = df["sma20"] - 2 * std20
    df["bw"] = (df["up"] - df["dn"]) / df["sma20"]

    band_range = df["up"] - df["dn"]
    df["b_percent"] = np.where(
        band_range != 0,
        (close - df["dn"]) / band_range,
        np.nan
    )
    df["b_ma10"] = df["b_percent"].rolling(10).mean()

    df["ema12"] = close.ewm(span=12, adjust=False).mean()
    df["ema26"] = close.ewm(span=26, adjust=False).mean()
    df["dif"] = df["ema12"] - df["ema26"]
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()

    df["v_ma5"] = vol.shift(1).rolling(5).mean()
    df["v_ratio"] = vol / df["v_ma5"].replace(0, np.nan)

    df = df.copy()
    return df

# ========= 抓單股資料 =========
@st.cache_data(ttl=300)
def load_stock_data(stock_id: str, period: str):
    mt = market_type_map.get(stock_id, "上市" if int(stock_id) < 4000 else "上櫃")
    ticker = f"{stock_id}.TW" if mt == "上市" else f"{stock_id}.TWO"

    df_raw = yf.download(ticker, period=period, progress=False).reset_index()
    df = get_indicators(df_raw)
    if df is None:
        return None

    if "date" not in df.columns:
        if "datetime" in df.columns:
            df["date"] = df["datetime"]
        elif "index" in df.columns:
            df["date"] = df["index"]

    df["date"] = pd.to_datetime(df["date"])
    return df

# ========= 側邊欄 =========
with st.sidebar:
    st.header("查詢條件")

    stock_id = st.text_input("股票代號", value="2330").strip()

    period_map = {
        "6個月": "6mo",
        "1年": "1y",
        "2年": "2y"
    }
    period_label = st.selectbox("資料期間", list(period_map.keys()), index=2)
    period = period_map[period_label]

    indicators = st.multiselect(
        "選擇要顯示的指標",
        options=[
            "收盤價", "SMA5", "SMA10", "SMA20", "SMA60",
            "布林上軌", "布林下軌",
            "DIF", "DEA",
            "成交量"
        ],
        default=["收盤價", "SMA20", "布林上軌", "布林下軌", "成交量"]
    )

    do_query = st.button("開始查詢")

# ========= 查詢 =========
if do_query:
    if not stock_id.isdigit():
        st.warning("請輸入 4 碼股票代號")
        st.stop()

    with st.spinner("抓取資料中..."):
        df = load_stock_data(stock_id, period)

    if df is None or df.empty:
        st.error("查無資料或資料不足")
        st.stop()

    stock_name = stock_names.get(stock_id, "")
    st.subheader(f"{stock_id} {stock_name}")

    # ===== 1. 主價格圖 =====
    price_cols = []
    if "收盤價" in indicators:
        price_cols.append(("close", "收盤價"))
    if "SMA5" in indicators:
        price_cols.append(("sma5", "SMA5"))
    if "SMA10" in indicators:
        price_cols.append(("sma10", "SMA10"))
    if "SMA20" in indicators:
        price_cols.append(("sma20", "SMA20"))
    if "SMA60" in indicators:
        price_cols.append(("sma60", "SMA60"))
    if "布林上軌" in indicators:
        price_cols.append(("up", "布林上軌"))
    if "布林下軌" in indicators:
        price_cols.append(("dn", "布林下軌"))

    if price_cols:
        fig, ax = plt.subplots(figsize=(14, 6))
        for col, label in price_cols:
            if col in df.columns:
                ax.plot(df["date"], df[col], label=label)

        ax.set_title(f"{stock_id} 價格與均線/布林通道")
        ax.set_xlabel("日期")
        ax.set_ylabel("價格")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # ===== 2. MACD 圖 =====
    if ("DIF" in indicators) or ("DEA" in indicators):
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        if "DIF" in indicators and "dif" in df.columns:
            ax2.plot(df["date"], df["dif"], label="DIF")
        if "DEA" in indicators and "dea" in df.columns:
            ax2.plot(df["date"], df["dea"], label="DEA")

        ax2.axhline(0, linewidth=1)
        ax2.set_title("MACD")
        ax2.set_xlabel("日期")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    # ===== 3. 成交量圖 =====
    if "成交量" in indicators and "vol" in df.columns:
        fig3, ax3 = plt.subplots(figsize=(14, 4))
        ax3.bar(df["date"], df["vol"] / 1000)
        ax3.set_title("成交量（張）")
        ax3.set_xlabel("日期")
        ax3.set_ylabel("張")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)

    # ===== 4. 顯示表格 =====
    show_cols = ["date", "open", "high", "low", "close", "vol"]

    mapping = {
        "SMA5": "sma5",
        "SMA10": "sma10",
        "SMA20": "sma20",
        "SMA60": "sma60",
        "布林上軌": "up",
        "布林下軌": "dn",
        "DIF": "dif",
        "DEA": "dea"
    }

    for k, v in mapping.items():
        if k in indicators and v in df.columns:
            show_cols.append(v)

    show_df = df[show_cols].copy()
    show_df["date"] = show_df["date"].dt.strftime("%Y-%m-%d")

    st.dataframe(show_df, use_container_width=True)

    csv_data = show_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下載目前表格 CSV",
        data=csv_data,
        file_name=f"{stock_id}_analysis.csv",
        mime="text/csv"
    )