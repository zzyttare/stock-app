import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

# FinMind 可選載入，避免環境或回應異常時整站掛掉
try:
    from FinMind.data import DataLoader
    FINMIND_IMPORT_OK = True
except Exception:
    DataLoader = None
    FINMIND_IMPORT_OK = False

# ========= 頁面設定 =========
st.set_page_config(page_title="台股技術分析 App", layout="wide")
st.title("台股技術分析 App")

# ========= Token =========
MY_TOKEN = os.getenv("FINMIND_TOKEN", "").strip()

# ========= 初始化 =========
@st.cache_resource
def init_system():
    stock_names = {}
    market_type_map = {}

    if not FINMIND_IMPORT_OK:
        return None, stock_names, market_type_map, "FinMind 模組載入失敗"

    if not MY_TOKEN:
        return None, stock_names, market_type_map, "未設定 FINMIND_TOKEN，將使用基本模式"

    try:
        api = DataLoader()
        api.login_by_token(api_token=MY_TOKEN)

        df = api.taiwan_stock_info()

        if df is None or df.empty:
            return None, stock_names, market_type_map, "FinMind 未回傳股票清單，將使用基本模式"

        required_cols = {"stock_id", "stock_name"}
        if not required_cols.issubset(df.columns):
            return None, stock_names, market_type_map, "FinMind 回傳欄位異常，將使用基本模式"

        df = df[df["stock_id"].astype(str).str.fullmatch(r"\d{4}")].copy()
        df["stock_id"] = df["stock_id"].astype(str)

        market_type_map = {
            sid: ("上櫃" if int(sid) >= 4000 else "上市")
            for sid in df["stock_id"]
        }
        stock_names = dict(zip(df["stock_id"], df["stock_name"]))

        return api, stock_names, market_type_map, f"FinMind 已連線，載入 {len(stock_names)} 檔股票資料"

    except Exception as e:
        return None, stock_names, market_type_map, f"FinMind 初始化失敗：{e}"

dl, stock_names, market_type_map, finmind_status = init_system()

# ========= 狀態提示 =========
with st.expander("系統狀態", expanded=False):
    if dl is not None:
        st.success(finmind_status)
    else:
        st.warning(finmind_status)
        st.caption("即使 FinMind 不可用，仍可查詢個股價格與技術指標，但股票名稱可能無法顯示。")


# ========= 輔助：欄位整理 =========
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # MultiIndex 欄位先攤平
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            parts = [str(x) for x in col if str(x) != "" and str(x).lower() != "nan"]
            # 取第一層可避免 yfinance 把 ticker 也拼進來造成怪欄位
            flat_cols.append(parts[0] if len(parts) > 0 else "unknown")
        df.columns = flat_cols

    # 全部轉小寫字串
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 常見名稱對應
    df = df.rename(columns={
        "adj close": "close",
        "volume": "vol",
        "trading_volume": "vol",
        "datetime": "date",
    })

    # 如果 index 被 reset 後跑成 index 欄，就當作 date
    if "date" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "date"})

    # 若有重複欄名，只保留第一個
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]

    return df


def safe_series(df: pd.DataFrame, col: str):
    """確保取出的是單一 Series，不是 DataFrame"""
    if col not in df.columns:
        return None

    obj = df[col]

    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return None
        obj = obj.iloc[:, 0]

    if not isinstance(obj, pd.Series):
        try:
            obj = pd.Series(obj)
        except Exception:
            return None

    return obj


# ========= 技術指標 =========
def get_indicators(df: pd.DataFrame):
    if df is None or df.empty or len(df) < 40:
        return None

    df = normalize_columns(df)

    required = {"open", "high", "low", "close", "vol", "date"}
    if not required.issubset(df.columns):
        return None

    date_s = safe_series(df, "date")
    open_s = safe_series(df, "open")
    high_s = safe_series(df, "high")
    low_s = safe_series(df, "low")
    close_s = safe_series(df, "close")
    vol_s = safe_series(df, "vol")

    if any(x is None for x in [date_s, open_s, high_s, low_s, close_s, vol_s]):
        return None

    out = pd.DataFrame({
        "date": pd.to_datetime(date_s, errors="coerce"),
        "open": pd.to_numeric(open_s, errors="coerce"),
        "high": pd.to_numeric(high_s, errors="coerce"),
        "low": pd.to_numeric(low_s, errors="coerce"),
        "close": pd.to_numeric(close_s, errors="coerce"),
        "vol": pd.to_numeric(vol_s, errors="coerce"),
    })

    out = out.dropna(subset=["date", "close"]).copy()

    if out.empty or len(out) < 40:
        return None

    close = out["close"]
    vol = out["vol"]

    out["sma5"] = close.rolling(5).mean()
    out["sma10"] = close.rolling(10).mean()
    out["sma20"] = close.rolling(20).mean()
    out["sma60"] = close.rolling(60).mean()

    std20 = close.rolling(20).std()
    out["up"] = out["sma20"] + 2 * std20
    out["dn"] = out["sma20"] - 2 * std20
    out["bw"] = (out["up"] - out["dn"]) / out["sma20"]

    band_range = out["up"] - out["dn"]
    out["b_percent"] = np.where(
        band_range != 0,
        (close - out["dn"]) / band_range,
        np.nan
    )
    out["b_ma10"] = out["b_percent"].rolling(10).mean()

    out["ema12"] = close.ewm(span=12, adjust=False).mean()
    out["ema26"] = close.ewm(span=26, adjust=False).mean()
    out["dif"] = out["ema12"] - out["ema26"]
    out["dea"] = out["dif"].ewm(span=9, adjust=False).mean()

    out["v_ma5"] = vol.shift(1).rolling(5).mean()
    out["v_ratio"] = vol / out["v_ma5"].replace(0, np.nan)

    return out


# ========= 輔助函式 =========
def guess_market(stock_id: str) -> str:
    if stock_id in market_type_map:
        return market_type_map[stock_id]
    try:
        return "上市" if int(stock_id) < 4000 else "上櫃"
    except Exception:
        return "上市"


def get_stock_name(stock_id: str) -> str:
    return stock_names.get(stock_id, "")


# ========= 抓單股資料 =========
@st.cache_data(ttl=300)
def load_stock_data(stock_id: str, period: str):
    try:
        mt = guess_market(stock_id)
        ticker = f"{stock_id}.TW" if mt == "上市" else f"{stock_id}.TWO"

        df_raw = yf.download(
            ticker,
            period=period,
            progress=False,
            auto_adjust=False,
            threads=False,
            group_by="column",
        ).reset_index()

        df = get_indicators(df_raw)
        if df is None or df.empty:
            return None, f"{ticker} 查無資料或資料不足"

        return df, ""
    except Exception as e:
        return None, f"資料下載失敗：{e}"


# ========= 側邊欄 =========
with st.sidebar:
    st.header("查詢條件")

    stock_id = st.text_input("股票代號", value="2330").strip()

    period_map = {
        "6個月": "6mo",
        "1年": "1y",
        "2年": "2y",
    }
    period_label = st.selectbox("資料期間", list(period_map.keys()), index=2)
    period = period_map[period_label]

    indicators = st.multiselect(
        "選擇要顯示的指標",
        options=[
            "收盤價", "SMA5", "SMA10", "SMA20", "SMA60",
            "布林上軌", "布林下軌",
            "DIF", "DEA",
            "成交量",
        ],
        default=["收盤價", "SMA20", "布林上軌", "布林下軌", "成交量"]
    )

    do_query = st.button("開始查詢", use_container_width=True)

# ========= 查詢 =========
if do_query:
    if not stock_id.isdigit() or len(stock_id) != 4:
        st.warning("請輸入 4 碼股票代號")
        st.stop()

    with st.spinner("抓取資料中..."):
        df, err = load_stock_data(stock_id, period)

    if err:
        st.error(err)
        st.stop()

    if df is None or df.empty:
        st.error("查無資料或資料不足")
        st.stop()

    stock_name = get_stock_name(stock_id)
    title_text = f"{stock_id} {stock_name}".strip()
    st.subheader(title_text)

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
        plt.close(fig)

    # ===== 2. MACD 圖 =====
    if ("DIF" in indicators) or ("DEA" in indicators):
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        has_macd = False

        if "DIF" in indicators and "dif" in df.columns:
            ax2.plot(df["date"], df["dif"], label="DIF")
            has_macd = True
        if "DEA" in indicators and "dea" in df.columns:
            ax2.plot(df["date"], df["dea"], label="DEA")
            has_macd = True

        if has_macd:
            ax2.axhline(0, linewidth=1)
            ax2.set_title("MACD")
            ax2.set_xlabel("日期")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
        plt.close(fig2)

    # ===== 3. 成交量圖 =====
    if "成交量" in indicators and "vol" in df.columns:
        fig3, ax3 = plt.subplots(figsize=(14, 4))
        ax3.bar(df["date"], df["vol"] / 1000)
        ax3.set_title("成交量（張）")
        ax3.set_xlabel("日期")
        ax3.set_ylabel("張")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)

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
        "DEA": "dea",
    }

    for k, v in mapping.items():
        if k in indicators and v in df.columns and v not in show_cols:
            show_cols.append(v)

    show_df = df[show_cols].copy()
    show_df["date"] = show_df["date"].dt.strftime("%Y-%m-%d")

    st.dataframe(show_df, use_container_width=True)

    csv_data = show_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下載目前表格 CSV",
        data=csv_data,
        file_name=f"{stock_id}_analysis.csv",
        mime="text/csv",
        use_container_width=True,
    )
