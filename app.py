import ssl
import urllib3
import logging
import re
import time
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
from FinMind.data import DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
logging.getLogger("FinMind").setLevel(logging.ERROR)

TWSE_TYPES = {"twse", "sii"}
TPEX_TYPES = {"otc", "tpex"}

DEFAULT_SCAN_DAYS = 5
DEFAULT_GUEST_SAMPLE = 300
DEFAULT_SLEEP_SEC = 0.15


# ============================================================
# 共用工具
# ============================================================
def valid_code(code: str) -> bool:
    code = str(code)
    return bool(re.match(r"^\d{4}$", code)) and not code.startswith("0")


def create_api(token: str) -> DataLoader:
    api = DataLoader()
    if token:
        api.token = token
    return api


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_stock_codes_cached(token: str):
    api = create_api(token)
    info = api.taiwan_stock_info()

    if info is None or info.empty:
        return [], [], []

    info.columns = [c.lower() for c in info.columns]
    type_col = next((c for c in info.columns if "type" in c), None)
    id_col = next((c for c in info.columns if "stock_id" in c or c == "id"), "stock_id")

    twse, tpex = [], []

    if type_col:
        type_series = info[type_col].astype(str).str.lower()

        twse = list(dict.fromkeys(
            c for c in info[type_series.isin(TWSE_TYPES)][id_col].astype(str)
            if valid_code(c)
        ))

        tpex = list(dict.fromkeys(
            c for c in info[type_series.isin(TPEX_TYPES)][id_col].astype(str)
            if valid_code(c)
        ))

        if len(tpex) == 0:
            all_types = set(type_series.unique())
            other_types = all_types - TWSE_TYPES - {"rotc", "roto", "興櫃", "emerg"}

            for t in sorted(other_types):
                cands = [
                    c for c in info[type_series == t][id_col].astype(str)
                    if valid_code(c)
                ]
                if len(cands) > 100:
                    tpex = list(dict.fromkeys(cands))
                    break
    else:
        twse = list(dict.fromkeys(
            c for c in info[id_col].astype(str)
            if valid_code(c)
        ))

    seen = set()
    twse_clean, tpex_clean = [], []

    for c in twse:
        if c not in seen:
            seen.add(c)
            twse_clean.append(c)

    for c in tpex:
        if c not in seen:
            seen.add(c)
            tpex_clean.append(c)

    return twse_clean, tpex_clean, twse_clean + tpex_clean


def get_recent_trade_dates(n: int) -> set:
    dates = set()
    cursor = datetime.today().date()

    while len(dates) < n:
        if cursor.weekday() < 5:
            dates.add(cursor)
        cursor -= timedelta(days=1)

    return dates


def normalize_daily_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.lower() for c in df.columns]
    df.rename(
        columns={
            "max": "high",
            "min": "low",
            "trading_volume": "volume",
            "volume_shares": "volume",
            "trading_money": "amount",
        },
        inplace=True,
    )
    return df


def is_finmind_rate_limit_error(error: Exception) -> bool:
    """
    判斷是否為 FinMind API 請求過於頻繁 / request 上限錯誤。
    """
    msg = str(error).lower()
    keywords = [
        "requests reach the upper limit",
        "upper limit",
        "too many requests",
        "rate limit",
        "request limit",
        "429",
    ]
    return any(k in msg for k in keywords)


def show_finmind_error(error: Exception, prefix: str = "操作失敗") -> None:
    """
    將 FinMind 常見錯誤轉成使用者看得懂的 Streamlit 訊息。
    """
    if is_finmind_rate_limit_error(error):
        st.error("請求過於頻繁，請一小時後再試。")
        st.info(
            "建議：降低掃描檔數、提高 API 間隔秒數，或使用有效 FinMind Token。"
            "訪客模式建議 0.3～0.5 秒；完整模式建議 0.8 秒以上。"
        )
    else:
        st.error(f"{prefix}：{error}")


def prepare_sample(mode: str, twse: list, tpex: list, all_codes: list, guest_sample: int):
    if mode == "訪客模式：隨機抽樣":
        half = int(guest_sample) // 2
        sampled = (
            random.sample(twse, min(half, len(twse))) +
            random.sample(tpex, min(half, len(tpex)))
        )
        sampled = list(dict.fromkeys(sampled))
        random.shuffle(sampled)
        return sampled, f"訪客模式：隨機抽 {len(sampled)} 檔"
    return all_codes, f"完整模式：掃描 {len(all_codes)} 檔"


def common_token_panel():
    st.sidebar.header("FinMind Token")
    token_input = st.sidebar.text_input(
        "請輸入 FinMind API Token（可留空）",
        type="password",
        help="Token 可留空，但完整掃描建議輸入 Token，避免 FinMind 請求限制。",
    )
    load_btn = st.sidebar.button("載入股票清單", type="primary")

    if "stock_list_loaded" not in st.session_state:
        st.session_state["stock_list_loaded"] = False

    if load_btn:
        token_value = token_input.strip()
        with st.spinner("正在載入股票清單..."):
            try:
                twse, tpex, all_codes = fetch_all_stock_codes_cached(token_value)

                if not all_codes:
                    st.session_state["stock_list_loaded"] = False
                    st.sidebar.error("無法取得股票清單。請確認 Token 或稍後再試。")
                    st.stop()

                st.session_state["token"] = token_value
                st.session_state["twse"] = twse
                st.session_state["tpex"] = tpex
                st.session_state["all_codes"] = all_codes
                st.session_state["stock_list_loaded"] = True
                st.rerun()

            except Exception as e:
                st.session_state["stock_list_loaded"] = False
                if is_finmind_rate_limit_error(e):
                    st.sidebar.error("請求過於頻繁，請一小時後再試。")
                    st.sidebar.info("建議降低操作頻率，或使用有效 FinMind Token。")
                else:
                    st.sidebar.error(f"載入股票清單失敗：{e}")
                st.stop()

    if not st.session_state["stock_list_loaded"]:
        st.info("請先在左側按「載入股票清單」。Token 可留空，但完整模式建議輸入 Token。")
        st.stop()

    token = st.session_state.get("token", "")
    twse = st.session_state.get("twse", [])
    tpex = st.session_state.get("tpex", [])
    all_codes = st.session_state.get("all_codes", [])

    return token, twse, tpex, all_codes


def stock_list_metrics(twse, tpex, all_codes):
    c1, c2, c3 = st.columns(3)
    c1.metric("上市股票", len(twse))
    c2.metric("上櫃股票", len(tpex))
    c3.metric("總股票數", len(all_codes))


# ============================================================
# 單檔技術分析
# ============================================================
def calc_ma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def calc_ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def calc_bollinger(series: pd.Series, n: int = 20, k: float = 2.0):
    mid = series.rolling(n, min_periods=n).mean()
    std = series.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return upper, lower


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def detect_technical_status(df: pd.DataFrame) -> pd.Series:
    status_list = []
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ma_cols = ["ma5", "ma10", "ma15", "ma60", "ma120", "ma240"]
    ma_labels = {
        "ma5": "MA5",
        "ma10": "MA10",
        "ma15": "MA15",
        "ma60": "MA60",
        "ma120": "MA120",
        "ma240": "MA240",
    }

    for i in range(len(df)):
        tags = []
        row = df.iloc[i]
        prev = df.iloc[i - 1] if i > 0 else None

        if pd.notna(row["bb_upper"]) and high.iloc[i] >= row["bb_upper"]:
            tags.append("觸布林上軌")
        if pd.notna(row["bb_lower"]) and low.iloc[i] <= row["bb_lower"]:
            tags.append("觸布林下軌")

        if prev is not None:
            if (
                pd.notna(row["macd"]) and pd.notna(row["signal_line"])
                and pd.notna(prev["macd"]) and pd.notna(prev["signal_line"])
            ):
                if prev["macd"] < prev["signal_line"] and row["macd"] > row["signal_line"]:
                    tags.append("MACD 金叉")
                elif prev["macd"] > prev["signal_line"] and row["macd"] < row["signal_line"]:
                    tags.append("MACD 死叉")

            for mc in ma_cols:
                if pd.notna(row[mc]) and pd.notna(prev[mc]):
                    p_close = close.iloc[i - 1]
                    c_close = close.iloc[i]
                    if p_close < prev[mc] and c_close > row[mc]:
                        tags.append(f"{ma_labels[mc]} 金叉")
                    elif p_close > prev[mc] and c_close < row[mc]:
                        tags.append(f"{ma_labels[mc]} 死叉")

        ma_vals = [row[mc] for mc in ma_cols]
        if all(pd.notna(v) for v in ma_vals):
            if all(ma_vals[j] > ma_vals[j + 1] for j in range(len(ma_vals) - 1)):
                tags.append("均線多頭排列")
            elif all(ma_vals[j] < ma_vals[j + 1] for j in range(len(ma_vals) - 1)):
                tags.append("均線空頭排列")

        status_list.append(" / ".join(tags) if tags else "—")

    return pd.Series(status_list, index=df.index)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_and_analyze_stock(stock_id: str, token: str = "") -> pd.DataFrame:
    today = datetime.today().date()
    output_start = today - timedelta(days=730)
    start_date = (today - timedelta(days=1460)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    api = create_api(token)

    raw = api.taiwan_stock_daily(
        stock_id=stock_id,
        start_date=start_date,
        end_date=end_date,
    )

    if raw is None or raw.empty:
        raise ValueError(f"查無資料，請確認股票代號（{stock_id}）或 Token 是否正確。")

    df = normalize_daily_df(raw)

    required_cols = ["date", "open", "close", "high", "low", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"資料缺少必要欄位：{col}")

    df = df[required_cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pct_chg"] = df["close"].pct_change() * 100
    df["bb_upper"], df["bb_lower"] = calc_bollinger(df["close"])

    for n in [5, 10, 15, 20, 60, 120, 240]:
        df[f"ma{n}"] = calc_ma(df["close"], n)

    df["macd"], df["signal_line"], df["histogram"] = calc_macd(df["close"])
    df["volume_k"] = (df["volume"] / 1000).round(0).astype("Int64")
    df["status"] = detect_technical_status(df)

    df = df[df["date"].dt.date >= output_start].reset_index(drop=True)
    return df


def make_technical_chart(df: pd.DataFrame, months: int):
    chart_df = df.copy()
    if months:
        start_date = chart_df["date"].max() - pd.DateOffset(months=months)
        chart_df = chart_df[chart_df["date"] >= start_date]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.58, 0.18, 0.24],
        subplot_titles=("K 線 / 均線 / 布林通道", "成交量", "MACD"),
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_df["date"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="K線",
        ),
        row=1,
        col=1,
    )

    for col, name in [
        ("ma5", "MA5"),
        ("ma10", "MA10"),
        ("ma20", "MA20"),
        ("ma60", "MA60"),
        ("ma120", "MA120"),
        ("ma240", "MA240"),
        ("bb_upper", "布林上軌"),
        ("bb_lower", "布林下軌"),
    ]:
        if col in chart_df.columns:
            fig.add_trace(
                go.Scatter(x=chart_df["date"], y=chart_df[col], mode="lines", name=name),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Bar(x=chart_df["date"], y=chart_df["volume_k"], name="成交量(張)"),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=chart_df["date"], y=chart_df["macd"], mode="lines", name="MACD"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=chart_df["date"], y=chart_df["signal_line"], mode="lines", name="Signal"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=chart_df["date"], y=chart_df["histogram"], name="Histogram"),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=900,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return fig


def render_single_stock_analysis(token: str):
    st.header("單檔技術分析＋圖表")

    c1, c2, c3 = st.columns([1, 1, 2])
    stock_id = c1.text_input("股票代號", value="2330", max_chars=10)
    chart_range = c2.selectbox("圖表範圍", ["近 3 個月", "近 6 個月", "近 1 年", "近 2 年"], index=2)
    run = c3.button("開始分析", type="primary")

    months_map = {
        "近 3 個月": 3,
        "近 6 個月": 6,
        "近 1 年": 12,
        "近 2 年": 24,
    }

    if run:
        stock_id = stock_id.strip()
        if not stock_id:
            st.error("請輸入股票代號。")
            return

        with st.spinner(f"正在取得 {stock_id} 資料並計算技術指標..."):
            try:
                df = fetch_and_analyze_stock(stock_id, token)
            except Exception as e:
                show_finmind_error(e, prefix="分析失敗")
                return

        latest = df.iloc[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("最新日期", latest["date"].strftime("%Y-%m-%d"))
        m2.metric("收盤", f"{latest['close']:.2f}")
        m3.metric("漲跌幅", f"{latest['pct_chg']:.2f}%" if pd.notna(latest["pct_chg"]) else "—")
        m4.metric("成交量(張)", f"{int(latest['volume_k']):,}" if pd.notna(latest["volume_k"]) else "—")

        fig = make_technical_chart(df, months_map[chart_range])
        st.plotly_chart(fig, use_container_width=True)

        display = df.copy()
        display["日期"] = display["date"].dt.strftime("%Y-%m-%d")

        col_map = {
            "open": "開盤",
            "close": "收盤",
            "high": "最高",
            "low": "最低",
            "pct_chg": "漲跌幅%",
            "bb_upper": "布林上軌",
            "bb_lower": "布林下軌",
            "ma5": "MA5",
            "ma10": "MA10",
            "ma15": "MA15",
            "ma60": "MA60",
            "ma120": "MA120",
            "ma240": "MA240",
            "macd": "MACD",
            "signal_line": "Signal",
            "histogram": "Histogram",
            "volume_k": "成交量(張)",
            "status": "狀態",
        }

        out_cols = ["日期"] + list(col_map.keys())
        display = display[out_cols].rename(columns=col_map).iloc[::-1].reset_index(drop=True)

        st.subheader("技術指標明細")
        st.dataframe(display, use_container_width=True)

        csv = display.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "下載 CSV",
            data=csv.encode("utf-8-sig"),
            file_name=f"{stock_id}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )


# ============================================================
# 子母懷抱掃描
# ============================================================
def detect_inside_bar(df: pd.DataFrame, stock_id: str, scan_dates: set, global_seen: set):
    signals = []
    df = normalize_daily_df(df)

    for col in ["open", "close", "date"]:
        if col not in df.columns:
            return signals

    df["date"] = pd.to_datetime(df["date"].astype(str))
    df = df.dropna(subset=["open", "close"]).sort_values("date").reset_index(drop=True)

    if len(df) < 2:
        return signals

    for i in range(1, len(df)):
        child = df.iloc[i]
        mother = df.iloc[i - 1]
        child_date = child["date"].date()

        if child_date not in scan_dates:
            continue

        key = (stock_id, child_date)
        if key in global_seen:
            continue

        m_open = float(mother["open"])
        m_close = float(mother["close"])
        c_open = float(child["open"])
        c_close = float(child["close"])

        if any(pd.isna(v) or v <= 0 for v in [m_open, m_close, c_open, c_close]):
            continue

        m_body_hi = max(m_open, m_close)
        m_body_lo = min(m_open, m_close)
        c_body_hi = max(c_open, c_close)
        c_body_lo = min(c_open, c_close)

        if c_body_hi < m_body_hi and c_body_lo > m_body_lo:
            global_seen.add(key)
            signals.append({
                "股票代號": stock_id,
                "母線日": mother["date"].strftime("%Y-%m-%d"),
                "子線日": child["date"].strftime("%Y-%m-%d"),
                "母實高": round(m_body_hi, 2),
                "母實低": round(m_body_lo, 2),
                "子實高": round(c_body_hi, 2),
                "子實低": round(c_body_lo, 2),
                "子收": round(c_close, 2),
            })

    return signals


# ============================================================
# 吞噬型態掃描
# ============================================================
def detect_engulfing(df: pd.DataFrame, stock_id: str, scan_dates: set, global_seen: set):
    signals = []
    df = normalize_daily_df(df)

    for col in ["open", "close", "date"]:
        if col not in df.columns:
            return signals

    df["date"] = pd.to_datetime(df["date"].astype(str))
    df = df.dropna(subset=["open", "close"]).sort_values("date").reset_index(drop=True)

    if len(df) < 2:
        return signals

    for i in range(1, len(df)):
        today = df.iloc[i]
        yest = df.iloc[i - 1]
        today_date = today["date"].date()

        if today_date not in scan_dates:
            continue

        key = (stock_id, today_date)
        if key in global_seen:
            continue

        y_open = float(yest["open"])
        y_close = float(yest["close"])
        t_open = float(today["open"])
        t_close = float(today["close"])

        if any(pd.isna(v) or v <= 0 for v in [y_open, y_close, t_open, t_close]):
            continue

        y_body_hi = max(y_open, y_close)
        y_body_lo = min(y_open, y_close)
        t_body_hi = max(t_open, t_close)
        t_body_lo = min(t_open, t_close)

        is_engulf = (t_body_hi > y_body_hi and t_body_lo < y_body_lo)
        if not is_engulf:
            continue

        is_bull = (y_close < y_open and t_close > t_open)
        is_bear = (y_close > y_open and t_close < t_open)

        if not is_bull and not is_bear:
            continue

        pattern = "多頭吞噬" if is_bull else "空頭吞噬"
        pct_chg = (t_close - y_close) / y_close * 100

        global_seen.add(key)
        signals.append({
            "股票代號": stock_id,
            "型態": pattern,
            "昨日": yest["date"].strftime("%Y-%m-%d"),
            "今日": today["date"].strftime("%Y-%m-%d"),
            "昨實高": round(y_body_hi, 2),
            "昨實低": round(y_body_lo, 2),
            "今實高": round(t_body_hi, 2),
            "今實低": round(t_body_lo, 2),
            "收盤": round(t_close, 2),
            "漲跌幅%": round(pct_chg, 2),
        })

    return signals


# ============================================================
# 布林收斂擴張掃描
# ============================================================
def compute_squeeze_bollinger(df: pd.DataFrame, bb_period: int, bb_std: float, bw_hist_period: int, bw_pct_thresh: float):
    df = df.copy()
    close = df["close"]
    df["ma"] = close.rolling(bb_period).mean()
    df["std"] = close.rolling(bb_period).std(ddof=1)
    df["upper"] = df["ma"] + bb_std * df["std"]
    df["lower"] = df["ma"] - bb_std * df["std"]
    df["bw"] = (df["upper"] - df["lower"]) / df["ma"]
    df["bw_low_thresh"] = df["bw"].rolling(bw_hist_period).quantile(bw_pct_thresh / 100)
    return df


def detect_squeeze(df: pd.DataFrame, stock_id: str, scan_dates: set, global_seen: set,
                   bb_period: int, bb_std: float, bw_hist_period: int, bw_pct_thresh: float,
                   squeeze_min: int, expand_ratio: float, vol_ratio: float):
    signals = []
    df = normalize_daily_df(df)

    for col in ["open", "close", "date", "volume"]:
        if col not in df.columns:
            return signals

    df["date"] = pd.to_datetime(df["date"].astype(str))
    df = df.dropna(subset=["open", "close", "volume"]).sort_values("date").reset_index(drop=True)

    for col in ["open", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "close", "volume"]).reset_index(drop=True)

    if len(df) < bb_period + bw_hist_period:
        return signals

    df = compute_squeeze_bollinger(df, bb_period, bb_std, bw_hist_period, bw_pct_thresh)
    df = df.dropna(subset=["bw", "bw_low_thresh"]).reset_index(drop=True)

    if len(df) < squeeze_min + 1:
        return signals

    df["is_squeeze"] = df["bw"] <= df["bw_low_thresh"]

    streaks = []
    streak = 0
    for sq in df["is_squeeze"]:
        streak = streak + 1 if sq else 0
        streaks.append(streak)
    df["squeeze_streak"] = streaks

    for i in range(1, len(df)):
        today = df.iloc[i]
        yesterday = df.iloc[i - 1]
        today_date = today["date"].date()

        if today_date not in scan_dates:
            continue

        key = (stock_id, today_date)
        if key in global_seen:
            continue

        if not yesterday["is_squeeze"]:
            continue
        if int(yesterday["squeeze_streak"]) < squeeze_min:
            continue
        if today["is_squeeze"]:
            continue

        bw_today = float(today["bw"])
        bw_yest = float(yesterday["bw"])

        if bw_yest <= 0 or bw_today <= bw_yest * expand_ratio:
            continue

        squeeze_len = int(yesterday["squeeze_streak"])
        squeeze_start_idx = max(0, i - squeeze_len)
        sq_vols = df.iloc[squeeze_start_idx:i]["volume"].astype(float)

        if sq_vols.empty or sq_vols.mean() <= 0:
            continue

        avg_sq_vol = float(sq_vols.mean())
        today_vol = float(today["volume"])

        if today_vol < avg_sq_vol * vol_ratio:
            continue

        cur_close = float(today["close"])
        prev_close = float(yesterday["close"])

        if prev_close <= 0:
            continue

        pct_chg = (cur_close - prev_close) / prev_close * 100
        vol_mult = today_vol / avg_sq_vol

        global_seen.add(key)
        signals.append({
            "股票代號": stock_id,
            "訊號日": today["date"].strftime("%Y-%m-%d"),
            "收斂天數": squeeze_len,
            "昨日BW%": round(bw_yest * 100, 3),
            "今日BW%": round(bw_today * 100, 3),
            "BW擴張%": round((bw_today / bw_yest - 1) * 100, 2),
            "收盤": round(cur_close, 2),
            "漲跌幅%": round(pct_chg, 2),
            "量能倍數": round(vol_mult, 2),
            "收斂期均量": int(avg_sq_vol),
            "今日成交量": int(today_vol),
        })

    return signals


def run_pattern_scan(token: str, sampled: list, scan_dates: set, fetch_start: str, fetch_end: str,
                     sleep_sec: float, detector, detector_kwargs=None):
    detector_kwargs = detector_kwargs or {}
    api = create_api(token)

    all_signals = []
    skipped = []
    global_seen = set()

    progress_bar = st.progress(0)
    status_text = st.empty()
    t0 = time.time()

    total = len(sampled)
    for idx, sid in enumerate(sampled, 1):
        try:
            raw = api.taiwan_stock_daily(stock_id=sid, start_date=fetch_start, end_date=fetch_end)

            if raw is None or raw.empty:
                skipped.append(sid)
            else:
                signals = detector(raw, sid, scan_dates, global_seen, **detector_kwargs)
                all_signals.extend(signals)

        except Exception as e:
            skipped.append(sid)

            if is_finmind_rate_limit_error(e):
                st.error("請求過於頻繁，請一小時後再試。")
                st.info(
                    "系統已停止本次掃描，避免繼續觸發 FinMind API 限流。"
                    "建議提高 API 間隔秒數，或降低抽樣檔數後再試。"
                )
                status_text.text(
                    f"掃描已中止：FinMind API 請求過於頻繁｜進度：{idx}/{total}｜目前訊號：{len(all_signals)}"
                )
                break

        progress_bar.progress(idx / total)
        status_text.text(f"掃描進度：{idx}/{total}｜目前訊號：{len(all_signals)}｜耗時：{time.time() - t0:.1f} 秒")
        time.sleep(sleep_sec)

    return all_signals, skipped, time.time() - t0


def render_scan_common(token: str, twse: list, tpex: list, all_codes: list, scan_type: str):
    st.header(scan_type)

    st.sidebar.divider()
    st.sidebar.header("掃描設定")

    mode = st.sidebar.radio(
        "執行模式",
        ["訪客模式：隨機抽樣", "完整模式：掃描全台股"],
        help="沒有 Token 時請使用訪客模式；完整模式容易受 FinMind 請求限制影響。",
    )
    scan_days = st.sidebar.number_input("掃描近幾個交易日", 1, 20, DEFAULT_SCAN_DAYS, 1)
    guest_sample = st.sidebar.number_input("訪客模式抽樣檔數", 50, 1000, DEFAULT_GUEST_SAMPLE, 50)
    sleep_sec = st.sidebar.number_input("每檔 API 間隔秒數", 0.05, 2.0, DEFAULT_SLEEP_SEC, 0.05)

    detector_kwargs = {}
    data_lookback_days = 14

    if scan_type == "布林收斂擴張掃描":
        st.markdown("""
**訊號條件**
- BandWidth 連續低於歷史低位。
- 收斂結束後，BandWidth 放大。
- 成交量相對收斂期均量放大。
""")
        st.sidebar.divider()
        st.sidebar.header("布林策略參數")

        bb_period = st.sidebar.number_input("布林週期", 10, 60, 20, 1)
        bb_std = st.sidebar.number_input("標準差倍數", 1.0, 4.0, 2.0, 0.1)
        bw_hist_period = st.sidebar.number_input("BandWidth 歷史視窗", 30, 180, 60, 5)
        bw_pct_thresh = st.sidebar.number_input("收斂百分位門檻", 5, 50, 20, 5)
        squeeze_min = st.sidebar.number_input("最少連續收斂天數", 2, 30, 5, 1)
        expand_ratio_pct = st.sidebar.number_input("BandWidth 擴張門檻 %", 1.0, 50.0, 5.0, 1.0)
        vol_ratio = st.sidebar.number_input("成交量放大倍率", 1.0, 5.0, 1.5, 0.1)

        detector = detect_squeeze
        detector_kwargs = {
            "bb_period": int(bb_period),
            "bb_std": float(bb_std),
            "bw_hist_period": int(bw_hist_period),
            "bw_pct_thresh": float(bw_pct_thresh),
            "squeeze_min": int(squeeze_min),
            "expand_ratio": 1 + float(expand_ratio_pct) / 100,
            "vol_ratio": float(vol_ratio),
        }
        data_lookback_days = max(180, int((int(bb_period) + int(bw_hist_period)) * 2.5))
        sort_cols = ["訊號日", "股票代號"]
        file_prefix = "boll_squeeze"

    elif scan_type == "子母懷抱掃描":
        st.caption("條件：子線實體完全包在母線實體內，只看開盤價與收盤價，不看影線。")
        detector = detect_inside_bar
        sort_cols = ["子線日", "股票代號"]
        file_prefix = "inside_bar"

    else:
        st.caption("條件：新 K 實體嚴格大於舊 K 實體，並區分多頭吞噬與空頭吞噬。")
        detector = detect_engulfing
        sort_cols = ["今日", "型態", "股票代號"]
        file_prefix = "engulfing"

    start_scan = st.sidebar.button("開始掃描", type="primary")

    stock_list_metrics(twse, tpex, all_codes)

    if not start_scan:
        st.info("設定完成後，請按左側「開始掃描」。")
        return

    if mode == "完整模式：掃描全台股" and not token:
        st.error("完整模式需要 FinMind Token。請輸入 Token 後重新載入股票清單，或改用訪客模式。")
        return

    scan_dates = get_recent_trade_dates(int(scan_days))
    sorted_dates = sorted(scan_dates)

    fetch_start = (min(scan_dates) - timedelta(days=data_lookback_days)).strftime("%Y-%m-%d")
    fetch_end = max(scan_dates).strftime("%Y-%m-%d")

    sampled, mode_label = prepare_sample(mode, twse, tpex, all_codes, int(guest_sample))

    st.warning(mode_label)
    st.info(f"掃描日期：{sorted_dates[0]} ～ {sorted_dates[-1]}")
    st.info(f"資料區間：{fetch_start} ～ {fetch_end}")

    all_signals, skipped, total_time = run_pattern_scan(
        token=token,
        sampled=sampled,
        scan_dates=scan_dates,
        fetch_start=fetch_start,
        fetch_end=fetch_end,
        sleep_sec=float(sleep_sec),
        detector=detector,
        detector_kwargs=detector_kwargs,
    )

    st.subheader("掃描結果")
    m1, m2, m3 = st.columns(3)
    m1.metric("實際掃描", len(sampled) - len(skipped))
    m2.metric("訊號數", len(all_signals))
    m3.metric("耗時秒數", f"{total_time:.1f}")

    if skipped:
        st.caption(f"跳過無資料或失敗股票：{len(skipped)} 檔")

    if not all_signals:
        st.info("本次掃描無訊號。")
        return

    result_df = pd.DataFrame(all_signals).sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1)).reset_index(drop=True)
    st.dataframe(result_df, use_container_width=True)

    csv = result_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="下載 CSV",
        data=csv.encode("utf-8-sig"),
        file_name=f"{file_prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


# ============================================================
# Streamlit 入口
# ============================================================
st.set_page_config(page_title="台股技術分析整合工具", layout="wide")
st.title("台股技術分析整合工具")
st.caption("整合單檔技術分析、子母懷抱、吞噬型態、布林收斂擴張掃描。")

with st.sidebar:
    page = st.selectbox(
        "功能選單",
        ["單檔技術分析", "子母懷抱掃描", "吞噬型態掃描", "布林收斂擴張掃描"],
    )

token, twse, tpex, all_codes = common_token_panel()

st.success("股票清單已載入。")

if page == "單檔技術分析":
    render_single_stock_analysis(token)
elif page == "子母懷抱掃描":
    render_scan_common(token, twse, tpex, all_codes, "子母懷抱掃描")
elif page == "吞噬型態掃描":
    render_scan_common(token, twse, tpex, all_codes, "吞噬型態掃描")
else:
    render_scan_common(token, twse, tpex, all_codes, "布林收斂擴張掃描")
