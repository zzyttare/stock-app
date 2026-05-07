# -*- coding: utf-8 -*-
"""
台股技術分析整合工具 Streamlit 版
功能：
1. 單檔技術分析：MA、布林通道、MACD、狀態判斷
2. 子母懷抱掃描：子線實體完全包在母線實體內
3. 純吞噬型態掃描：多頭吞噬 / 空頭吞噬，實體嚴格吞噬

資料來源：FinMind API
執行：streamlit run app.py
"""

import ssl
import urllib3
import logging
import re
import time
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from FinMind.data import DataLoader

# ─────────────────────────── 基本設定 ───────────────────────────
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
logging.getLogger("FinMind").setLevel(logging.ERROR)

SCAN_TRADE_DAYS = 5
GUEST_SAMPLE = 300
TWSE_TYPES = {"twse", "sii"}
TPEX_TYPES = {"otc", "tpex"}

# ─────────────────────────── 共用工具 ───────────────────────────

def valid_code(code: str) -> bool:
    code = str(code)
    return bool(re.match(r"^\d{4}$", code)) and not code.startswith("0")


def create_api(token: str = "") -> DataLoader:
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

        # 有些 FinMind 回傳 type 命名可能不同，若找不到上櫃，嘗試用其他類別補抓
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
        twse = list(dict.fromkeys(c for c in info[id_col].astype(str) if valid_code(c)))

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

    all_codes = twse_clean + tpex_clean
    return twse_clean, tpex_clean, all_codes


def get_recent_trade_dates(n: int) -> set:
    """簡易抓近 n 個平日。遇國定假日仍可能包含非交易日，但抓資料時會自然略過。"""
    dates = set()
    cursor = datetime.today().date()
    while len(dates) < n:
        if cursor.weekday() < 5:
            dates.add(cursor)
        cursor -= timedelta(days=1)
    return dates


def normalize_daily_columns(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"max": "high", "min": "low", "trading_volume": "volume_shares"}, inplace=True)
    return df


# ─────────────────────────── 單檔技術分析 ───────────────────────────

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


def detect_status(df: pd.DataFrame) -> pd.Series:
    status_list = []
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ma_cols = ["ma5", "ma10", "ma15", "ma60", "ma120", "ma240"]
    ma_labels = {
        "ma5": "MA5", "ma10": "MA10", "ma15": "MA15",
        "ma60": "MA60", "ma120": "MA120", "ma240": "MA240"
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
    raw = api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)

    if raw is None or raw.empty:
        raise ValueError(f"查無資料，請確認股票代號 {stock_id} 或 Token 是否正確。")

    raw = normalize_daily_columns(raw)
    required = ["date", "open", "close", "high", "low", "volume_shares"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"資料欄位不足：{missing}")

    df = raw[required].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for col in ["open", "close", "high", "low", "volume_shares"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pct_chg"] = df["close"].pct_change() * 100
    df["bb_upper"], df["bb_lower"] = calc_bollinger(df["close"])

    for n in [5, 10, 15, 60, 120, 240]:
        df[f"ma{n}"] = calc_ma(df["close"], n)

    df["macd"], df["signal_line"], df["histogram"] = calc_macd(df["close"])
    df["volume_k"] = (df["volume_shares"] / 1000).round(0).astype("Int64")
    df["status"] = detect_status(df)
    df = df[df["date"].dt.date >= output_start].reset_index(drop=True)
    return df


def format_analysis_df(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "date": "日期",
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
    out = df[list(col_map.keys())].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out = out.rename(columns=col_map)
    out = out.iloc[::-1].reset_index(drop=True)
    return out


# ─────────────────────────── 子母懷抱掃描 ───────────────────────────

def detect_inside_bar(df: pd.DataFrame, stock_id: str, scan_dates: set, global_seen: set) -> list:
    signals = []
    df = normalize_daily_columns(df)

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

        key = (stock_id, child_date, "inside")
        if key in global_seen:
            continue

        m_open, m_close = float(mother["open"]), float(mother["close"])
        c_open, c_close = float(child["open"]), float(child["close"])

        if any(pd.isna(v) or v <= 0 for v in [m_open, m_close, c_open, c_close]):
            continue

        m_body_hi, m_body_lo = max(m_open, m_close), min(m_open, m_close)
        c_body_hi, c_body_lo = max(c_open, c_close), min(c_open, c_close)

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


# ─────────────────────────── 吞噬型態掃描 ───────────────────────────

def detect_engulfing(df: pd.DataFrame, stock_id: str, scan_dates: set, global_seen: set) -> list:
    signals = []
    df = normalize_daily_columns(df)

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

        key = (stock_id, today_date, "engulfing")
        if key in global_seen:
            continue

        y_open, y_close = float(yest["open"]), float(yest["close"])
        t_open, t_close = float(today["open"]), float(today["close"])

        if any(pd.isna(v) or v <= 0 for v in [y_open, y_close, t_open, t_close]):
            continue

        y_body_hi, y_body_lo = max(y_open, y_close), min(y_open, y_close)
        t_body_hi, t_body_lo = max(t_open, t_close), min(t_open, t_close)

        is_engulf = t_body_hi > y_body_hi and t_body_lo < y_body_lo
        if not is_engulf:
            continue

        is_bull = y_close < y_open and t_close > t_open
        is_bear = y_close > y_open and t_close < t_open
        if not is_bull and not is_bear:
            continue

        pct_chg = (t_close - y_close) / y_close * 100
        pattern = "多頭吞噬" if is_bull else "空頭吞噬"

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
            "漲跌%": round(pct_chg, 2),
        })

    return signals


# ─────────────────────────── 掃描主流程 ───────────────────────────

def make_sampled_codes(mode: str, twse: list, tpex: list, all_codes: list, guest_sample: int) -> list:
    if mode == "訪客模式：隨機抽樣":
        half = int(guest_sample) // 2
        sampled = random.sample(twse, min(half, len(twse))) + random.sample(tpex, min(half, len(tpex)))
        sampled = list(dict.fromkeys(sampled))
        random.shuffle(sampled)
        return sampled
    return all_codes


def run_pattern_scan(token: str, sampled: list, scan_dates: set, fetch_start: str, fetch_end: str, sleep_sec: float, scan_kind: str):
    api = create_api(token)
    all_signals = []
    skipped = []
    global_seen = set()

    progress_bar = st.progress(0)
    status_text = st.empty()
    t0 = time.time()

    detector = detect_inside_bar if scan_kind == "inside" else detect_engulfing

    for idx, sid in enumerate(sampled, 1):
        try:
            raw = api.taiwan_stock_daily(stock_id=sid, start_date=fetch_start, end_date=fetch_end)
            if raw is None or raw.empty:
                skipped.append(sid)
            else:
                all_signals.extend(detector(raw, sid, scan_dates, global_seen))
        except Exception:
            skipped.append(sid)

        progress_bar.progress(idx / len(sampled))
        status_text.text(
            f"掃描進度：{idx}/{len(sampled)}｜目前訊號：{len(all_signals)}｜耗時：{time.time() - t0:.1f} 秒"
        )
        time.sleep(float(sleep_sec))

    return all_signals, skipped, time.time() - t0


# ─────────────────────────── Streamlit UI ───────────────────────────

st.set_page_config(page_title="台股技術分析整合工具", layout="wide")
st.title("台股技術分析整合工具")
st.caption("整合：單檔技術分析、子母懷抱掃描、純吞噬型態掃描。資料來源：FinMind API。")

if "token_verified" not in st.session_state:
    st.session_state["token_verified"] = False
if "token" not in st.session_state:
    st.session_state["token"] = ""

with st.sidebar:
    st.header("FinMind Token")
    token_input = st.text_input(
        "請輸入 FinMind API Token",
        value=st.session_state.get("token", ""),
        type="password",
        placeholder="輸入 Token 後按驗證"
    )

    verify_token = st.button("驗證 Token / 載入股票清單", type="primary")

    st.caption("需先驗證 Token，才能使用掃描全台股功能。")
    st.warning("若 FinMind 等級不足，可能出現掃描不完全、部分股票無資料、速度變慢或 API 限流等問題。")

if verify_token:
    if not token_input:
        st.error("請先輸入 FinMind Token。")
        st.stop()

    with st.spinner("正在驗證 Token 並載入股票清單..."):
        try:
            twse, tpex, all_codes = fetch_all_stock_codes_cached(token_input)
            if not all_codes:
                st.session_state["token_verified"] = False
                st.error("Token 驗證失敗，或無法取得股票清單。")
                st.stop()

            st.session_state["token"] = token_input
            st.session_state["twse"] = twse
            st.session_state["tpex"] = tpex
            st.session_state["all_codes"] = all_codes
            st.session_state["token_verified"] = True
            st.rerun()
        except Exception as e:
            st.session_state["token_verified"] = False
            st.error(f"驗證失敗：{e}")
            st.stop()

if not st.session_state["token_verified"]:
    st.info("請先在左側輸入 FinMind Token，並按「驗證 Token / 載入股票清單」。")
    st.stop()

token = st.session_state["token"]
twse = st.session_state["twse"]
tpex = st.session_state["tpex"]
all_codes = st.session_state["all_codes"]

st.success("Token 驗證成功，股票清單已載入。")

col_a, col_b, col_c = st.columns(3)
col_a.metric("上市股票", len(twse))
col_b.metric("上櫃股票", len(tpex))
col_c.metric("總股票數", len(all_codes))


with st.sidebar:
    st.divider()
    st.header("功能選擇")
    page_choice = st.radio(
        "請選擇功能",
        ["單檔技術分析", "子母懷抱掃描", "純吞噬型態掃描"],
        index=0
    )

# ── 掃描設定共用函式 ──
def render_scan_settings(key_prefix: str):
    with st.sidebar:
        st.divider()
        st.header("掃描設定")
        mode = st.radio(
            "執行模式",
            ["訪客模式：隨機抽樣", "完整模式：掃描全台股"],
            key=f"{key_prefix}_mode"
        )
        scan_days = st.number_input(
            "掃描近幾個交易日",
            min_value=1,
            max_value=20,
            value=SCAN_TRADE_DAYS,
            step=1,
            key=f"{key_prefix}_scan_days"
        )
        st.caption("可輸入範圍：1～20；建議 5～10。")

        guest_sample = st.number_input(
            "訪客模式抽樣檔數",
            min_value=50,
            max_value=1000,
            value=GUEST_SAMPLE,
            step=50,
            key=f"{key_prefix}_guest_sample"
        )
        st.caption("可輸入範圍：50～1000；數字越大掃描越久。")

        sleep_sec = st.number_input(
            "每檔 API 間隔秒數",
            min_value=0.05,
            max_value=2.0,
            value=0.15,
            step=0.05,
            key=f"{key_prefix}_sleep_sec"
        )
        st.caption("可輸入範圍：0.05～2 秒；建議 0.15～0.3 秒，太低可能被 API 限流。")

    return mode, int(scan_days), int(guest_sample), float(sleep_sec)


def prepare_scan_dates(scan_days: int):
    scan_dates = get_recent_trade_dates(scan_days)
    sorted_dates = sorted(scan_dates)
    fetch_start = (min(scan_dates) - timedelta(days=7)).strftime("%Y-%m-%d")
    fetch_end = max(scan_dates).strftime("%Y-%m-%d")
    return scan_dates, sorted_dates, fetch_start, fetch_end


if page_choice == "單檔技術分析":
    st.subheader("單檔技術分析")
    st.caption("輸入股票代號，查詢近兩年資料，計算 MA、布林通道、MACD 與狀態訊號。")

    c1, c2 = st.columns([1, 3])
    with c1:
        stock_id = st.text_input("股票代號", value="2330", max_chars=4)
        run_single = st.button("開始分析", type="primary")

    if run_single:
        if not valid_code(stock_id):
            st.error("請輸入正確 4 碼台股股票代號。")
        else:
            with st.spinner(f"正在取得 {stock_id} 近四年資料並輸出近兩年分析..."):
                try:
                    df = fetch_and_analyze_stock(stock_id, token)
                    display_df = format_analysis_df(df)

                    st.metric("資料筆數", len(display_df))
                    st.dataframe(display_df, use_container_width=True, height=600)

                    latest = display_df.iloc[0]
                    st.info(f"最新日期：{latest['日期']}｜收盤：{latest['收盤']}｜狀態：{latest['狀態']}")

                    csv = display_df.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(
                        "下載單檔分析 CSV",
                        data=csv.encode("utf-8-sig"),
                        file_name=f"{stock_id}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"分析失敗：{e}")

elif page_choice == "子母懷抱掃描":
    st.subheader("子母懷抱掃描")
    st.caption("條件：子線實體完全包在母線實體內，只看開盤價與收盤價，不看影線。")

    mode, scan_days, guest_sample, sleep_sec = render_scan_settings("inside")
    start_inside = st.button("開始子母懷抱掃描", type="primary")

    if start_inside:
        scan_dates, sorted_dates, fetch_start, fetch_end = prepare_scan_dates(scan_days)
        sampled = make_sampled_codes(mode, twse, tpex, all_codes, guest_sample)

        st.info(f"掃描日期：{sorted_dates[0]} ～ {sorted_dates[-1]}")
        st.info(f"資料區間：{fetch_start} ～ {fetch_end}")
        st.warning(f"{mode}：本次掃描 {len(sampled)} 檔")

        signals, skipped, total_time = run_pattern_scan(token, sampled, scan_dates, fetch_start, fetch_end, sleep_sec, "inside")

        st.subheader("掃描結果")
        c1, c2, c3 = st.columns(3)
        c1.metric("實際掃描", len(sampled) - len(skipped))
        c2.metric("訊號數", len(signals))
        c3.metric("耗時秒數", f"{total_time:.1f}")

        if skipped:
            st.caption(f"跳過無資料或失敗股票：{len(skipped)} 檔")

        if not signals:
            st.info("本次掃描無訊號。")
        else:
            result_df = pd.DataFrame(signals).sort_values(["子線日", "股票代號"], ascending=[False, True]).reset_index(drop=True)
            st.dataframe(result_df, use_container_width=True)
            csv = result_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "下載子母懷抱 CSV",
                data=csv.encode("utf-8-sig"),
                file_name=f"inside_bar_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

elif page_choice == "純吞噬型態掃描":
    st.subheader("純吞噬型態掃描")
    st.caption("條件：今日 K 線實體嚴格吞噬昨日 K 線實體；多頭吞噬為昨陰今陽，空頭吞噬為昨陽今陰。")

    mode, scan_days, guest_sample, sleep_sec = render_scan_settings("engulf")
    start_engulf = st.button("開始吞噬型態掃描", type="primary")

    if start_engulf:
        scan_dates, sorted_dates, fetch_start, fetch_end = prepare_scan_dates(scan_days)
        sampled = make_sampled_codes(mode, twse, tpex, all_codes, guest_sample)

        st.info(f"掃描日期：{sorted_dates[0]} ～ {sorted_dates[-1]}")
        st.info(f"資料區間：{fetch_start} ～ {fetch_end}")
        st.warning(f"{mode}：本次掃描 {len(sampled)} 檔")

        signals, skipped, total_time = run_pattern_scan(token, sampled, scan_dates, fetch_start, fetch_end, sleep_sec, "engulfing")

        st.subheader("掃描結果")
        c1, c2, c3 = st.columns(3)
        c1.metric("實際掃描", len(sampled) - len(skipped))
        c2.metric("訊號數", len(signals))
        c3.metric("耗時秒數", f"{total_time:.1f}")

        if skipped:
            st.caption(f"跳過無資料或失敗股票：{len(skipped)} 檔")

        if not signals:
            st.info("本次掃描無訊號。")
        else:
            result_df = pd.DataFrame(signals).sort_values(["今日", "型態", "股票代號"], ascending=[False, True, True]).reset_index(drop=True)
            bull = result_df[result_df["型態"] == "多頭吞噬"]
            bear = result_df[result_df["型態"] == "空頭吞噬"]

            st.write(f"多頭吞噬：{len(bull)} 筆｜空頭吞噬：{len(bear)} 筆")
            st.dataframe(result_df, use_container_width=True)

            csv = result_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "下載吞噬型態 CSV",
                data=csv.encode("utf-8-sig"),
                file_name=f"engulfing_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
