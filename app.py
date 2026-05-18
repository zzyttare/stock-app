import ssl
import urllib3
import logging
import re
import time
import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from FinMind.data import DataLoader

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
logging.getLogger("FinMind").setLevel(logging.ERROR)

DEFAULT_BB_PERIOD = 20
DEFAULT_BB_STD = 2.0
DEFAULT_BW_HIST_PERIOD = 60
DEFAULT_BW_PCT_THRESH = 20
DEFAULT_SQUEEZE_MIN = 5
DEFAULT_EXPAND_RATIO = 1.05
DEFAULT_VOL_RATIO = 1.5
DEFAULT_SCAN_TRADE_DAYS = 5
DEFAULT_GUEST_SAMPLE = 300
DEFAULT_SLEEP_SEC = 0.15

TWSE_TYPES = {"twse", "sii"}
TPEX_TYPES = {"otc", "tpex"}


def valid_code(code: str) -> bool:
    code = str(code)
    return bool(re.match(r"^\\d{4}$", code)) and not code.startswith("0")


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
            "trading_money": "amount",
        },
        inplace=True,
    )
    return df


def compute_bollinger(
    df: pd.DataFrame,
    bb_period: int,
    bb_std: float,
    bw_hist_period: int,
    bw_pct_thresh: float,
) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    df["ma"] = close.rolling(bb_period).mean()
    df["std"] = close.rolling(bb_period).std(ddof=1)
    df["upper"] = df["ma"] + bb_std * df["std"]
    df["lower"] = df["ma"] - bb_std * df["std"]
    df["bw"] = (df["upper"] - df["lower"]) / df["ma"]
    df["bw_low_thresh"] = (
        df["bw"].rolling(bw_hist_period).quantile(bw_pct_thresh / 100)
    )
    return df


def detect_squeeze(
    raw: pd.DataFrame,
    stock_id: str,
    scan_dates: set,
    global_seen: set,
    bb_period: int,
    bb_std: float,
    bw_hist_period: int,
    bw_pct_thresh: float,
    squeeze_min: int,
    expand_ratio: float,
    vol_ratio: float,
) -> list:
    signals = []
    df = normalize_daily_df(raw)

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

    df = compute_bollinger(df, bb_period, bb_std, bw_hist_period, bw_pct_thresh)
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


def run_scan(
    token: str,
    sampled: list,
    scan_dates: set,
    fetch_start: str,
    fetch_end: str,
    sleep_sec: float,
    bb_period: int,
    bb_std: float,
    bw_hist_period: int,
    bw_pct_thresh: float,
    squeeze_min: int,
    expand_ratio: float,
    vol_ratio: float,
):
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
                all_signals.extend(
                    detect_squeeze(
                        raw, sid, scan_dates, global_seen,
                        bb_period, bb_std, bw_hist_period, bw_pct_thresh,
                        squeeze_min, expand_ratio, vol_ratio
                    )
                )
        except Exception:
            skipped.append(sid)

        progress_bar.progress(idx / total)
        status_text.text(
            f"掃描進度：{idx}/{total}｜目前訊號：{len(all_signals)}｜耗時：{time.time() - t0:.1f} 秒"
        )
        time.sleep(sleep_sec)

    return all_signals, skipped, time.time() - t0


st.set_page_config(page_title="台股布林通道收斂擴張掃描器", layout="wide")
st.title("台股布林通道收斂擴張掃描器")
st.caption("Bollinger Squeeze Scanner：尋找波動壓縮後，帶寬放大且成交量確認的股票。")

if "token_verified" not in st.session_state:
    st.session_state["token_verified"] = False

with st.expander("策略條件說明", expanded=True):
    st.markdown("""
**收斂定義**
- BandWidth = `(布林上軌 - 布林下軌) / 布林中軌`
- BandWidth 連續多日低於近 60 日 BandWidth 的第 20 百分位。

**訊號成立條件**
1. 前一日仍在收斂，且收斂天數達門檻。
2. 今日離開收斂狀態。
3. 今日 BandWidth 大於昨日 BandWidth × 擴張倍率。
4. 今日成交量大於收斂期間日均量 × 成交量倍率。
""")

with st.sidebar:
    st.header("FinMind Token")
    token_input = st.text_input("請輸入 FinMind API Token", type="password")
    verify_token = st.button("驗證 Token / 載入股票清單", type="primary")

    st.divider()
    st.header("掃描設定")
    mode = st.radio(
        "執行模式",
        ["訪客模式：隨機抽樣", "完整模式：掃描全台股"],
        disabled=not st.session_state["token_verified"],
    )
    scan_days = st.number_input("掃描近幾個交易日", 1, 20, DEFAULT_SCAN_TRADE_DAYS, 1, disabled=not st.session_state["token_verified"])
    guest_sample = st.number_input("訪客模式抽樣檔數", 50, 1000, DEFAULT_GUEST_SAMPLE, 50, disabled=not st.session_state["token_verified"])
    sleep_sec = st.number_input("每檔 API 間隔秒數", 0.05, 2.0, DEFAULT_SLEEP_SEC, 0.05, disabled=not st.session_state["token_verified"])

    st.divider()
    st.header("策略參數")
    bb_period = st.number_input("布林週期", 10, 60, DEFAULT_BB_PERIOD, 1, disabled=not st.session_state["token_verified"])
    bb_std = st.number_input("標準差倍數", 1.0, 4.0, DEFAULT_BB_STD, 0.1, disabled=not st.session_state["token_verified"])
    bw_hist_period = st.number_input("BandWidth 歷史視窗", 30, 180, DEFAULT_BW_HIST_PERIOD, 5, disabled=not st.session_state["token_verified"])
    bw_pct_thresh = st.number_input("收斂百分位門檻", 5, 50, DEFAULT_BW_PCT_THRESH, 5, disabled=not st.session_state["token_verified"])
    squeeze_min = st.number_input("最少連續收斂天數", 2, 30, DEFAULT_SQUEEZE_MIN, 1, disabled=not st.session_state["token_verified"])
    expand_ratio_pct = st.number_input("BandWidth 擴張門檻 %", 1.0, 50.0, (DEFAULT_EXPAND_RATIO - 1) * 100, 1.0, disabled=not st.session_state["token_verified"])
    vol_ratio = st.number_input("成交量放大倍率", 1.0, 5.0, DEFAULT_VOL_RATIO, 0.1, disabled=not st.session_state["token_verified"])

    start_scan = st.button("開始掃描", disabled=not st.session_state["token_verified"])
    st.warning("完整掃描全台股耗時較久，且可能受 FinMind API 權限、請求次數與頻率限制影響。")


if verify_token:
    if not token_input:
        st.error("請先輸入 FinMind Token。")
        st.stop()

    with st.spinner("正在驗證 Token 並載入股票清單..."):
        try:
            twse, tpex, all_codes = fetch_all_stock_codes_cached(token_input)
            if not all_codes:
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
c1, c2, c3 = st.columns(3)
c1.metric("上市股票", len(twse))
c2.metric("上櫃股票", len(tpex))
c3.metric("總股票數", len(all_codes))

if start_scan:
    scan_dates = get_recent_trade_dates(int(scan_days))
    sorted_dates = sorted(scan_dates)

    data_lookback_days = int((int(bb_period) + int(bw_hist_period)) * 2.5)
    data_lookback_days = max(180, data_lookback_days)

    fetch_start = (min(scan_dates) - timedelta(days=data_lookback_days)).strftime("%Y-%m-%d")
    fetch_end = max(scan_dates).strftime("%Y-%m-%d")

    if mode == "訪客模式：隨機抽樣":
        half = int(guest_sample) // 2
        sampled = random.sample(twse, min(half, len(twse))) + random.sample(tpex, min(half, len(tpex)))
        sampled = list(dict.fromkeys(sampled))
        random.shuffle(sampled)
        mode_label = f"訪客模式：隨機抽 {len(sampled)} 檔"
    else:
        sampled = all_codes
        mode_label = f"完整模式：掃描 {len(sampled)} 檔"

    st.warning(mode_label)
    st.info(f"掃描日期：{sorted_dates[0]} ～ {sorted_dates[-1]}")
    st.info(f"資料區間：{fetch_start} ～ {fetch_end}")

    expand_ratio = 1 + float(expand_ratio_pct) / 100

    all_signals, skipped, total_time = run_scan(
        token, sampled, scan_dates, fetch_start, fetch_end, float(sleep_sec),
        int(bb_period), float(bb_std), int(bw_hist_period), float(bw_pct_thresh),
        int(squeeze_min), float(expand_ratio), float(vol_ratio)
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
    else:
        result_df = pd.DataFrame(all_signals).sort_values(["訊號日", "股票代號"], ascending=[False, True]).reset_index(drop=True)
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="下載 CSV",
            data=csv.encode("utf-8-sig"),
            file_name=f"boll_squeeze_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
