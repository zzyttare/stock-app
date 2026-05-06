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

SCAN_TRADE_DAYS = 5
GUEST_SAMPLE = 300
TWSE_TYPES = {"twse", "sii"}
TPEX_TYPES = {"otc", "tpex"}


def valid_code(code):
    code = str(code)
    return bool(re.match(r"^\d{4}$", code)) and not code.startswith("0")


def create_api(token):
    api = DataLoader()
    if token:
        api.token = token
    return api


@st.cache_data(ttl=3600)
def fetch_all_stock_codes_cached(token):
    api = create_api(token)

    info = api.taiwan_stock_info()

    if info is None or info.empty:
        return [], [], []

    info.columns = [c.lower() for c in info.columns]

    type_col = next((c for c in info.columns if "type" in c), None)
    id_col = next(
        (c for c in info.columns if "stock_id" in c or c == "id"),
        "stock_id"
    )

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

    all_codes = twse_clean + tpex_clean
    return twse_clean, tpex_clean, all_codes


def get_recent_trade_dates(n):
    dates = set()
    cursor = datetime.today().date()

    while len(dates) < n:
        if cursor.weekday() < 5:
            dates.add(cursor)
        cursor -= timedelta(days=1)

    return dates


def detect_inside_bar(df, stock_id, scan_dates, global_seen):
    signals = []

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"max": "high", "min": "low"}, inplace=True)

    for col in ["open", "close", "date"]:
        if col not in df.columns:
            return signals

    df["date"] = pd.to_datetime(df["date"].astype(str))
    df = (
        df.dropna(subset=["open", "close"])
        .sort_values("date")
        .reset_index(drop=True)
    )

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


def run_scan(token, sampled, scan_dates, fetch_start, fetch_end, sleep_sec):
    api = create_api(token)

    all_signals = []
    skipped = []
    global_seen = set()

    progress_bar = st.progress(0)
    status_text = st.empty()

    t0 = time.time()

    for idx, sid in enumerate(sampled, 1):
        try:
            raw = api.taiwan_stock_daily(
                stock_id=sid,
                start_date=fetch_start,
                end_date=fetch_end,
            )

            if raw is None or raw.empty:
                skipped.append(sid)
            else:
                sigs = detect_inside_bar(raw, sid, scan_dates, global_seen)
                all_signals.extend(sigs)

        except Exception:
            skipped.append(sid)

        progress_bar.progress(idx / len(sampled))
        status_text.text(
            f"掃描進度：{idx}/{len(sampled)}｜目前訊號：{len(all_signals)}｜耗時：{time.time() - t0:.1f} 秒"
        )

        time.sleep(sleep_sec)

    return all_signals, skipped, time.time() - t0


st.set_page_config(
    page_title="台股子母懷抱掃描器",
    layout="wide"
)

st.title("台股子母懷抱掃描器")
st.caption("條件：子線實體完全包在母線實體內，只看開盤價與收盤價，不看影線。")

if "token_verified" not in st.session_state:
    st.session_state["token_verified"] = False

with st.sidebar:
    st.header("第 1 步：FinMind Token")

    token_input = st.text_input(
        "請輸入 FinMind API Token",
        type="password",
        placeholder="請輸入你的 FinMind Token"
    )

    verify_token = st.button("驗證 Token / 載入股票清單", type="primary")

    st.divider()

    st.header("第 2 步：掃描設定")

    mode = st.radio(
        "執行模式",
        ["訪客模式：隨機抽樣", "完整模式：掃描全台股"],
        disabled=not st.session_state["token_verified"]
    )

    scan_days = st.number_input(
        "掃描近幾個交易日",
        min_value=1,
        max_value=20,
        value=SCAN_TRADE_DAYS,
        step=1,
        disabled=not st.session_state["token_verified"]
    )

    guest_sample = st.number_input(
        "訪客模式抽樣檔數",
        min_value=50,
        max_value=1000,
        value=GUEST_SAMPLE,
        step=50,
        disabled=not st.session_state["token_verified"]
    )

    sleep_sec = st.number_input(
        "每檔 API 間隔秒數",
        min_value=0.05,
        max_value=2.0,
        value=0.15,
        step=0.05,
        disabled=not st.session_state["token_verified"]
    )

    start_scan = st.button(
        "開始掃描",
        disabled=not st.session_state["token_verified"]
    )


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

            st.success("Token 驗證成功，股票清單已載入。")
            st.info(f"上市：{len(twse)} 檔｜上櫃：{len(tpex)} 檔｜合計：{len(all_codes)} 檔")

        except Exception as e:
            st.session_state["token_verified"] = False
            st.error(f"驗證失敗：{e}")
            st.stop()


if not st.session_state["token_verified"]:
    st.info("請先在左側輸入 FinMind Token，並按「驗證 Token / 載入股票清單」。")
    st.stop()


st.success("Token 已驗證，可以開始掃描。")

twse = st.session_state["twse"]
tpex = st.session_state["tpex"]
all_codes = st.session_state["all_codes"]
token = st.session_state["token"]

col_a, col_b, col_c = st.columns(3)
col_a.metric("上市股票", len(twse))
col_b.metric("上櫃股票", len(tpex))
col_c.metric("總股票數", len(all_codes))


if start_scan:
    scan_dates = get_recent_trade_dates(int(scan_days))
    sorted_dates = sorted(scan_dates)

    fetch_start = (min(scan_dates) - timedelta(days=7)).strftime("%Y-%m-%d")
    fetch_end = max(scan_dates).strftime("%Y-%m-%d")

    st.info(f"掃描日期：{sorted_dates[0]} ～ {sorted_dates[-1]}")
    st.info(f"資料區間：{fetch_start} ～ {fetch_end}")

    if mode == "訪客模式：隨機抽樣":
        half = int(guest_sample) // 2

        sampled = (
            random.sample(twse, min(half, len(twse))) +
            random.sample(tpex, min(half, len(tpex)))
        )

        sampled = list(dict.fromkeys(sampled))
        random.shuffle(sampled)

        mode_label = f"訪客模式：隨機抽 {len(sampled)} 檔"
    else:
        sampled = all_codes
        mode_label = f"完整模式：掃描 {len(sampled)} 檔"

    st.warning(mode_label)

    all_signals, skipped, total_time = run_scan(
        token=token,
        sampled=sampled,
        scan_dates=scan_dates,
        fetch_start=fetch_start,
        fetch_end=fetch_end,
        sleep_sec=float(sleep_sec)
    )

    st.subheader("掃描結果")

    col1, col2, col3 = st.columns(3)
    col1.metric("實際掃描", len(sampled) - len(skipped))
    col2.metric("訊號數", len(all_signals))
    col3.metric("耗時秒數", f"{total_time:.1f}")

    if skipped:
        st.caption(f"跳過無資料或失敗股票：{len(skipped)} 檔")

    if not all_signals:
        st.info("本次掃描無訊號。")
    else:
        result_df = (
            pd.DataFrame(all_signals)
            .sort_values(["子線日", "股票代號"], ascending=[False, True])
            .reset_index(drop=True)
        )

        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False, encoding="utf-8-sig")

        st.download_button(
            label="下載 CSV",
            data=csv.encode("utf-8-sig"),
            file_name=f"inside_bar_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
else:
    st.info("設定完成後，請按左側「開始掃描」。")
