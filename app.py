# -*- coding: utf-8 -*-
from pathlib import Path
import urllib.request

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="形态查看器", layout="wide")

DATA_DIR = Path("data_cache")
DATA_DIR.mkdir(exist_ok=True)

# ✅ 写死你的 Release 直链（最省事）
EVENTS_URL = "https://github.com/qiwu5196/imagecheck-data/releases/download/v1/events_slim.parquet"
WEEKLY_URL = "https://github.com/qiwu5196/imagecheck-data/releases/download/v1/weekly_cache.parquet"

EVENTS_PATH = DATA_DIR / "events_slim.parquet"
WEEKLY_PATH = DATA_DIR / "weekly_cache.parquet"


def fetch(url: str, out_path: Path):
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_path.as_posix())


@st.cache_data(show_spinner=False)
def load_events(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_weekly(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def plot_candles(weekly: pd.DataFrame, signal_week_id_str: str, left_weeks: int, right_weeks: int):
    weekly = weekly.sort_values("date").reset_index(drop=True)
    if weekly.empty:
        return None

    pos = weekly.index[weekly["week_id_str"] == signal_week_id_str]
    if len(pos) == 0:
        return None
    p = int(pos[0])

    lo = max(0, p - left_weeks)
    hi = min(len(weekly) - 1, p + right_weeks)
    sub = weekly.iloc[lo:hi + 1].copy()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=sub["date"],
        open=sub["open"],
        high=sub["high"],
        low=sub["low"],
        close=sub["close"],
        name="周K",
        # A股：红涨绿跌
        increasing_line_color="red",
        increasing_fillcolor="red",
        decreasing_line_color="green",
        decreasing_fillcolor="green",
    ))

    # 高亮 t-2 / t-1 / t（信号周）
    highlight_idx = [p - 2, p - 1, p]
    highlight_idx = [i for i in highlight_idx if lo <= i <= hi]
    for i in highlight_idx:
        d = weekly.at[i, "date"]
        fig.add_vrect(
            x0=d - pd.Timedelta(days=1),
            x1=d + pd.Timedelta(days=1),
            fillcolor="rgba(255, 200, 0, 0.18)",
            line_width=0
        )

    sig_date = weekly.at[p, "date"]
    fig.add_vline(x=sig_date, line_width=2, line_dash="dash")

    fig.update_layout(
        height=620,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"周线K图（信号周：{signal_week_id_str}）"
    )
    return fig


# ---------------- UI ----------------
st.sidebar.title("形态查看器（在线版）")
st.sidebar.caption("首次打开会自动下载数据，之后走缓存。")

if st.sidebar.button("重新下载数据"):
    if EVENTS_PATH.exists():
        EVENTS_PATH.unlink()
    if WEEKLY_PATH.exists():
        WEEKLY_PATH.unlink()
    st.cache_data.clear()
    st.rerun()

# 下载并加载
try:
    fetch(EVENTS_URL, EVENTS_PATH)
    fetch(WEEKLY_URL, WEEKLY_PATH)
except Exception as e:
    st.error(f"数据下载失败：{e}")
    st.stop()

try:
    ev = load_events(str(EVENTS_PATH))
    weekly_all = load_weekly(str(WEEKLY_PATH))
except Exception as e:
    st.error(f"数据读取失败：{e}")
    st.stop()

# 清洗
ev = ev.copy()
if "week_id_str" not in ev.columns:
    if "week_id" in ev.columns:
        ev["week_id_str"] = ev["week_id"].astype(str)
    else:
        ev["week_id_str"] = ""

if "信号打点日" in ev.columns:
    ev["信号打点日"] = pd.to_datetime(ev["信号打点日"], errors="coerce")

weekly_all = weekly_all.copy()
weekly_all["股票代码"] = weekly_all["股票代码"].astype(str)
weekly_all["week_id_str"] = weekly_all["week_id_str"].astype(str)
weekly_all["date"] = pd.to_datetime(weekly_all["date"], errors="coerce")
weekly_all = weekly_all.dropna(subset=["date", "open", "high", "low", "close"])

left_weeks = st.sidebar.slider("向前显示周数", 5, 120, 40, 1)
right_weeks = st.sidebar.slider("向后显示周数", 5, 120, 20, 1)
keyword = st.sidebar.text_input("筛选（股票代码/股票名称）", value="")

show = ev
if keyword.strip():
    kw = keyword.strip()
    mask = False
    if "股票代码" in show.columns:
        mask = mask | show["股票代码"].astype(str).str.contains(kw, na=False)
    if "股票名称" in show.columns:
        mask = mask | show["股票名称"].astype(str).str.contains(kw, na=False)
    show = show.loc[mask].copy()

prefer_cols = [
    "股票代码","股票名称","信号打点日","week_id_str",
    "形态第1根周阴线涨跌幅","形态第2根周阴线涨跌幅",
    "触发后第1周涨跌幅","触发后第2周涨跌幅","触发后第3周涨跌幅","触发后第4周涨跌幅","触发后第5周涨跌幅"
]
cols = [c for c in prefer_cols if c in show.columns]
if not cols:
    cols = show.columns.tolist()

st.subheader("事件表（点击一行查看形态）")

if "信号打点日" in show.columns and "股票代码" in show.columns:
    show = show.sort_values(["信号打点日","股票代码"], ascending=[False, True])
show = show.reset_index(drop=True)

event = st.dataframe(
    show[cols],
    use_container_width=True,
    height=360,
    on_select="rerun",
    selection_mode="single-row",
)

selected_rows = event.selection.get("rows", []) if hasattr(event, "selection") else []
row_id = int(selected_rows[0]) if selected_rows else 0
row = show.iloc[row_id].to_dict()

code = str(row.get("股票代码",""))
sig_week = str(row.get("week_id_str",""))

c1, c2, c3 = st.columns(3)
c1.metric("股票代码", code)
c2.metric("信号周 week_id", sig_week)
c3.metric("信号打点日", str(row.get("信号打点日","")))

weekly = weekly_all.loc[weekly_all["股票代码"] == code].copy()
weekly = weekly.sort_values("date").reset_index(drop=True)

if weekly.empty:
    st.error(f"周线缓存里找不到该股票：{code}")
    st.stop()

fig = plot_candles(weekly, sig_week, left_weeks=left_weeks, right_weeks=right_weeks)
if fig is None:
    st.error("没找到该事件对应的信号周（week_id 不匹配）。")
    st.stop()

st.subheader("周线K图（高亮 t-2/t-1/t）")
st.plotly_chart(fig, use_container_width=True)

st.subheader("该事件关键字段")
st.json({k: row.get(k) for k in cols})

