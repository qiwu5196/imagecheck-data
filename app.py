# -*- coding: utf-8 -*-
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="此处路过一位形态查看器", layout="wide")

DATA_DIR = Path("data_cache")
DATA_DIR.mkdir(exist_ok=True)

# =========================
# 数据集配置：以后加因子就往这里加
# =========================
DATASETS = {
    "上影召阳": {
        "tag": "v1",
        "events_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/v1/events_slim.parquet",
        "weekly_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/v1/weekly_cache.parquet",
        "highlight_offsets": [-3, -2, -1, 0],
        "pattern_n": 4,
    },
    "三阴见底": {
        "tag": "三阴见底",
        "events_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E4%B8%89%E9%98%B4%E8%A7%81%E5%BA%95/events_slim_sanyin.parquet",
        "weekly_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E4%B8%89%E9%98%B4%E8%A7%81%E5%BA%95/weekly_cache_sanyin.parquet",
        "highlight_offsets": [-2, -1, 0],
        "pattern_n": 3,
    },
    "双阴变势": {
        "tag": "双阴变势",
        "events_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E5%8F%8C%E9%98%B4%E5%8F%98%E5%8A%BF/events_slim_shuangyin.parquet",
        "weekly_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E5%8F%8C%E9%98%B4%E5%8F%98%E5%8A%BF/weekly_cache_shuangyin.parquet",
        "highlight_offsets": [-1, 0],
        "pattern_n": 2,
    },
}

# =========================
# 小工具
# =========================
def safe_name(s: str) -> str:
    return str(s).encode("utf-8").hex()

def fetch(url: str, out_path: Path):
    """存在且非空就不下载"""
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_path.as_posix())

@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def _wk_body_drop(open_: float, close_: float) -> float:
    if open_ is None or close_ is None:
        return np.nan
    try:
        o = float(open_)
        c = float(close_)
    except Exception:
        return np.nan
    if o <= 0:
        return np.nan
    return (o - c) / o

def _prep_weekly_index(weekly_all: pd.DataFrame, code_col: str = "股票代码"):
    wk = weekly_all.copy()
    wk[code_col] = wk[code_col].astype(str)
    wk["week_id_str"] = wk["week_id_str"].astype(str)
    wk["date"] = pd.to_datetime(wk["date"], errors="coerce")

    for c in ["open", "high", "low", "close"]:
        wk[c] = pd.to_numeric(wk[c], errors="coerce")

    if "ret_c2c" in wk.columns:
        wk["ret_c2c"] = pd.to_numeric(wk["ret_c2c"], errors="coerce")

    wk = wk.dropna(subset=["date", "open", "high", "low", "close"])

    grouped = {}
    for code, g in wk.groupby(code_col, sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        g["prev_close"] = g["close"].shift(1)

        if "ret_c2c" not in g.columns:
            pc = g["prev_close"]
            g["ret_c2c"] = np.where(
                (pc > 0) & (g["close"].notna()) & (pc.notna()),
                (g["close"] / pc) - 1.0,
                np.nan
            )
        else:
            calc = np.where(
                (g["prev_close"] > 0) & (g["close"].notna()) & (g["prev_close"].notna()),
                (g["close"] / g["prev_close"]) - 1.0,
                np.nan
            )
            g["ret_c2c"] = g["ret_c2c"].where(g["ret_c2c"].notna(), calc)

        pos_map = {wid: i for i, wid in enumerate(g["week_id_str"].tolist())}
        grouped[code] = (g, pos_map)
    return grouped

def plot_candles(
    weekly: pd.DataFrame,
    signal_week_id_str: str,
    left_weeks: int,
    right_weeks: int,
    highlight_offsets=None,
    stock_code: str = "",
    stock_name: str = "",
):
    weekly = weekly.sort_values("date").reset_index(drop=True)
    if weekly.empty:
        return None

    pos = weekly.index[weekly["week_id_str"] == str(signal_week_id_str)]
    if len(pos) == 0:
        return None
    p = int(pos[0])

    lo = max(0, p - left_weeks)
    hi = min(len(weekly) - 1, p + right_weeks)
    sub = weekly.iloc[lo:hi + 1].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=sub["date"],
            open=sub["open"],
            high=sub["high"],
            low=sub["low"],
            close=sub["close"],
            name="周K",
            increasing_line_color="red",
            increasing_fillcolor="red",
            decreasing_line_color="green",
            decreasing_fillcolor="green",
        )
    )

    if highlight_offsets is None:
        highlight_offsets = [-2, -1, 0]
    highlight_idx = [p + int(off) for off in highlight_offsets]
    highlight_idx = [i for i in highlight_idx if lo <= i <= hi]

    for i in highlight_idx:
        d = weekly.at[i, "date"]
        fig.add_vrect(
            x0=d - pd.Timedelta(days=1),
            x1=d + pd.Timedelta(days=1),
            fillcolor="rgba(255, 200, 0, 0.18)",
            line_width=0,
            layer="below",
        )

    # ✅触发周虚线：更细、更浅、放到下面
    sig_date = weekly.at[p, "date"]
    fig.add_vline(
        x=sig_date,
        line_width=1,
        line_dash="dash",
        line_color="rgba(0,0,0,0.35)",
        layer="below",
    )

    title_text = f"{stock_code} {stock_name}（信号周：{signal_week_id_str}）".strip()

    fig.update_layout(
        height=620,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        title=title_text,
    )
    return fig

def fill_pattern_week_returns_c2c_from_weekly(
    ev: pd.DataFrame,
    weekly_all: pd.DataFrame,
    n: int,
    week_id_col: str = "week_id_str",
    code_col: str = "股票代码",
) -> pd.DataFrame:
    ev = ev.copy()
    need_cols = [f"形态第{i}根周阴线涨跌幅" for i in range(1, n + 1)]
    miss = [c for c in need_cols if c not in ev.columns]
    if not miss:
        return ev

    grouped = _prep_weekly_index(weekly_all, code_col=code_col)

    def _calc_row(row):
        code = str(row.get(code_col, ""))
        wid = str(row.get(week_id_col, ""))
        if code not in grouped:
            return pd.Series({c: np.nan for c in need_cols})
        g, pos_map = grouped[code]
        p = pos_map.get(wid, None)
        if p is None:
            return pd.Series({c: np.nan for c in need_cols})

        out = {}
        for i in range(1, n + 1):
            idx = p - (n - i)
            col = f"形态第{i}根周阴线涨跌幅"
            out[col] = g.at[idx, "ret_c2c"] if 0 <= idx < len(g) else np.nan
        return pd.Series(out)

    add = ev.apply(_calc_row, axis=1)
    for c in need_cols:
        if c not in ev.columns:
            ev[c] = add[c].values
    return ev

def fill_forward_5w_returns_c2c_from_weekly(
    ev: pd.DataFrame,
    weekly_all: pd.DataFrame,
    week_id_col: str = "week_id_str",
    code_col: str = "股票代码",
) -> pd.DataFrame:
    ev = ev.copy()
    need_cols = [f"触发后第{i}周涨跌幅" for i in range(1, 6)]
    miss = [c for c in need_cols if c not in ev.columns]
    if not miss:
        return ev

    grouped = _prep_weekly_index(weekly_all, code_col=code_col)

    def _calc_row(row):
        code = str(row.get(code_col, ""))
        wid = str(row.get(week_id_col, ""))
        if code not in grouped:
            return pd.Series({c: np.nan for c in need_cols})
        g, pos_map = grouped[code]
        p = pos_map.get(wid, None)
        if p is None:
            return pd.Series({c: np.nan for c in need_cols})

        out = {}
        for i in range(1, 6):
            idx = p + i
            col = f"触发后第{i}周涨跌幅"
            out[col] = g.at[idx, "ret_c2c"] if 0 <= idx < len(g) else np.nan
        return pd.Series(out)

    add = ev.apply(_calc_row, axis=1)
    for c in need_cols:
        if c not in ev.columns:
            ev[c] = add[c].values
    return ev

def fill_event_week_returns_from_weekly_if_missing(
    ev: pd.DataFrame,
    weekly_all: pd.DataFrame,
    week_id_col: str = "week_id_str",
    code_col: str = "股票代码",
) -> pd.DataFrame:
    ev = ev.copy()
    grouped = _prep_weekly_index(weekly_all, code_col=code_col)

    def _calc_row(row):
        code = str(row.get(code_col, ""))
        wid = str(row.get(week_id_col, ""))
        if code not in grouped:
            return pd.Series({"周涨幅_收对收": np.nan, "周涨幅_开到收": np.nan})
        g, pos_map = grouped[code]
        p = pos_map.get(wid, None)
        if p is None:
            return pd.Series({"周涨幅_收对收": np.nan, "周涨幅_开到收": np.nan})

        c2c = g.at[p, "ret_c2c"]
        o = g.at[p, "open"]
        c = g.at[p, "close"]
        o2c = (c - o) / o if (o is not None and pd.notna(o) and float(o) > 0) else np.nan
        return pd.Series({"周涨幅_收对收": c2c, "周涨幅_开到收": o2c})

    need_any = any(col not in ev.columns for col in ["周涨幅_收对收", "周涨幅_开到收"])
    if not need_any:
        if "周涨幅" not in ev.columns and "周涨幅_收对收" in ev.columns:
            ev["周涨幅"] = ev["周涨幅_收对收"]
        return ev

    add = ev.apply(_calc_row, axis=1)

    if "周涨幅_收对收" not in ev.columns:
        ev["周涨幅_收对收"] = add["周涨幅_收对收"].values
    if "周涨幅_开到收" not in ev.columns:
        ev["周涨幅_开到收"] = add["周涨幅_开到收"].values

    if "周涨幅" not in ev.columns:
        ev["周涨幅"] = ev["周涨幅_收对收"]
    return ev


# =========================
# UI
# =========================
st.sidebar.title("形态查看器")
st.sidebar.caption("首次加载会下载，之后走本地缓存；建议每次打开点一下‘清空所有缓存’，避免遗漏数据更新。")

dataset_name = st.sidebar.selectbox("选择形态/因子", list(DATASETS.keys()), index=0)
cfg = DATASETS[dataset_name]
pattern_n = int(cfg.get("pattern_n", 3))

ds_key = safe_name(dataset_name)
tag_key = safe_name(str(cfg.get("tag", "v")))
EVENTS_PATH = DATA_DIR / f"events__{ds_key}__{tag_key}.parquet"
WEEKLY_PATH = DATA_DIR / f"weekly__{ds_key}__{tag_key}.parquet"

cbtn1, cbtn2 = st.sidebar.columns(2)
if cbtn1.button("重新下载当前形态"):
    if EVENTS_PATH.exists():
        EVENTS_PATH.unlink()
    if WEEKLY_PATH.exists():
        WEEKLY_PATH.unlink()
    st.cache_data.clear()
    st.rerun()

if cbtn2.button("清空所有缓存"):
    for p in DATA_DIR.glob("*.parquet"):
        try:
            p.unlink()
        except Exception:
            pass
    st.cache_data.clear()
    st.rerun()

try:
    fetch(cfg["events_url"], EVENTS_PATH)
    fetch(cfg["weekly_url"], WEEKLY_PATH)
except Exception as e:
    st.error(f"数据下载失败：{e}")
    st.stop()

try:
    ev = load_parquet(str(EVENTS_PATH))
    weekly_all = load_parquet(str(WEEKLY_PATH))
except Exception as e:
    st.error(f"数据读取失败：{e}")
    st.stop()

# ---------------- 清洗 events ----------------
ev = ev.copy()
if "week_id_str" not in ev.columns:
    if "week_id" in ev.columns:
        ev["week_id_str"] = ev["week_id"].astype(str)
    else:
        ev["week_id_str"] = ""

if "信号打点日" in ev.columns:
    ev["信号打点日"] = pd.to_datetime(ev["信号打点日"], errors="coerce")

if "股票代码" in ev.columns:
    ev["股票代码"] = ev["股票代码"].astype(str)
if "股票名称" in ev.columns:
    ev["股票名称"] = ev["股票名称"].astype(str)

# ---------------- 清洗 weekly ----------------
weekly_all = weekly_all.copy()
if "股票代码" not in weekly_all.columns:
    st.error("weekly_cache 缺少 股票代码")
    st.stop()
weekly_all["股票代码"] = weekly_all["股票代码"].astype(str)

if "week_id_str" not in weekly_all.columns:
    if "week_id" in weekly_all.columns:
        weekly_all["week_id_str"] = weekly_all["week_id"].astype(str)
    else:
        st.error("weekly_cache 缺少 week_id_str/week_id，无法定位信号周")
        st.stop()
else:
    weekly_all["week_id_str"] = weekly_all["week_id_str"].astype(str)

if "date" not in weekly_all.columns:
    st.error("weekly_cache 缺少 date")
    st.stop()
weekly_all["date"] = pd.to_datetime(weekly_all["date"], errors="coerce")

for c in ["open", "high", "low", "close"]:
    if c not in weekly_all.columns:
        st.error(f"weekly_cache 缺少 {c}")
        st.stop()

if "ret_c2c" in weekly_all.columns:
    weekly_all["ret_c2c"] = pd.to_numeric(weekly_all["ret_c2c"], errors="coerce")

weekly_all = weekly_all.dropna(subset=["date", "open", "high", "low", "close"])

# ✅补列
ev = fill_event_week_returns_from_weekly_if_missing(ev, weekly_all)
ev = fill_pattern_week_returns_c2c_from_weekly(ev, weekly_all, n=pattern_n)
ev = fill_forward_5w_returns_c2c_from_weekly(ev, weekly_all)

# =========================
# 侧边栏参数
# =========================
left_weeks = st.sidebar.slider("向前显示周数", 5, 200, 40, 5)
right_weeks = st.sidebar.slider("向后显示周数", 5, 200, 40, 5)
keyword = st.sidebar.text_input("筛选（股票代码/股票名称）", value="")

# =========================
# 事件表筛选
# =========================
show = ev
if keyword.strip():
    kw = keyword.strip()
    mask = False
    if "股票代码" in show.columns:
        mask = mask | show["股票代码"].astype(str).str.contains(kw, na=False)
    if "股票名称" in show.columns:
        mask = mask | show["股票名称"].astype(str).str.contains(kw, na=False)
    show = show.loc[mask].copy()

base_cols = ["股票代码", "股票名称", "信号打点日", "week_id_str"]
pattern_drop_cols = [f"形态第{i}根周阴线涨跌幅" for i in range(1, pattern_n + 1)]
forward_cols = [f"触发后第{i}周涨跌幅" for i in range(1, 6)]
misc_cols = ["形态长度", "先导阴线数", "周涨幅_收对收", "周涨幅_开到收", "周涨幅", "周实体占比"]

prefer_cols = base_cols + misc_cols + pattern_drop_cols + forward_cols
cols = [c for c in prefer_cols if c in show.columns]
if not cols:
    cols = show.columns.tolist()

st.title("作者最新话：上下键切换功能加不进去啊..算了能用就行")
st.caption(f"当前：{dataset_name}（events={len(ev)}，weekly rows={len(weekly_all)}）")

st.subheader("事件表（点击一行查看形态）")

if "信号打点日" in show.columns and "股票代码" in show.columns:
    show = show.sort_values(["信号打点日", "股票代码"], ascending=[False, True])
show = show.reset_index(drop=True)

if show.empty:
    st.warning("没有匹配到事件记录，请换个关键词或切换形态。")
    st.stop()

event = st.dataframe(
    show[cols],
    use_container_width=True,
    height=360,
    on_select="rerun",
    selection_mode="single-row",
)

selected_rows = event.selection.get("rows", []) if hasattr(event, "selection") else []
row_id = int(selected_rows[0]) if selected_rows else 0
row_id = max(0, min(row_id, len(show) - 1))
row = show.iloc[row_id].to_dict()

code = str(row.get("股票代码", ""))
sname = str(row.get("股票名称", ""))
sig_week = str(row.get("week_id_str", ""))  # ✅你原来漏了这个，导致直接报错

c1, c2, c3 = st.columns(3)
c1.metric("股票代码", code)
c2.metric("股票名称", sname)
c3.metric("信号周 week_id", sig_week)

weekly = weekly_all.loc[weekly_all["股票代码"] == code].copy()
weekly = weekly.sort_values("date").reset_index(drop=True)

if weekly.empty:
    st.error(f"周线缓存里找不到该股票：{code}")
    st.stop()

fig = plot_candles(
    weekly,
    sig_week,
    left_weeks=left_weeks,
    right_weeks=right_weeks,
    highlight_offsets=cfg.get("highlight_offsets", [-2, -1, 0]),
    stock_code=code,
    stock_name=sname,
)
if fig is None:
    st.error("没找到该事件对应的信号周（week_id 不匹配）。")
    st.stop()

st.subheader("周K线图（高亮为形态相关周）")
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"形态 {pattern_n} 根周涨跌幅（收对收）")
drop_show = {c: row.get(c) for c in pattern_drop_cols if c in row}
if drop_show:
    pretty = {}
    for k, v in drop_show.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            pretty[k] = None
        else:
            pretty[k] = f"{float(v) * 100:.2f}%"
    st.json(pretty)
else:
    st.info("事件表里没有形态涨跌幅列，且自动补算未成功（可能 weekly_cache 缺列或 week_id 对不上）。")

st.subheader(f"形态 {pattern_n} 根周实体跌幅（开到收）")
try:
    pos = weekly.index[weekly["week_id_str"] == str(sig_week)]
    if len(pos) > 0:
        p = int(pos[0])
        idxs = [p - (pattern_n - i) for i in range(1, pattern_n + 1)]
        body_pretty = {}
        for i, idx in enumerate(idxs, 1):
            if idx < 0 or idx >= len(weekly):
                body_pretty[f"形态第{i}根实体跌幅"] = None
            else:
                bd = _wk_body_drop(weekly.at[idx, "open"], weekly.at[idx, "close"])
                body_pretty[f"形态第{i}根实体跌幅"] = None if (bd is None or (isinstance(bd, float) and np.isnan(bd))) else f"{bd*100:.2f}%"
        st.json(body_pretty)
    else:
        st.info("无法定位信号周，未计算实体跌幅对照。")
except Exception:
    st.info("实体跌幅对照计算失败（不影响主功能）。")

st.subheader("该事件关键字段")
st.json({k: row.get(k) for k in cols})
