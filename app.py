# -*- coding: utf-8 -*-
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="形态查看器", layout="wide")

DATA_DIR = Path("data_cache")
DATA_DIR.mkdir(exist_ok=True)

# =========================
# 数据集配置：以后加因子就往这里加
# - tag：你发新版本/新release就改tag（会自动用新缓存文件名，避免旧缓存干扰）
# - highlight_offsets：高亮哪些周（相对信号周 p 的偏移）
# =========================
DATASETS = {
    "上影召阳": {
        "tag": "v1",  
        "events_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/v1/events_slim.parquet",
        "weekly_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/v1/weekly_cache.parquet",
        "highlight_offsets": [-3, -2, -1, 0],  # ✅四根：t-3/t-2/t-1/t
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
    """稳定：永远生成纯英文数字文件名（避免中文/特殊字符导致跨平台问题）"""
    return s.encode("utf-8").hex()

def fetch(url: str, out_path: Path):
    """存在且非空就不下载"""
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_path.as_posix())

@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def _wk_drop(open_: float, close_: float) -> float:
    """周跌幅（实体方向）：(open-close)/open；open<=0 返回 nan"""
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

def plot_candles(
    weekly: pd.DataFrame,
    signal_week_id_str: str,
    left_weeks: int,
    right_weeks: int,
    highlight_offsets=None,
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
            # A股：红涨绿跌
            increasing_line_color="red",
            increasing_fillcolor="red",
            decreasing_line_color="green",
            decreasing_fillcolor="green",
        )
    )

    # 高亮区间（默认信号周 p 的若干偏移）
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
        )

    sig_date = weekly.at[p, "date"]
    fig.add_vline(x=sig_date, line_width=2, line_dash="dash")

    fig.update_layout(
        height=620,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"周线K图（信号周：{signal_week_id_str}）",
    )
    return fig

def fill_pattern_week_drops_from_weekly(
    ev: pd.DataFrame,
    weekly_all: pd.DataFrame,
    n: int,
    week_id_col: str = "week_id_str",
    code_col: str = "股票代码",
) -> pd.DataFrame:
    """
    如果 events 里缺少「形态第X根周阴线涨跌幅」列，则用 weekly_cache 自动补齐。
    规则：以信号周为 p，往前取 n-1 ... 0 共 n 根，对应：
      第1根 = t-(n-1)
      ...
      第n根 = t
    跌幅定义：(open-close)/open （小数：0.055=5.5%）
    """
    ev = ev.copy()
    need_cols = [f"形态第{i}根周阴线涨跌幅" for i in range(1, n + 1)]
    miss = [c for c in need_cols if c not in ev.columns]
    if not miss:
        return ev

    # 构建 (code, week_id_str) -> 在该股票weekly里的位置p
    wk = weekly_all[[code_col, "week_id_str", "open", "close", "date"]].copy()
    wk["week_id_str"] = wk["week_id_str"].astype(str)
    wk[code_col] = wk[code_col].astype(str)

    # 每只股票建立索引映射：week_id_str -> idx
    grouped = {}
    for code, g in wk.groupby(code_col, sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        pos_map = {wid: i for i, wid in enumerate(g["week_id_str"].tolist())}
        grouped[code] = (g, pos_map)

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
        # 第1根是 t-(n-1)
        for i in range(1, n + 1):
            idx = p - (n - i)
            col = f"形态第{i}根周阴线涨跌幅"
            if idx < 0 or idx >= len(g):
                out[col] = np.nan
            else:
                out[col] = _wk_drop(g.at[idx, "open"], g.at[idx, "close"])
        return pd.Series(out)

    add = ev.apply(_calc_row, axis=1)
    for c in need_cols:
        if c not in ev.columns:
            ev[c] = add[c].values
    return ev


# =========================
# UI
# =========================
st.sidebar.title("形态查看器（在线版）")
st.sidebar.caption("切换形态会加载对应的数据；首次会下载，之后走本地缓存。")

dataset_name = st.sidebar.selectbox("选择形态/因子", list(DATASETS.keys()), index=0)
cfg = DATASETS[dataset_name]
pattern_n = int(cfg.get("pattern_n", 3))

# 每个数据集 + tag 用不同缓存文件名，互不覆盖；tag 改了会自动用新缓存
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

# 下载并加载（当前数据集）
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

# 兼容：week_id_str / week_id
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

# 必需 OHLC
for c in ["open", "high", "low", "close"]:
    if c not in weekly_all.columns:
        st.error(f"weekly_cache 缺少 {c}")
        st.stop()

weekly_all = weekly_all.dropna(subset=["date", "open", "high", "low", "close"])

# ✅ 如果当前形态需要“形态第1..第N根周阴线涨跌幅”，但events里没有，就自动补齐
ev = fill_pattern_week_drops_from_weekly(ev, weekly_all, n=pattern_n)

# =========================
# 侧边栏参数
# =========================
left_weeks = st.sidebar.slider("向前显示周数", 5, 120, 40, 1)
right_weeks = st.sidebar.slider("向后显示周数", 5, 120, 20, 1)
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

# 展示列：尽量通用
base_cols = ["股票代码", "股票名称", "信号打点日", "week_id_str"]
pattern_drop_cols = [f"形态第{i}根周阴线涨跌幅" for i in range(1, pattern_n + 1)]
forward_cols = [
    "触发后第1周涨跌幅", "触发后第2周涨跌幅", "触发后第3周涨跌幅",
    "触发后第4周涨跌幅", "触发后第5周涨跌幅",
]
misc_cols = ["形态长度", "先导阴线数", "周涨幅", "周实体占比"]

prefer_cols = base_cols + misc_cols + pattern_drop_cols + forward_cols
cols = [c for c in prefer_cols if c in show.columns]
if not cols:
    cols = show.columns.tolist()

st.title("形态查看器")
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
sig_week = str(row.get("week_id_str", ""))

c1, c2, c3 = st.columns(3)
c1.metric("股票代码", code)
c2.metric("信号周 week_id", sig_week)
c3.metric("信号打点日", str(row.get("信号打点日", "")))

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
)
if fig is None:
    st.error("没找到该事件对应的信号周（week_id 不匹配）。")
    st.stop()

st.subheader("周线K图（高亮形态相关周）")
st.plotly_chart(fig, use_container_width=True)

# ===== 额外：把形态的 N 根周K 的跌幅在下方也单独列一下（方便核对）=====
st.subheader(f"形态 {pattern_n} 根周阴线跌幅（实体跌幅 = (open-close)/open）")
drop_show = {c: row.get(c) for c in pattern_drop_cols if c in row}
if drop_show:
    # 美化成百分比展示
    pretty = {}
    for k, v in drop_show.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            pretty[k] = None
        else:
            pretty[k] = f"{float(v) * 100:.2f}%"
    st.json(pretty)
else:
    st.info("事件表里没有形态跌幅列，且自动补算未成功（可能 weekly_cache 缺列或 week_id 对不上）。")

st.subheader("该事件关键字段")
st.json({k: row.get(k) for k in cols})
