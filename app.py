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
        "desc": "上影召阳：连续出现的4根周阴线（允许其中某一根是收盘等于开盘的“阴十字”也算在连续段里）；最后一根周K的上影线至少达到实体的50%，并且它要么触及上一周的中心价、要么在区间意义上吞没上一周（用最高/最低价范围来判断吞没）；最后一根的中心值要求落在上影区间内——可以贴着上影顶端算进去，但不允许刚好卡在上影起点那条边界；倒数第二根（第3根）要求实体下跌至少约5%，同时它的下影线很短（不超过股价约3.5%）；四根阴线合起来，从第一根开盘到第四根收盘的累计跌幅要超过20%；最后一根实体本身要很小（不超过股价约1%，股价参照用收盘价来算）。",
    },
    "三阴见底": {
        "tag": "三阴见底",
        "events_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E4%B8%89%E9%98%B4%E8%A7%81%E5%BA%95/events_slim_sanyin.parquet",
        "weekly_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E4%B8%89%E9%98%B4%E8%A7%81%E5%BA%95/weekly_cache_sanyin.parquet",
        "highlight_offsets": [-2, -1, 0],
        "pattern_n": 3,
        "desc": "三阴见底：连续出现的3根周阴线；前两根周阴线对跌幅不设最低要求，但上下影线相对实体不超过约四分之一；最后一根周阴线要求上影线不超过实体的10%，同时必须是长下影（下影长度至少达到实体的2倍），并且中心值允许更靠近收盘价一侧。",
    },
    "双阴变势": {
        "tag": "双阴变势",
        "events_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E5%8F%8C%E9%98%B4%E5%8F%98%E5%8A%BF/events_slim_shuangyin.parquet",
        "weekly_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E5%8F%8C%E9%98%B4%E5%8F%98%E5%8A%BF/weekly_cache_shuangyin.parquet",
        "highlight_offsets": [-1, 0],
        "pattern_n": 2,
        "desc": "双阴变势：信号窗口前面先允许紧挨着出现1到2根周阴线当“铺垫”；随后进入“双阴”本体两根周K：第一根（A）必须是带长下影的周阴线（下影至少是实体的1.2倍），同时上影很短（不超过实体的30%）；第二根（B）允许是阴十字/小实体阴线（实体占比≤0.15也算十字/小实体），并且它必须明显弱于A：实体不超过A的70%、整根K线总长度不超过A（你这里放到1.00，等于允许差不多一样长，但不能更长）；最后还有一个硬条件是“孕育”：B必须完全落在A的高低区间里（高点不高于A、低点不低于A），不允许越界。",
    },
    "双影上扬": {
        "tag": "双影上扬",
        "events_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E5%8F%8C%E5%BD%B1%E4%B8%8A%E6%89%AC/events_slim_sysy.parquet",
        "weekly_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E5%8F%8C%E5%BD%B1%E4%B8%8A%E6%89%AC/weekly_cache_sysy.parquet",
        "highlight_offsets": [-2, -1, 0],
        "pattern_n": 3,
        "desc": "双影上扬：第一根如果是周阳线，那它必须是偏“小”的阳线（实体占比不超过0.3、开到收涨幅不超过3%），否则形态直接不成立；后两根必须是周阳线，而且两根都要求影线很克制——上影线不超过实体的20%，下影线不超过实体的10%，并且下影必须小于上影（允许下影为0的光脚阳线），同时中心值必须落在阳线实体内部；信号周之前的过去5周（不含信号周）收对收累计涨跌幅的绝对值不超过5%。",
    },
    "阳线加阳": {
        "tag": "阳线加阳",
        "events_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E9%98%B3%E7%BA%BF%E5%8A%A0%E9%98%B3/events_slim_yxjy.parquet",
        "weekly_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E9%98%B3%E7%BA%BF%E5%8A%A0%E9%98%B3/weekly_cache_yxjy.parquet",
        "highlight_offsets": [-2, -1, 0],
        "pattern_n": 3,
        "desc": "阳线加阳：第一根阳线 w1 必须很强——实体不能太小，且开到收至少涨 10%，同时 w1 的实体还要比前一周 w0 的实体大 1.2 倍、全周振幅不比 w0 小；中间那根 w2 必须明显短小，实体最多只有 w1 的 35%，如果 w2 是阴线则只能落在你定义的两类情形里（要么向上跳空且阴线实体短、收盘贴近上一周收盘并且不能跌破 w1 的中心值；要么是孕线阴线且实体相对影线更“敦实”、收盘在 w1 中心附近或更低），如果 w2 是小阳或十字则只要足够短就放行；第三根 w3 必须是阳线，并且当 w2 是小阳时你现在要求 w3 的实体至少和 w1 一样大（因为你把比例设成 1），当 w2 是十字时还要求 w3 与 w2 形成孕线；最后成交量列会在候选列里自动匹配，匹配不到就直接不会触发。",
    },
    "上影孕阳": {
        "tag": "上影孕阳",
        "events_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E4%B8%8A%E5%BD%B1%E5%AD%95%E9%98%B3/events_slim_syyy.parquet",
        "weekly_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E4%B8%8A%E5%BD%B1%E5%AD%95%E9%98%B3/weekly_cache_syyy.parquet",
        "highlight_offsets": [-1, 0],
        "pattern_n": 2,
        "desc": "上影孕阳：形态第一根 w1 必须是阳线，且开到收至少涨 3%，同时整根振幅至少达到开盘价的 6%，并且上影要很长（上影 ≥ 实体的 1.8 倍），下影不作要求（设为 0 等于允许没下影）。第二根 w2 也必须是阳线，但要明显更短（全长和实体都不超过 w1 的 40%），并且完全被 w1 包住（孕线：H2≤H1 且 L2≥L1）。最后还要求 w2 的周成交量/成交额必须显著小于 w1（≤ 65%）。",
    },
    "阴阳齐天": {
        "tag": "阴阳齐天",
        "events_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E9%98%B4%E9%98%B3%E9%BD%90%E5%A4%A9/events_slim_yyqt.parquet",
        "weekly_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E9%98%B4%E9%98%B3%E9%BD%90%E5%A4%A9/weekly_cache_yyqt.parquet",
        "highlight_offsets": [-2, -1, 0],
        "pattern_n": 3,
        "desc": "阴阳齐天：这个因子只在周末行找“先阳后阴”的两周组合（w1 是阳线、w2 是阴线），并且两根K线不能太小（实体至少 0.01、全长至少 0.02），同时要求它们在“力度”和“波动”上看起来差不多——也就是 w2 的实体涨幅（|C−O|/O）和整根振幅（(H−L)/O）都要落在 w1 的 0.8～1.2 倍区间里；另外影线要短（上下影都不超过实体的 0.25 倍），w2 的最高价允许略微高于 w1，但最多只能高 3%。w1 前一周 w0 要么也是一根“和 w1 相仿”的阳线（同样用 0.8～1.2 倍区间衡量），要么是明显更小的阴线（它的振幅和实体涨幅都不超过 w1 的 60%）；最后成交量必须配合：w2 的周成交量要显著小于 w1（不超过 w1 的 65%），且成交量必须为正。",
    },
    "下影雨滴": {
        "tag": "下影雨滴",
        "events_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E4%B8%8B%E5%BD%B1%E9%9B%A8%E6%BB%B4/events_slim_xyyd.parquet",
        "weekly_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E4%B8%8B%E5%BD%B1%E9%9B%A8%E6%BB%B4/weekly_cache_xyyd.parquet",
        "highlight_offsets": [-2, -1, 0],
        "pattern_n": 3,
        "desc": "下影雨滴：第一根关键周K（w1）必须是明显的大阴线：开到收至少跌 4%，实体占全长至少 35%，且还要比上一周至少大 10% 才行。同时 w1 必须有长下影：下影长度至少是实体的 1.5 倍，上影允许有但必须不超过下影的 50%。后面两根周K（w2、w3）不管阴阳都行，但要明显短小：它们的振幅和实体都不能超过 w1 的 70%，而且这两周的最低价都不能跌破 w1 的最低价。",
    },
    "下跳两点": {
        "tag": "下跳两点",
        "events_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E4%B8%8B%E8%B7%B3%E4%B8%A4%E7%82%B9/events_slim_xtld.parquet",
        "weekly_url": "https://github.com/qiwu5196/imagecheck-data/releases/download/%E4%B8%8B%E8%B7%B3%E4%B8%A4%E7%82%B9/weekly_cache_xtld.parquet",
        "highlight_offsets": [-2, -1, 0],
        "pattern_n": 3,
        "desc": "下跳两点：形态里第一根 w1 必须是很强的阴线（开到收至少跌 5%，而且实体占全长至少 75%）。第二根 w2 必须向下跳空开盘（开盘价低于 w1 收盘价即可），并且只能是阴线或十字（十字要求实体/全长 ≤ 0.12），同时 w2 必须明显短于 w1（全长、实体都不超过 w1 的 40%）。第三根 w3 只要满足“要么比 w2 更低、要么被 w2 完全包住（孕线）”两种之一即可；最后还要求 w3 的周成交量/成交额（会在候选列里自动找得到的那一列）至少是 w2 的 1.2倍且必须为正。",
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

st.markdown("### 作者最新话：上下键切换功能加不进去啊..算了能用就行")

st.caption(f"当前：{dataset_name}（events={len(ev)}，weekly rows={len(weekly_all)}）")

# ✅形态定义：显示在事件表上方
desc = str(cfg.get("desc", "")).strip()
if desc:
    st.info(desc)

st.subheader("事件表（列表左侧勾选切换股票）")

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
sig_week = str(row.get("week_id_str", ""))

c1, c2, c3 = st.columns(3)
c1.metric("股票代码", code)
c2.metric("股票名称", sname)
c3.metric("信号周", sig_week)

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

st.subheader(f"形态 {pattern_n} 根周涨跌幅（上周收盘对本周收盘）")
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

st.subheader(
    f"形态 {pattern_n} 根周实体跌幅（本周开盘对本周收盘）-对齐东财版-此版本不会计入周开盘时的高开低开对整周涨跌幅的影响"
)
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

st.subheader("该股表格包含内容（如需全股票完整表格请找作者:D）")
st.json({k: row.get(k) for k in cols})
