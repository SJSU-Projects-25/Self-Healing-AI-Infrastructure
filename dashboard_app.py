"""
LENS Monitor Dashboard — dashboard_app.py
=========================================
Clean rewrite focused on label-aware telemetry visualization.

Usage:
    streamlit run dashboard_app.py

Data layout expected (relative to this script):
    data/
        Label0/   telemetry_<node>_<jobid>_label0.csv   (healthy)
        Label1/   telemetry_<node>_<jobid>_label1.csv   (allreduce_pause)
        Label2/   telemetry_<node>_<jobid>_label2.csv   (IB contention)
        Label5/   telemetry_<node>_<jobid>_label5.csv   (idle)

Key CSV columns (0-indexed):
    0   timestamp
    1   node_id
    2   job_id
    6   gpu_util_pct
    26  ib_port_xmit_data_delta      — Data Sent Rate
    30  ib_port_xmit_wait_delta      — Congestion Backpressure
    33  ib_symbol_error_delta        — Physical Link Errors
    35  ib_sq_num_rnr_delta          — RDMA Stalls
    -1  label
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Map label number → folder name, display name, and semantic meaning
LABEL_CONFIG: dict[int, dict] = {
    0: {
        "folder": "Label0",
        "name": "Healthy Training",
        "short": "healthy",
        "color": "#22c55e",
        "description": "GPU util 99–100%, sustained IB throughput",
    },
    1: {
        "folder": "Label1",
        "name": "AllReduce Pause",
        "short": "allreduce_pause",
        "color": "#f59e0b",
        "description": "Brief GPU util drop to 0% during collective sync",
    },
    2: {
        "folder": "Label2",
        "name": "IB Network Contention",
        "short": "ib_contention",
        "color": "#ef4444",
        "description": "IB transmit waiting rises and DDP steps slow during contention",
    },
    5: {
        "folder": "Label5",
        "name": "Idle",
        "short": "idle",
        "color": "#94a3b8",
        "description": "GPU util 0%, xmit_data very low (no training job)",
    },
}

# Metric definitions: (csv_column_name, display_name, y-axis label)
METRICS = [
    ("gpu_util_pct",            "GPU Utilization",                    "GPU util (%)"),
    ("ib_port_xmit_data_delta", "IB Traffic Volume (communication activity)", "bytes sent / poll"),
    ("ib_port_xmit_packets_delta", "IB Packet Activity (communication activity)", "packets / poll"),
    ("ib_port_xmit_wait_delta", "IB Transmit Wait / Backpressure (contention signal)", "waiting events / poll"),
    ("step_time_ms",             "DDP Training Step Time",             "step time (ms)"),
]

# Label 2's primary story is contention → network waiting → training impact.
# Keep the drilldown focused on three charts; traffic and packet counters
# remain available in the comparison tab and statistics table.
LABEL2_PRIMARY_METRICS = [
    ("ib_port_xmit_wait_delta", "IB Transmit Wait / Backpressure (contention signal)", "waiting events / poll"),
    ("step_time_ms", "DDP Training Step Time (impact)", "step time (ms)"),
    ("gpu_util_pct", "GPU Utilization (context)", "GPU util (%)"),
]

LABEL1_PRIMARY_METRICS = [
    ("step_time_ms", "DDP Training Step Time (sync-pause impact)", "step time (ms)"),
    ("gpu_util_pct", "GPU Utilization (context)", "GPU util (%)"),
    ("ib_port_xmit_wait_delta", "IB Transmit Wait / Backpressure (context)", "waiting events / poll"),
]

NODE_COLORS = ["#2563eb", "#0f766e", "#c2410c", "#7c3aed", "#64748b"]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LENS · Monitor Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Layout ── */
.block-container { padding-top: 1rem; padding-bottom: 1.5rem; }

/* ── Header bar ── */
.lens-header {
    background: linear-gradient(120deg, #0f172a 0%, #1e293b 60%, #134e4a 100%);
    border-radius: 16px;
    padding: 1.1rem 1.4rem 1rem 1.4rem;
    margin-bottom: 1.2rem;
}
.lens-header h1 {
    color: #f1f5f9;
    margin: 0 0 0.2rem 0;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.lens-header p { color: #94a3b8; margin: 0; font-size: 0.92rem; }

/* ── Label badge row ── */
.label-badge {
    display: inline-block;
    padding: 0.22rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.4rem;
    color: white;
}

/* ── Stat cards ── */
.stat-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.stat-card .label  { font-size: 0.73rem; text-transform: uppercase;
                     letter-spacing: 0.07em; color: #64748b; }
.stat-card .value  { font-size: 1.7rem; font-weight: 700; color: #0f172a;
                     line-height: 1.1; }
.stat-card .sub    { font-size: 0.82rem; color: #64748b; margin-top: 0.15rem; }

/* ── Section headings ── */
.section-heading {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #64748b;
    margin: 0.8rem 0 0.3rem 0;
}

/* ── Stats table ── */
.stats-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.stats-table th {
    background: #f8fafc;
    padding: 0.5rem 0.75rem;
    text-align: left;
    font-weight: 600;
    color: #475569;
    border-bottom: 2px solid #e2e8f0;
}
.stats-table td { padding: 0.45rem 0.75rem; border-bottom: 1px solid #f1f5f9; }
.stats-table tr:last-child td { border-bottom: none; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _label_dirs(label_num: int) -> list[Path]:
    """Return supported data directories without requiring a fixed layout."""
    cfg = LABEL_CONFIG[label_num]
    candidates = [
        BASE_DIR / "data" / cfg["folder"],
        BASE_DIR / "data" / cfg["folder"].lower(),
        BASE_DIR / f"Label_{label_num}_Data",
    ]
    candidates.extend(sorted(BASE_DIR.glob(f"*Label_{label_num}_Data")))
    return list(dict.fromkeys(d for d in candidates if d.is_dir()))


def _normalise_frame(df: pd.DataFrame, source_name: str, label_num: int) -> pd.DataFrame:
    """Normalise uploaded/repository CSVs while preserving unknown columns."""
    cfg = LABEL_CONFIG[label_num]
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["_source_file"] = source_name

    aliases = {
        "gpu_utilization": "gpu_util_pct",
        "gpu_util": "gpu_util_pct",
        "ib_xmit_data_delta": "ib_port_xmit_data_delta",
        "ib_xmit_wait_delta": "ib_port_xmit_wait_delta",
        "ib_xmit_packets_delta": "ib_port_xmit_packets_delta",
        "step_time": "step_time_ms",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    numeric_cols = [
        "timestamp", "gpu_util_pct", "ib_port_xmit_data_delta",
        "ib_port_xmit_packets_delta", "ib_port_xmit_wait_delta",
        "ib_symbol_error_delta", "ib_sq_num_rnr_delta", "step_time_ms",
        "stress_active",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ignore marker/log CSVs accidentally copied into a telemetry folder.
    has_telemetry = "timestamp" in df.columns and any(
        c in df.columns for c in ("node_id", "gpu_util_pct", "ib_port_xmit_data_delta")
    )
    if not has_telemetry:
        return pd.DataFrame()

    if "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df["ts_rel"] = df.groupby("_source_file")["timestamp"].transform(
            lambda s: s - s.min()
        )

    if "node_id" in df.columns:
        df["node"] = df["node_id"].astype(str).str.replace(r"\..*", "", regex=True)
    else:
        extracted = df["_source_file"].str.extract(r"(?:telemetry_)?(g\d+)", expand=False)
        df["node"] = extracted.fillna("unknown")

    df["label_num"] = label_num
    df["label_name"] = cfg["name"]
    df["label_color"] = cfg["color"]
    return df


def _attach_stress_intervals(data: pd.DataFrame, search_dirs: list[Path]) -> pd.DataFrame:
    """Attach Label 2 ground-truth stress phases from stress_active CSVs."""
    if data.empty or "timestamp" not in data.columns:
        return data

    data = data.copy()
    if "stress_active" not in data.columns:
        data["stress_active"] = 0
    data["stress_active"] = data["stress_active"].fillna(0).astype(int)

    for directory in search_dirs:
        for marker_path in sorted(directory.glob("stress_active_*.csv")):
            try:
                markers = pd.read_csv(marker_path)
                markers["timestamp"] = pd.to_numeric(markers["timestamp"], errors="coerce")
                markers = markers.dropna(subset=["timestamp"])
                if markers.empty:
                    continue
                host = str(markers.get("hostname", pd.Series([marker_path.stem])).iloc[0])
                short_host = host.split(".")[0]
                starts = markers.loc[markers["stress_active"] == 1, "timestamp"]
                ends = markers.loc[markers["stress_active"] == 0, "timestamp"]
                if starts.empty or ends.empty:
                    continue
                start = float(starts.min())
                end = float(ends.max())
                mask = (
                    (data["node"].astype(str) == short_host)
                    & (data["timestamp"] >= start)
                    & (data["timestamp"] <= end)
                )
                data.loc[mask, "stress_active"] = 1
            except Exception:
                continue

    data["phase"] = data["stress_active"].map({1: "Stress", 0: "Non-stress"})
    return data


def _load_step_time(search_dirs: list[Path], label_num: int) -> list[pd.DataFrame]:
    """Load optional DDP step-time CSVs for any label.

    Step time is printed by the DDP workload logs, not by the NIC poller.
    The analyzer writes files such as ``step_time_91062.csv`` with columns
    ``timestamp,node,step,step_time_ms``.  The node is often ``rank0`` because
    rank 0 emits the timing line for the distributed job; it is therefore a
    global training signal rather than a per-worker NIC measurement.
    """
    frames: list[pd.DataFrame] = []
    for directory in search_dirs:
        paths = sorted(directory.glob("step_time_*.csv"))
        paths += sorted(directory.glob("ddp_step_time_*.csv"))
        for csv_path in paths:
            try:
                raw = pd.read_csv(csv_path, low_memory=False)
                raw.columns = [str(c).strip() for c in raw.columns]
                aliases = {
                    "step_time": "step_time_ms",
                    "step_time_msec": "step_time_ms",
                    "time": "timestamp",
                }
                for old, new in aliases.items():
                    if old in raw.columns and new not in raw.columns:
                        raw[new] = raw[old]
                if "timestamp" not in raw.columns or "step_time_ms" not in raw.columns:
                    continue

                frame = pd.DataFrame({
                    "timestamp": pd.to_numeric(raw["timestamp"], errors="coerce"),
                    "step_time_ms": pd.to_numeric(raw["step_time_ms"], errors="coerce"),
                })
                if "node" in raw.columns:
                    frame["node"] = raw["node"].astype(str).str.strip()
                elif "hostname" in raw.columns:
                    frame["node"] = raw["hostname"].astype(str).str.strip()
                else:
                    match = re.search(r"(g\d+)", csv_path.name)
                    frame["node"] = match.group(1) if match else "global"

                frame["node"] = frame["node"].replace({"nan": "rank0", "": "rank0"})

                frame = frame.dropna(subset=["timestamp", "step_time_ms"])
                if frame.empty:
                    continue
                frame["_source_file"] = csv_path.name
                frame["ts"] = pd.to_datetime(frame["timestamp"], unit="s", errors="coerce")
                frame["ts_rel"] = frame["timestamp"] - frame["timestamp"].min()
                frame["_source_type"] = "step_time"
                frame["label_num"] = label_num
                frame["label_name"] = LABEL_CONFIG[label_num]["name"]
                frame["label_color"] = LABEL_CONFIG[label_num]["color"]
                frames.append(frame)
            except Exception:
                continue
    return frames


# Backward-compatible name for any code that imported the previous helper.
def _load_label2_step_time(search_dirs: list[Path]) -> list[pd.DataFrame]:
    return _load_step_time(search_dirs, 2)


def _attach_label1_intervals(data: pd.DataFrame, search_dirs: list[Path]) -> pd.DataFrame:
    """Attach Label 1 synchronization-pause markers to telemetry and timing rows."""
    if data.empty or "timestamp" not in data.columns:
        return data

    data = data.copy()
    if "label1_active" not in data.columns:
        data["label1_active"] = 0
    data["label1_active"] = pd.to_numeric(data["label1_active"], errors="coerce").fillna(0).astype(int)

    for directory in search_dirs:
        for marker_path in sorted(directory.glob("label1_active_*.csv")):
            try:
                markers = pd.read_csv(marker_path)
                markers["timestamp"] = pd.to_numeric(markers["timestamp"], errors="coerce")
                markers["label1_active"] = pd.to_numeric(markers["label1_active"], errors="coerce")
                markers = markers.dropna(subset=["timestamp", "label1_active"]).sort_values("timestamp")
                if markers.empty:
                    continue

                host = str(markers.get("hostname", pd.Series([marker_path.stem])).iloc[0])
                short_host = host.split(".")[0]
                starts = markers.loc[markers["label1_active"] == 1, "timestamp"].tolist()
                ends = markers.loc[markers["label1_active"] == 0, "timestamp"].tolist()
                intervals = []
                for start in starts:
                    end = next((candidate for candidate in ends if candidate >= start), None)
                    if end is not None:
                        intervals.append((float(start), float(end)))

                telemetry_mask = data["node"].astype(str).eq(short_host)
                step_mask = data.get("_source_type", pd.Series("", index=data.index)).eq("step_time")
                for start, end in intervals:
                    active_mask = (
                        (data["timestamp"] >= start)
                        & (data["timestamp"] <= end)
                        & (telemetry_mask | step_mask)
                    )
                    data.loc[active_mask, "label1_active"] = 1
            except Exception:
                continue

    data["phase"] = data["label1_active"].map({1: "Sync pause", 0: "Normal"})
    return data


@st.cache_data(show_spinner=False)
def load_label_data(label_num: int) -> pd.DataFrame:
    """Load telemetry CSVs for a label and attach Label 2 stress phases."""
    search_dirs = _label_dirs(label_num)
    frames: list[pd.DataFrame] = []
    for directory in search_dirs:
        for csv_path in sorted(directory.glob("*.csv")):
            if csv_path.name.startswith("stress_active_"):
                continue
            try:
                frame = _normalise_frame(
                    pd.read_csv(csv_path, low_memory=False), csv_path.name, label_num
                )
                if not frame.empty:
                    frames.append(frame)
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)
    step_frames = _load_step_time(search_dirs, label_num)
    if step_frames:
        data = pd.concat([data, *step_frames], ignore_index=True)
    if label_num == 1:
        data = _attach_label1_intervals(data, search_dirs)
    if label_num == 2:
        data = _attach_stress_intervals(data, search_dirs)
    return data


def load_uploaded_label2(uploaded_files) -> pd.DataFrame:
    """Load CSVs supplied through the sidebar uploader as Label 2 telemetry."""
    frames = []
    for uploaded in uploaded_files or []:
        if uploaded.name.startswith("stress_active_"):
            continue
        try:
            frame = _normalise_frame(pd.read_csv(uploaded, low_memory=False), uploaded.name, 2)
            if not frame.empty:
                frames.append(frame)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def get_nodes(df: pd.DataFrame) -> list[str]:
    if df.empty or "node" not in df.columns:
        return []
    # Keep only real node names — starts with g followed by a digit
    valid = df["node"].dropna()
    valid = valid[valid.str.match(r'^g\d+$')]
    return sorted(valid.unique().tolist())


def filter_by_node(df: pd.DataFrame, node_sel: str) -> pd.DataFrame:
    if node_sel == "All nodes":
        return df
    return df[df["node"] == node_sel].copy()

# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

CHART_TEMPLATE = dict(
    template="plotly_white",
    margin=dict(l=8, r=8, t=40, b=8),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=280,
)


def _make_time_axis(df: pd.DataFrame) -> pd.Series:
    """Return best available time series for x-axis."""
    if "ts" in df.columns and df["ts"].notna().any():
        return df["ts"]
    return df.get("ts_rel", pd.Series(range(len(df)), index=df.index))


def _add_stress_regions(fig: go.Figure, df: pd.DataFrame) -> None:
    """Shade known Label 2 stress samples so the causal interval is obvious."""
    if "stress_active" not in df.columns or "timestamp" not in df.columns:
        return
    active = df[df["stress_active"] == 1]
    if active.empty:
        return
    start = active["timestamp"].min()
    end = active["timestamp"].max()
    x0 = pd.to_datetime(start, unit="s", errors="coerce")
    x1 = pd.to_datetime(end, unit="s", errors="coerce")
    if pd.notna(x0) and pd.notna(x1):
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="#ef4444",
            opacity=0.10,
            line_width=0,
            annotation_text="IB stress",
            annotation_position="top left",
        )


def _add_label1_regions(fig: go.Figure, df: pd.DataFrame) -> None:
    """Shade Label 1 synchronization-pause intervals."""
    if "label1_active" not in df.columns or "timestamp" not in df.columns:
        return
    active = df[df["label1_active"] == 1]
    if active.empty:
        return
    start = active["timestamp"].min()
    end = active["timestamp"].max()
    x0 = pd.to_datetime(start, unit="s", errors="coerce")
    x1 = pd.to_datetime(end, unit="s", errors="coerce")
    if pd.notna(x0) and pd.notna(x1):
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="#f59e0b",
            opacity=0.14,
            line_width=0,
            annotation_text="sync pause",
            annotation_position="top left",
        )


def build_single_label_charts(df: pd.DataFrame, node_sel: str, label_num: int) -> None:
    """Tab 1 — Three stacked charts for a single label / node selection."""
    cfg = LABEL_CONFIG[label_num]
    node_df = filter_by_node(df, node_sel)

    if node_df.empty:
        st.warning("No data for this selection.")
        return

    # Group by node for "All nodes" case — one trace per node
    x_label = "Time" if "ts" in node_df.columns and node_df["ts"].notna().any() else "Seconds since start"

    palette = NODE_COLORS
    if label_num == 2:
        metric_specs = LABEL2_PRIMARY_METRICS
    elif label_num == 1:
        metric_specs = LABEL1_PRIMARY_METRICS
    else:
        metric_specs = [spec for spec in METRICS if spec[0] != "step_time_ms"]

    for metric_col, metric_name, y_label in metric_specs:
        if metric_col not in node_df.columns or node_df[metric_col].notna().sum() == 0:
            if metric_col == "step_time_ms" and label_num in (1, 2):
                st.info(
                    "DDP Training Step Time is not available in the loaded telemetry. "
                    "Add a `step_time_*.csv` file with timestamp, node, and step_time_ms columns."
                )
            continue

        # DDP timing is normally emitted only by rank 0.  It is a global
        # training signal, so keep showing it when a worker node is selected
        # instead of incorrectly reporting an empty chart for g8/g11.
        chart_df = node_df
        global_step_time = False
        if metric_col == "step_time_ms":
            chart_df = node_df[node_df[metric_col].notna()].copy()
            if chart_df.empty:
                chart_df = df[df[metric_col].notna()].copy()
                global_step_time = not chart_df.empty

        if chart_df.empty:
            continue

        groups = (
            chart_df.groupby("node")
            if node_sel == "All nodes" or metric_col == "step_time_ms"
            else [(node_sel, chart_df)]
        )

        fig = go.Figure()

        for i, (node_name, gdf) in enumerate(groups):
            gdf = gdf.sort_values("timestamp") if "timestamp" in gdf.columns else gdf
            x = _make_time_axis(gdf)
            y = gdf[metric_col]
            color = palette[i % len(palette)]

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode="lines",
                name=node_name,
                line=dict(color=color, width=1.8),
                hovertemplate=f"%{{x}}<br>{metric_name}: %{{y:,.1f}}<extra>{node_name}</extra>",
            ))

        fig.update_layout(
            title=f"{metric_name}  —  {cfg['name']}",
            yaxis_title=y_label,
            xaxis_title=x_label,
            legend_title_text="Node",
            **CHART_TEMPLATE,
        )
        if label_num == 2:
            _add_stress_regions(fig, node_df)
        elif label_num == 1:
            _add_label1_regions(fig, node_df)
        st.plotly_chart(fig, use_container_width=True)

        if metric_col == "step_time_ms":
            st.caption(
                "DDP step time is emitted by rank 0 for the distributed job. "
                + ("It is shown globally for the selected worker node." if global_step_time else "")
            )

    if label_num == 2:
        st.caption(
            "Reading Label 2: the red shaded interval is the known IB stress period. "
            "Colored lines identify nodes; they do not represent different labels or severity levels. "
            "The primary evidence is higher network waiting together with slower DDP step time."
        )
    elif label_num == 1:
        st.caption(
            "Label 1 timing is a controlled synchronization-pause signal. "
            "Amber regions mark the recorded pause intervals; they are not InfiniBand fabric congestion."
        )


def build_comparison_charts(
    loaded_data: dict[int, pd.DataFrame],
    selected_labels: list[int],
    metric_col: str,
    metric_name: str,
    y_label: str,
) -> go.Figure:
    """Tab 2 — One line per label on a shared timeline (median across nodes)."""
    fig = go.Figure()

    for lnum in selected_labels:
        df = loaded_data.get(lnum)
        if df is None or df.empty:
            continue
        if metric_col not in df.columns:
            continue

        cfg = LABEL_CONFIG[lnum]

        # Use relative time (seconds from start) so labels are comparable even if
        # collected at different wall-clock times.
        if "ts_rel" not in df.columns:
            continue

        # Median per second bucket across all nodes — keeps chart clean
        df = df.copy()
        df["t_bucket"] = df["ts_rel"].round(1)
        agg = df.groupby("t_bucket")[metric_col].median().reset_index()

        fig.add_trace(go.Scatter(
            x=agg["t_bucket"],
            y=agg[metric_col],
            mode="lines",
            name=cfg["name"],
            line=dict(color=cfg["color"], width=2.2),
            hovertemplate=f"t=%{{x:.1f}}s<br>{metric_name}: %{{y:,.1f}}<extra>{cfg['name']}</extra>",
        ))

    if len(selected_labels) == 1:
        only_label = LABEL_CONFIG[selected_labels[0]]
        chart_title = f"{metric_name}  —  {only_label['name']}"
    else:
        chart_title = f"{metric_name}  —  Label comparison (median across nodes)"

    fig.update_layout(
        title=chart_title,
        yaxis_title=y_label,
        xaxis_title="Seconds since start",
        **CHART_TEMPLATE,
    )
    return fig


def build_stats_table(loaded_data: dict[int, pd.DataFrame], selected_labels: list[int]) -> str:
    """Tab 3 — Return an HTML stats table for selected labels."""
    rows = []
    for lnum in selected_labels:
        df = loaded_data.get(lnum)
        cfg = LABEL_CONFIG[lnum]
        if df is None or df.empty:
            rows.append({
                "label_num": lnum,
                "label_name": cfg["name"],
                "n_rows": 0,
                "avg_gpu_util": "—",
                "avg_xmit_data": "—",
                "avg_xmit_wait": "—",
                "peak_xmit_wait": "—",
                "avg_step_time": "—",
                "peak_step_time": "—",
                "stress_rows": "—",
            })
            continue

        def _fmt(col, fmt=".1f"):
            if col in df.columns:
                v = df[col].mean()
                return f"{v:{fmt}}" if pd.notna(v) else "—"
            return "—"

        rows.append({
            "label_num": lnum,
            "label_name": cfg["name"],
            "n_rows": len(df),
            "avg_gpu_util": _fmt("gpu_util_pct"),
            "avg_xmit_data": _fmt("ib_port_xmit_data_delta", ",.0f"),
            "avg_xmit_wait": _fmt("ib_port_xmit_wait_delta", ",.0f"),
            "peak_xmit_wait": (
                f"{df['ib_port_xmit_wait_delta'].max():,.0f}"
                if "ib_port_xmit_wait_delta" in df.columns
                and df["ib_port_xmit_wait_delta"].notna().any() else "—"
            ),
            "avg_step_time": _fmt("step_time_ms"),
            "peak_step_time": (
                f"{df['step_time_ms'].max():.1f}"
                if "step_time_ms" in df.columns
                and df["step_time_ms"].notna().any() else "—"
            ),
            "stress_rows": (
                f"{int((df['stress_active'] == 1).sum()):,}"
                if "stress_active" in df.columns else "—"
            ),
        })

    # Build HTML table
    header = """
    <table class="stats-table">
    <thead><tr>
        <th>Label</th>
        <th>Name</th>
        <th>Rows</th>
        <th>Avg GPU util (%)</th>
        <th>Avg xmit_data_delta</th>
        <th>Avg xmit_wait_delta</th>
        <th>Peak xmit_wait_delta</th>
        <th>Avg DDP step (ms)</th>
        <th>Peak DDP step (ms)</th>
        <th>Stress rows</th>
    </tr></thead><tbody>
    """
    body = ""
    for r in rows:
        cfg = LABEL_CONFIG[r["label_num"]]
        badge = f'<span class="label-badge" style="background:{cfg["color"]}">{r["label_num"]}</span>'
        body += f"""<tr>
            <td>{badge}</td>
            <td>{r["label_name"]}</td>
            <td>{r["n_rows"]:,}</td>
            <td>{r["avg_gpu_util"]}</td>
            <td>{r["avg_xmit_data"]}</td>
            <td>{r["avg_xmit_wait"]}</td>
            <td>{r["peak_xmit_wait"]}</td>
            <td>{r["avg_step_time"]}</td>
            <td>{r["peak_step_time"]}</td>
            <td>{r["stress_rows"]}</td>
        </tr>"""
    return header + body + "</tbody></table>"

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="lens-header">
        <h1>🔬 LENS · Monitor Dashboard</h1>
        <p>GPU &amp; InfiniBand telemetry explorer — SJSU HPC3 · CMPE 295A/B</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Label Selection")

        # Load data for all labels (cached)
        loaded_data: dict[int, pd.DataFrame] = {}
        available_labels: list[int] = []
        for lnum, cfg in LABEL_CONFIG.items():
            with st.spinner(f"Loading Label {lnum}…"):
                df = load_label_data(lnum)
            loaded_data[lnum] = df
            if not df.empty:
                available_labels.append(lnum)

        st.markdown("### Add Label 2 CSVs")
        uploaded_label2 = st.file_uploader(
            "Upload Label 2 telemetry CSVs",
            type=["csv"],
            accept_multiple_files=True,
            help=(
                "Upload telemetry_*.csv files. Optional stress_active_*.csv "
                "files can be kept in the same data/Label2 folder for phase shading."
            ),
        )
        uploaded_df = load_uploaded_label2(uploaded_label2)
        if not uploaded_df.empty:
            loaded_data[2] = pd.concat([loaded_data.get(2, pd.DataFrame()), uploaded_df], ignore_index=True)
            if 2 not in available_labels:
                available_labels.append(2)
            st.success(f"Loaded {len(uploaded_df):,} uploaded Label 2 rows.")

        if not available_labels:
            st.error("No data found. Add CSVs under `data/Label0/`, `data/Label2/`, etc., or upload Label 2 telemetry above.")
            st.stop()

        # ── Tab 1 controls ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Tab 1 · Single-label drilldown")

        label_options = {
            f"Label {n} — {LABEL_CONFIG[n]['name']}": n
            for n in available_labels
        }
        selected_label_key = st.selectbox("Select label", list(label_options.keys()))
        selected_label = label_options[selected_label_key]
        cfg_sel = LABEL_CONFIG[selected_label]

        df_sel = loaded_data[selected_label]
        node_list = ["All nodes"] + get_nodes(df_sel)
        selected_node = st.selectbox("Select node", node_list)

        # ── Tab 2 & 3 controls ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Tabs 2 & 3 · Comparison")
        compare_labels = st.multiselect(
            "Labels to compare",
            options=available_labels,
            default=available_labels,
            format_func=lambda n: f"Label {n} — {LABEL_CONFIG[n]['name']}",
        )

        # ── Legend ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Label legend")
        for lnum in available_labels:
            c = LABEL_CONFIG[lnum]
            status = "✅" if not loaded_data[lnum].empty else "❌"
            nrows = len(loaded_data[lnum]) if not loaded_data[lnum].empty else 0
            st.markdown(
                f'<span class="label-badge" style="background:{c["color"]}">{lnum}</span> '
                f'**{c["name"]}** {status}<br>'
                f'<span style="font-size:0.8rem;color:#64748b;margin-left:2.2rem;">'
                f'{c["description"]}<br>{nrows:,} rows</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

    # ── KPI row ─────────────────────────────────────────────────────────────
    kpi_cols = st.columns(len(available_labels))
    for col, lnum in zip(kpi_cols, available_labels):
        df = loaded_data[lnum]
        cfg = LABEL_CONFIG[lnum]
        avg_gpu = f"{df['gpu_util_pct'].mean():.1f}%" if not df.empty and "gpu_util_pct" in df.columns else "—"
        avg_wait = (
            f"{df['ib_port_xmit_wait_delta'].mean():,.0f}"
            if not df.empty and "ib_port_xmit_wait_delta" in df.columns else "—"
        )
        n_nodes = df["node"].nunique() if not df.empty else 0
        with col:
            if lnum == 2:
                main_value = avg_wait
                main_label = "avg IB transmit wait"
                subline = f"GPU util {avg_gpu} · {n_nodes} node(s)"
            else:
                main_value = avg_gpu
                main_label = "avg GPU utilization"
                subline = f"IB data rows · {n_nodes} node(s)"
            st.markdown(f"""
            <div class="stat-card">
                <div class="label">
                    <span class="label-badge" style="background:{cfg['color']};font-size:0.7rem">{lnum}</span>
                    {cfg['name']}
                </div>
                <div class="value">{main_value}</div>
                <div class="sub">{main_label} · {subline}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📈 Single-label drilldown",
        "🔀 Label comparison",
        "📊 Statistics table",
    ])

    # ====================================================================
    # TAB 1 — Single label, node selector, 3 charts on shared timeline
    # ====================================================================
    with tab1:
        badge_html = (
            f'<span class="label-badge" style="background:{cfg_sel["color"]}">'
            f'Label {selected_label}</span>'
        )
        st.markdown(
            f"{badge_html} &nbsp;**{cfg_sel['name']}** — "
            f"*{cfg_sel['description']}*"
            f"&ensp;|&ensp; Node: **{selected_node}**",
            unsafe_allow_html=True,
        )

        if selected_label == 2:
            stress_rows = int((df_sel.get("stress_active", pd.Series(dtype=int)) == 1).sum())
            st.markdown("### Label 2: what the charts mean")
            st.info(
                "Label 2 means InfiniBand contention affecting distributed training. "
                "The three charts below show the contention signal, its training impact, "
                "and GPU context. The red shaded interval is when additional RDMA traffic was active."
            )
            if stress_rows:
                st.info(
                    f"Label 2 ground truth: {stress_rows:,} telemetry rows fall inside a known "
                    "IB stress interval. Focus on network waiting and DDP step time; "
                    "GPU utilization may remain high."
                )
            else:
                st.warning(
                    "No stress_active markers were found for this Label 2 data. "
                    "Charts still render, but the stress interval cannot be shaded."
                )
        elif selected_label == 1:
            st.info(
                "Label 1 still uses InfiniBand: DDP AllReduce produces normal traffic and packets. "
                "The Label 1 signature is periodic DDP step-time spikes without sustained "
                "IB transmit-wait/backpressure like Label 2."
            )

        build_single_label_charts(df_sel, selected_node, selected_label)

    # ====================================================================
    # TAB 2 — Side-by-side label comparison, one line per label
    # ====================================================================
    with tab2:
        if not compare_labels:
            st.info("Select at least one label in the sidebar to compare.")
        else:
            st.markdown(
                '<p class="section-heading">Median per second — one line per label</p>',
                unsafe_allow_html=True,
            )
            # Check that at least some data has ts_rel
            has_rel = any(
                not loaded_data[lnum].empty and "ts_rel" in loaded_data[lnum].columns
                for lnum in compare_labels
            )
            if not has_rel:
                st.warning("Relative timestamps could not be computed — check that `timestamp` column exists.")
            else:
                # Keep DDP timing out of the mixed telemetry stack.  It is a
                # training-level signal with a different source and duration
                # than the NIC poller, so it is clearer in separate charts.
                for metric_col, metric_name, y_label in METRICS:
                    if metric_col == "step_time_ms":
                        continue
                    has_metric = any(
                        not loaded_data[lnum].empty
                        and metric_col in loaded_data[lnum].columns
                        and loaded_data[lnum][metric_col].notna().any()
                        for lnum in compare_labels
                    )
                    if not has_metric:
                        continue
                    fig = build_comparison_charts(
                        loaded_data, compare_labels, metric_col, metric_name, y_label
                    )
                    st.plotly_chart(fig, use_container_width=True)

                step_labels = [
                    lnum
                    for lnum in compare_labels
                    if not loaded_data[lnum].empty
                    and "step_time_ms" in loaded_data[lnum].columns
                    and loaded_data[lnum]["step_time_ms"].notna().any()
                ]

                st.markdown("### DDP Training Step Time — shown separately")
                if not step_labels:
                    st.info(
                        "No step_time_*.csv file was found for the selected labels. "
                        "Place the generated DDP step-time CSV inside the matching label folder."
                    )
                else:
                    for lnum in step_labels:
                        cfg = LABEL_CONFIG[lnum]
                        fig = build_comparison_charts(
                            loaded_data,
                            [lnum],
                            "step_time_ms",
                            "DDP Training Step Time",
                            "step time (ms)",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        if lnum == 1:
                            st.caption(
                                "Label 1 only: periodic spikes show the controlled synchronization pause. "
                                "This chart is not overlaid with Label 2."
                            )
                        elif lnum == 2:
                            st.caption(
                                "Label 2 only: step-time elevation is shown separately from Label 1."
                            )

                st.markdown("""
                **Reading this chart:**
                - **Healthy (green):** GPU and communication activity are stable for the workload.
                - **AllReduce Pause (amber):** DDP still sends AllReduce traffic, but has brief synchronization-related step-time spikes without sustained backpressure.
                - **IB Network Contention (red):** `xmit_wait_delta` rises because the NIC waits for network credits; DDP step time may increase. GPU utilization can remain high.
                - **Idle (gray):** GPU utilization and IB traffic are near zero.
                """)

    # ====================================================================
    # TAB 3 — Label statistics table
    # ====================================================================
    with tab3:
        if not compare_labels:
            st.info("Select at least one label in the sidebar.")
        else:
            st.markdown(
                '<p class="section-heading">Per-label aggregate statistics</p>',
                unsafe_allow_html=True,
            )
            st.markdown(build_stats_table(loaded_data, compare_labels), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Also offer a download of the combined data
            if st.button("⬇ Export combined CSV"):
                parts = [
                    loaded_data[lnum]
                    for lnum in compare_labels
                    if not loaded_data[lnum].empty
                ]
                if parts:
                    combined = pd.concat(parts, ignore_index=True)
                    csv_bytes = combined.to_csv(index=False).encode()
                    st.download_button(
                        "Download combined_telemetry.csv",
                        data=csv_bytes,
                        file_name="combined_telemetry.csv",
                        mime="text/csv",
                    )


if __name__ == "__main__":
    main()
