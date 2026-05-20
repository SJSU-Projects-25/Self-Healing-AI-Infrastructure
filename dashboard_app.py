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
        Label2/   telemetry_<node>_<jobid>_label2.csv   (network_stall)
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
        "name": "Network Stall",
        "short": "network_stall",
        "color": "#ef4444",
        "description": "Sustained GPU util 0%, IB backpressure elevated",
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
    ("gpu_util_pct",            "GPU Utilization",        "GPU util (%)"),
    ("ib_port_xmit_data_delta", "Data Sent Rate",         "xmit_data_delta"),
    ("ib_port_xmit_wait_delta", "Congestion Backpressure","xmit_wait_delta"),
]

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

@st.cache_data(show_spinner=False)
def load_label_data(label_num: int) -> pd.DataFrame:
    """Load all CSVs for a given label number. Returns empty DataFrame if folder missing."""
    cfg = LABEL_CONFIG[label_num]
    folder = BASE_DIR / "data" / cfg["folder"]

    # Also check legacy layout at repo root (e.g. 34662-Label_0_Data)
    legacy_candidates = sorted(BASE_DIR.glob(f"*Label_{label_num}_Data"))

    if not folder.exists() and not legacy_candidates:
        return pd.DataFrame()

    search_dirs = [folder] if folder.exists() else legacy_candidates

    frames: list[pd.DataFrame] = []
    for d in search_dirs:
        for csv_path in sorted(d.glob("*.csv")):
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                df["_source_file"] = csv_path.name
                frames.append(df)
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    # Normalise numeric columns
    numeric_cols = [
        "timestamp", "gpu_util_pct",
        "ib_port_xmit_data_delta", "ib_port_xmit_wait_delta",
        "ib_symbol_error_delta", "ib_sq_num_rnr_delta",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Derive timestamps
    if "timestamp" in data.columns:
        data["ts"] = pd.to_datetime(data["timestamp"], unit="s", errors="coerce")
        # Fallback: treat as seconds-since-start per file so charts always have a valid x-axis
        data["ts_rel"] = data.groupby("_source_file")["timestamp"].transform(
            lambda s: s - s.min()
        )

    # Node short name
    if "node_id" in data.columns:
        data["node"] = (
            data["node_id"]
            .astype(str)
            .str.replace(r"\..*", "", regex=True)   # strip FQDN
        )
    else:
        data["node"] = data["_source_file"].str.extract(r"telemetry_(\w+)_")[0].fillna("unknown")

    data["label_num"] = label_num
    data["label_name"] = cfg["name"]
    data["label_color"] = cfg["color"]

    return data

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


def build_single_label_charts(df: pd.DataFrame, node_sel: str, label_num: int) -> None:
    """Tab 1 — Three stacked charts for a single label / node selection."""
    cfg = LABEL_CONFIG[label_num]
    node_df = filter_by_node(df, node_sel)

    if node_df.empty:
        st.warning("No data for this selection.")
        return

    # Group by node for "All nodes" case — one trace per node
    groups = (
        node_df.groupby("node")
        if node_sel == "All nodes"
        else [(node_sel, node_df)]
    )

    x_label = "Time" if "ts" in node_df.columns and node_df["ts"].notna().any() else "Seconds since start"

    palette = ["#4f8ef7", "#f59e0b", "#22c55e", "#8b5cf6", "#ef4444"]

    for metric_col, metric_name, y_label in METRICS:
        if metric_col not in node_df.columns:
            st.caption(f"⚠ Column `{metric_col}` not found in data — skipping.")
            continue

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
            **CHART_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)


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

    fig.update_layout(
        title=f"{metric_name}  —  Label comparison (median across nodes)",
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

        if not available_labels:
            st.error("No data found. Expected `data/Label0/`, `data/Label1/`, etc. relative to this script.")
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
        n_nodes = df["node"].nunique() if not df.empty else 0
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="label">
                    <span class="label-badge" style="background:{cfg['color']};font-size:0.7rem">{lnum}</span>
                    {cfg['name']}
                </div>
                <div class="value">{avg_gpu}</div>
                <div class="sub">avg GPU util · {n_nodes} node(s)</div>
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
                for metric_col, metric_name, y_label in METRICS:
                    fig = build_comparison_charts(
                        loaded_data, compare_labels, metric_col, metric_name, y_label
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                **Reading this chart:**
                - **Healthy (green):** GPU util stays near 100%, xmit_data is high and sustained.
                - **AllReduce Pause (amber):** Brief GPU dip at collective sync points, xmit_data drops momentarily.
                - **Network Stall (red):** GPU util collapses to 0%, xmit_wait spikes — traffic is queued.
                - **Idle (gray):** GPU util 0%, xmit_data near zero — no training job running.
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
