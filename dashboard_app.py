from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
DATASET_SOURCES = {
    "34662 / label 0": BASE_DIR / "34662-Label_0_Data",
    "40875 / label 1": BASE_DIR / "40875-Label_1_Data",
}

SUMMARY_METRICS = [
    ("gpu_util_pct", "GPU utilization", "%"),
    ("gpu_mem_util_pct", "GPU memory", "%"),
    ("network_total_mb_s", "Network throughput", "MB/s"),
    ("wait_pressure", "IB pressure", "index"),
]

TREND_METRICS = [
    "gpu_util_pct",
    "gpu_mem_util_pct",
    "network_total_mb_s",
    "ib_port_xmit_wait_delta",
    "ib_port_rcv_errors_delta",
]

st.set_page_config(
    page_title="LENS POC Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 1.5rem; }
      .hero {
        padding: 1rem 1.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        color: white;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.20);
        margin-bottom: 1rem;
      }
      .hero h1 { margin: 0; font-size: 2rem; }
      .hero p { margin: 0.35rem 0 0 0; color: rgba(255,255,255,0.88); }
      .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 0.95rem 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        min-height: 110px;
      }
      .metric-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.25rem;
      }
      .metric-value { font-size: 1.9rem; font-weight: 700; line-height: 1.05; color: #0f172a; }
      .metric-sub { margin-top: 0.25rem; color: #475569; font-size: 0.86rem; }
      .metric-chip {
        display: inline-block;
        margin-top: 0.45rem;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: #ecfeff;
        color: #0f766e;
        font-size: 0.74rem;
        font-weight: 600;
      }
      .panel {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 1rem 1rem 0.5rem 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
      }
      .panel h3 { margin-top: 0; margin-bottom: 0.2rem; }
      .panel .subtle { color: #64748b; font-size: 0.9rem; margin-bottom: 0.6rem; }
      .small-note { color: #64748b; font-size: 0.85rem; }
      .insight-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 0.85rem 0.95rem;
        margin-top: 0.4rem;
      }
      .insight-box ul { margin: 0.45rem 0 0 1.15rem; }
      .insight-box li { margin-bottom: 0.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_interleaved_labeled_csv(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next(csv.reader([lines[0]]))
    rows: list[dict[str, object]] = []

    for index in range(2, len(lines), 2):
        telemetry_line = lines[index].strip() if index < len(lines) else ""
        if not telemetry_line:
            continue
        telemetry = next(csv.reader([telemetry_line]))
        label = None
        if index + 1 < len(lines):
            label_line = lines[index + 1].strip()
            if label_line.startswith(","):
                label_parts = next(csv.reader([label_line]))
                if len(label_parts) > 1 and label_parts[1].strip():
                    label = label_parts[1].strip()
        row = {column: telemetry[i] if i < len(telemetry) else "" for i, column in enumerate(header)}
        row["label"] = label
        rows.append(row)

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for cohort, folder in DATASET_SOURCES.items():
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv")):
            if path.name.endswith("_labeled.csv"):
                frame = _read_interleaved_labeled_csv(path)
                source_kind = "labeled"
            else:
                frame = pd.read_csv(path)
                frame["label"] = pd.NA
                source_kind = "raw"

            frame["cohort"] = cohort
            frame["source_file"] = path.name
            frame["source_kind"] = source_kind
            frame["node_label"] = frame.get("node_id", path.stem)
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    for column in data.columns:
        if column in {"node_id", "label", "cohort", "source_file", "source_kind", "node_label"}:
            continue
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["timestamp_dt"] = pd.to_datetime(data["timestamp"], unit="s", errors="coerce")
    data["timestamp_label"] = data["timestamp_dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    data["gpu_mem_util_pct"] = np.where(
        data["gpu_mem_total_mb"] > 0,
        100.0 * data["gpu_mem_used_mb"] / data["gpu_mem_total_mb"],
        np.nan,
    )

    data["node_short"] = data["node_id"].fillna("").str.replace(".hpc.coe", "", regex=False)
    data["node_key"] = data["node_short"] + " | " + data["cohort"].astype(str)
    data["node_display"] = data["node_short"] + " | " + data["cohort"].astype(str)
    data["label_clean"] = data["label"].fillna("unlabeled")
    data["job_label"] = data["job_id"].fillna(-1).astype("Int64").astype(str)

    data["dt_s"] = data.groupby(["source_file"])["timestamp"].diff()
    data["dt_s"] = data["dt_s"].fillna(data["polling_interval_actual"])
    data["dt_s"] = data["dt_s"].fillna(0.5).clip(lower=0.001)

    data["network_xmit_mb_s"] = (
        data["ib_port_xmit_data_delta"].fillna(0) / data["dt_s"] / (1024**2)
    )
    data["network_rcv_mb_s"] = (
        data["ib_port_rcv_data_delta"].fillna(0) / data["dt_s"] / (1024**2)
    )
    data["network_total_mb_s"] = data["network_xmit_mb_s"] + data["network_rcv_mb_s"]
    data["wait_pressure"] = (
        data["ib_port_xmit_wait_delta"].fillna(0)
        + data["ib_sq_num_to_delta"].fillna(0)
        + data["ib_port_rcv_errors_delta"].fillna(0)
        + data["ib_symbol_error_delta"].fillna(0)
    )

    def _robust_z(series: pd.Series) -> pd.Series:
        median = series.median(skipna=True)
        mad = (series - median).abs().median(skipna=True)
        if pd.isna(mad) or mad == 0:
            scale = series.std(ddof=0)
            if pd.isna(scale) or scale == 0:
                scale = 1.0
        else:
            scale = 1.4826 * mad
        return (series - median) / scale

    for column in ["gpu_util_pct", "gpu_mem_util_pct", "network_total_mb_s", "wait_pressure"]:
        data[f"{column}_rz"] = data.groupby("source_file")[column].transform(_robust_z)

    data["gpu_util_roll"] = data.groupby("source_file")["gpu_util_pct"].transform(
        lambda s: s.rolling(window=12, min_periods=3).mean()
    )
    data["network_roll"] = data.groupby("source_file")["network_total_mb_s"].transform(
        lambda s: s.rolling(window=12, min_periods=3).mean()
    )
    data["gpu_delta_from_roll"] = data["gpu_util_pct"] - data["gpu_util_roll"]
    data["network_delta_from_roll"] = data["network_total_mb_s"] - data["network_roll"]

    events: list[pd.DataFrame] = []
    for _, group in data.groupby("source_file", sort=False):
        group = group.copy()
        event_mask = (
            (group["ib_port_rcv_errors_delta"].fillna(0) > 0)
            | (group["ib_sq_num_to_delta"].fillna(0) > 0)
            | (group["wait_pressure"].fillna(0) >= 6000)
            | ((group["gpu_util_pct"] < 30) & (group["gpu_util_roll"] > 80))
            | (group["network_delta_from_roll"].fillna(0) > 30)
            | (group["network_delta_from_roll"].fillna(0) < -30)
        )
        event_rows = group.loc[event_mask].copy()
        if not event_rows.empty:
            event_rows["event_type"] = event_rows.apply(_classify_event, axis=1)
            event_rows["severity"] = event_rows.apply(_severity_for_event, axis=1)
            events.append(event_rows)

    if events:
        event_frame = pd.concat(events, ignore_index=True)
    else:
        event_frame = pd.DataFrame(columns=list(data.columns) + ["event_type", "severity"])

    data["event_flag"] = False
    if not event_frame.empty:
        event_keys = set(zip(event_frame["source_file"], event_frame["timestamp"]))
        data["event_flag"] = [
            (source_file, timestamp) in event_keys for source_file, timestamp in zip(data["source_file"], data["timestamp"])
        ]

    data.attrs["events"] = event_frame
    return data


def _classify_event(row: pd.Series) -> str:
    if pd.notna(row.get("ib_port_rcv_errors_delta")) and float(row.get("ib_port_rcv_errors_delta", 0) or 0) > 0:
        return "Transport error"
    if pd.notna(row.get("ib_sq_num_to_delta")) and float(row.get("ib_sq_num_to_delta", 0) or 0) > 0:
        return "Queue timeout"
    if float(row.get("wait_pressure_rz", 0) or 0) > 2.25:
        return "Network backpressure"
    if float(row.get("gpu_util_pct_rz", 0) or 0) < -2.0 and float(row.get("network_total_mb_s_rz", 0) or 0) > 1.25:
        return "GPU stall under load"
    if float(row.get("gpu_util_pct_rz", 0) or 0) > 2.5:
        return "GPU burst"
    if float(row.get("gpu_mem_util_pct", 0) or 0) > 85:
        return "High GPU memory"
    return "Signal spike"


def _severity_for_event(row: pd.Series) -> str:
    if _classify_event(row) in {"Transport error", "Queue timeout"}:
        return "High"
    if _classify_event(row) in {"Network backpressure", "GPU stall under load"}:
        return "Medium"
    return "Low"


def _format_value(value: float | int | None, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    if suffix == "%":
        return f"{value:,.1f}%"
    if suffix == "MB/s":
        return f"{value:,.2f} MB/s"
    if suffix == "index":
        return f"{value:,.0f}"
    return f"{value:,.2f}{suffix}"


def _metric_card(title: str, value: str, subtext: str, chip: str) -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-title">{title}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub">{subtext}</div>
      <div class="metric-chip">{chip}</div>
    </div>
    """


def _window_to_minutes(window: str) -> float | None:
    mapping = {"1h": 60.0, "6h": 360.0, "24h": 1440.0, "All": None}
    return mapping[window]


def _time_filtered_frame(frame: pd.DataFrame, window: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    minutes = _window_to_minutes(window)
    if minutes is None:
        return frame.copy()
    end_time = frame["timestamp_dt"].max()
    start_time = end_time - pd.Timedelta(minutes=minutes)
    return frame.loc[frame["timestamp_dt"] >= start_time].copy()


def _build_utilization_figure(frame: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp_dt"],
            y=frame["gpu_util_pct"],
            mode="lines",
            name="GPU utilization",
            line=dict(color="#4f8ef7", width=2.2),
            hovertemplate="%{x|%H:%M:%S}<br>GPU util: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp_dt"],
            y=frame["gpu_util_roll"],
            mode="lines",
            name="Rolling mean",
            line=dict(color="#0f766e", width=2, dash="dot"),
            hovertemplate="%{x|%H:%M:%S}<br>Rolling mean: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    incidents = frame.loc[frame["event_flag"]]
    if not incidents.empty:
        fig.add_trace(
            go.Scatter(
                x=incidents["timestamp_dt"],
                y=incidents["gpu_util_pct"],
                mode="markers",
                name="Incident",
                marker=dict(size=9, color="#ef4444", line=dict(color="white", width=1)),
                hovertemplate="%{x|%H:%M:%S}<br>Incident<extra></extra>",
            ),
            secondary_y=False,
        )
    fig.update_layout(
        title=title,
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="GPU utilization (%)", secondary_y=False, range=[0, max(100, float(frame["gpu_util_pct"].max() or 0) + 5)])
    fig.update_yaxes(title_text="", secondary_y=True, visible=False)
    fig.update_xaxes(title_text="Time")
    return fig


def _build_network_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp_dt"],
            y=frame["network_total_mb_s"],
            mode="lines",
            name="Network throughput",
            line=dict(color="#14b8a6", width=2.1),
            hovertemplate="%{x|%H:%M:%S}<br>Network: %{y:.2f} MB/s<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp_dt"],
            y=frame["network_xmit_mb_s"],
            mode="lines",
            name="XMIT",
            line=dict(color="#8b5cf6", width=1.4, dash="dot"),
            hovertemplate="%{x|%H:%M:%S}<br>XMIT: %{y:.2f} MB/s<extra></extra>",
        )
    )
    incidents = frame.loc[frame["event_flag"]]
    if not incidents.empty:
        fig.add_trace(
            go.Scatter(
                x=incidents["timestamp_dt"],
                y=incidents["network_total_mb_s"],
                mode="markers",
                name="Incident",
                marker=dict(size=8, color="#ef4444"),
                hovertemplate="%{x|%H:%M:%S}<br>Incident<extra></extra>",
            )
        )
    fig.update_layout(
        title="Network throughput",
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="MB/s")
    fig.update_xaxes(title_text="Time")
    return fig


def _build_incident_timeline(events: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if events.empty:
        fig.add_annotation(
            text="No incidents detected in the selected window.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#64748b", size=14),
        )
    else:
        event_types = list(dict.fromkeys(events["event_type"].tolist()))
        color_map = {
            "Transport error": "#ef4444",
            "Queue timeout": "#f97316",
            "Network backpressure": "#eab308",
            "GPU stall under load": "#22c55e",
            "GPU burst": "#3b82f6",
            "High GPU memory": "#8b5cf6",
            "Signal spike": "#64748b",
        }
        for event_type in event_types:
            subset = events.loc[events["event_type"] == event_type]
            fig.add_trace(
                go.Scatter(
                    x=subset["timestamp_dt"],
                    y=[event_type] * len(subset),
                    mode="markers",
                    name=event_type,
                    marker=dict(size=10, color=color_map.get(event_type, "#0f766e")),
                    customdata=np.stack([subset["source_file"], subset["node_short"]], axis=-1),
                    hovertemplate="%{x|%H:%M:%S}<br>%{customdata[1]}<br>%{y}<extra></extra>",
                )
            )
    fig.update_layout(
        title="Incident markers",
        height=220,
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="")
    return fig


def _build_correlation_heatmap(frame: pd.DataFrame) -> go.Figure:
    columns = [
        "gpu_util_pct",
        "gpu_mem_util_pct",
        "network_total_mb_s",
        "ib_port_xmit_wait_delta",
        "ib_port_rcv_errors_delta",
        "ib_sq_num_to_delta",
        "wait_pressure",
    ]
    corr = frame[columns].corr().fillna(0)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="Tealrose",
            zmin=-1,
            zmax=1,
            hovertemplate="%{y} ↔ %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(title="Feature correlation", height=420, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def _top_insights(frame: pd.DataFrame, events: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if frame.empty:
        return ["No data loaded from the selected folders."]

    mean_gpu = frame.groupby("source_file")["gpu_util_pct"].mean().sort_values(ascending=False)
    if not mean_gpu.empty:
        top_gpu_file = mean_gpu.index[0]
        notes.append(f"Highest average GPU utilization: {top_gpu_file} ({mean_gpu.iloc[0]:.1f}%).")

    mean_network = frame.groupby("source_file")["network_total_mb_s"].mean().sort_values(ascending=False)
    if not mean_network.empty:
        top_net_file = mean_network.index[0]
        notes.append(f"Largest average network activity: {top_net_file} ({mean_network.iloc[0]:.2f} MB/s).")

    if "label_clean" in frame.columns:
        label_counts = frame.loc[frame["label_clean"].notna() & (frame["label_clean"] != "unlabeled"), "label_clean"].value_counts()
        if not label_counts.empty:
            notes.append(
                f"Labeled rows are currently dominated by '{label_counts.index[0]}' ({label_counts.iloc[0]} rows)."
            )

    if events.empty:
        notes.append("No rule-based incidents were detected in the current window.")
    else:
        notes.append(f"Detected {len(events)} signal-based incident candidates across the filtered data.")

    return notes[:4]


def _render_metric_row(frame: pd.DataFrame, events: pd.DataFrame) -> None:
    total_samples = len(frame)
    total_nodes = frame["node_short"].nunique()
    avg_gpu = frame["gpu_util_pct"].mean()
    avg_mem = frame["gpu_mem_util_pct"].mean()
    avg_network = frame["network_total_mb_s"].mean()
    incidents = len(events)

    cards = st.columns(4)
    card_specs = [
        ("Telemetry records", f"{total_samples:,}", f"{total_nodes} nodes in view", "Telemetry loaded"),
        ("GPU utilization", _format_value(avg_gpu, "%"), f"Median {frame['gpu_util_pct'].median():.1f}%", "Raw telemetry"),
        ("GPU memory", _format_value(avg_mem, "%"), f"Avg used {frame['gpu_mem_used_mb'].mean():.0f} MB", "Memory pressure"),
        ("Network throughput", _format_value(avg_network, "MB/s"), f"{incidents} incident candidates", "IB traffic"),
    ]
    for col, (title, value, subtext, chip) in zip(cards, card_specs):
        with col:
            st.markdown(_metric_card(title, value, subtext, chip), unsafe_allow_html=True)


def _render_label_summary(frame: pd.DataFrame) -> None:
    labeled = frame.loc[frame["label_clean"].notna() & (frame["label_clean"] != "unlabeled")]
    if labeled.empty:
        st.info("No label rows were found in the selected window. The dashboard still uses telemetry-based anomaly markers.")
        return

    counts = labeled["label_clean"].value_counts().reset_index()
    counts.columns = ["label", "count"]
    fig = go.Figure(
        data=go.Bar(
            x=counts["label"],
            y=counts["count"],
            marker_color="#0f766e",
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    fig.update_layout(title="Label distribution", height=260, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _render_node_table(frame: pd.DataFrame) -> None:
    summary = (
        frame.groupby(["cohort", "node_short", "source_kind"])
        .agg(
            samples=("timestamp", "count"),
            avg_gpu=("gpu_util_pct", "mean"),
            avg_mem=("gpu_mem_util_pct", "mean"),
            avg_net=("network_total_mb_s", "mean"),
            incidents=("event_flag", "sum"),
        )
        .reset_index()
        .sort_values(["incidents", "avg_net"], ascending=[False, False])
    )
    summary["avg_gpu"] = summary["avg_gpu"].round(1)
    summary["avg_mem"] = summary["avg_mem"].round(1)
    summary["avg_net"] = summary["avg_net"].round(2)
    st.dataframe(summary, use_container_width=True, hide_index=True)


def _render_event_log(events: pd.DataFrame) -> None:
    if events.empty:
        st.success("No incident candidates detected in the selected window.")
        return

    log = events.sort_values("timestamp_dt", ascending=False).head(10).copy()
    log["recommended_action"] = log["event_type"].map(
        {
            "Transport error": "Check IB fabric health and node logs.",
            "Queue timeout": "Inspect network congestion and HCA counters.",
            "Network backpressure": "Review traffic imbalance and link pressure.",
            "GPU stall under load": "Check workload sync and GPU scheduling.",
            "GPU burst": "Verify expected ramp-up or job phase change.",
            "High GPU memory": "Confirm batch sizing and memory headroom.",
            "Signal spike": "Inspect nearby telemetry for the source of change.",
        }
    )
    log["recommended_action"] = log["recommended_action"].fillna("Review the nearby telemetry window.")

    display = log[["timestamp_label", "node_short", "event_type", "severity", "recommended_action"]].rename(
        columns={
            "timestamp_label": "time",
            "node_short": "node",
            "event_type": "event",
            "severity": "severity",
            "recommended_action": "admin note",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_design_note() -> None:
    st.markdown(
        """
        <div class="insight-box">
                    <strong>Project focus:</strong> Monitoring GPU and network telemetry to identify faults today and support automated remediation in future work.
          <ul>
                        <li>Designed for quick administrative review of summarized health signals and incident trends.</li>
                        <li>Supports drill-down inspection of node-level telemetry, anomalies, and dataset provenance.</li>
                        <li>Built with a clean Streamlit + Plotly interface that can evolve into a broader monitoring workflow.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Training Run Monitor</h1>
          <p>Monitoring GPU and InfiniBand telemetry to detect faults, compare labeled runs, and support future automated recovery.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("The 40875 labeled cohort currently contains healthy examples, while the 34662 cohort provides baseline telemetry. The dashboard highlights likely incident windows from telemetry changes rather than assuming a node is faulty.")

    frame = load_dashboard_data()
    if frame.empty:
        st.error("No telemetry files were found in the expected dataset folders.")
        return

    events = frame.attrs.get("events", pd.DataFrame())
    if not isinstance(events, pd.DataFrame):
        events = pd.DataFrame(events)

    with st.sidebar:
        st.header("Filters")
        cohort_choices = sorted(frame["cohort"].dropna().unique().tolist())
        selected_cohorts = st.multiselect("Dataset cohort", cohort_choices, default=cohort_choices)
        filtered_for_cohort = frame.loc[frame["cohort"].isin(selected_cohorts)].copy()

        node_options = (
            filtered_for_cohort[["node_key", "node_display", "cohort", "node_short"]]
            .drop_duplicates()
            .sort_values(["node_short", "cohort"])
        )
        node_choices = node_options["node_key"].tolist()
        node_label_map = dict(zip(node_options["node_key"], node_options["node_display"]))
        selected_nodes = st.multiselect(
            "Telemetry nodes",
            node_choices,
            default=node_choices,
            format_func=lambda key: node_label_map.get(key, key),
        )

        st.caption("Nodes are shown by telemetry source and cohort. For example, g4 and g11 appear under the 34662 / label 0 dataset because that is where their records exist.")

        window = st.selectbox("Time window", ["1h", "6h", "24h", "All"], index=0)
        show_labeled_only = st.checkbox("Only labeled rows", value=False)
        show_raw_only = st.checkbox("Only unlabeled rows", value=False)

    filtered = filtered_for_cohort.loc[filtered_for_cohort["node_key"].isin(selected_nodes)].copy()
    if show_labeled_only:
        filtered = filtered.loc[filtered["label_clean"].notna() & (filtered["label_clean"] != "unlabeled")].copy()
    if show_raw_only:
        filtered = filtered.loc[filtered["label_clean"].isna() | (filtered["label_clean"] == "unlabeled")].copy()

    filtered = _time_filtered_frame(filtered, window)
    if filtered.empty:
        st.warning("No rows match the current filters. Expand the cohort, node, or time window selections.")
        return

    filtered_events = events.loc[
        events["source_file"].isin(filtered["source_file"].unique())
        & events["timestamp"].isin(filtered["timestamp"].unique())
    ].copy()

    _render_metric_row(filtered, filtered_events)
    _render_design_note()

    overview_tab, drilldown_tab, insights_tab = st.tabs(["Overview", "Node drilldown", "Insights"])

    with overview_tab:
        left, right = st.columns([1.55, 1])
        with left:
            st.markdown('<div class="panel"><h3>Utilization trend</h3><div class="subtle">GPU utilization with rolling mean and incident markers.</div>', unsafe_allow_html=True)
            st.plotly_chart(_build_utilization_figure(filtered, "GPU utilization"), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="panel"><h3>Network activity</h3><div class="subtle">Derived throughput from InfiniBand delta counters.</div>', unsafe_allow_html=True)
            st.plotly_chart(_build_network_figure(filtered), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel"><h3>Incident markers</h3><div class="subtle">Rule-based markers for admin review.</div>', unsafe_allow_html=True)
            st.plotly_chart(_build_incident_timeline(filtered_events), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="panel"><h3>Dataset comparison</h3><div class="subtle">Quick comparison of the two cohorts.</div>', unsafe_allow_html=True)
            comparison = (
                filtered.groupby("cohort")
                .agg(
                    samples=("timestamp", "count"),
                    avg_gpu=("gpu_util_pct", "mean"),
                    avg_mem=("gpu_mem_util_pct", "mean"),
                    avg_network=("network_total_mb_s", "mean"),
                    incidents=("event_flag", "sum"),
                )
                .reset_index()
            )
            fig = go.Figure(
                data=[
                    go.Bar(name="GPU util", x=comparison["cohort"], y=comparison["avg_gpu"], marker_color="#4f8ef7"),
                    go.Bar(name="Network MB/s", x=comparison["cohort"], y=comparison["avg_network"], marker_color="#14b8a6"),
                ]
            )
            fig.update_layout(barmode="group", height=280, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><h3>What an admin should notice</h3><div class="subtle">Auto-generated summary from the selected window.</div>', unsafe_allow_html=True)
        for note in _top_insights(filtered, filtered_events):
            st.markdown(f"- {note}")
        st.markdown('</div>', unsafe_allow_html=True)

    with drilldown_tab:
        drill_options = (
            filtered[["node_key", "node_display", "node_short", "cohort"]]
            .drop_duplicates()
            .sort_values(["node_short", "cohort"])
        )
        drill_nodes = drill_options["node_key"].tolist()
        drill_label_map = dict(zip(drill_options["node_key"], drill_options["node_display"]))
        selected_node = st.selectbox("Select telemetry node", drill_nodes, index=0 if drill_nodes else None, format_func=lambda key: drill_label_map.get(key, key))
        node_frame = filtered.loc[filtered["node_key"] == selected_node].copy() if selected_node else filtered.copy()
        node_events = filtered_events.loc[filtered_events["node_key"] == selected_node].copy() if selected_node else filtered_events.copy()

        st.markdown('<div class="panel"><h3>Node detail</h3><div class="subtle">Use this view to inspect metric alignment and incident timing on a single node.</div>', unsafe_allow_html=True)
        node_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.64, 0.36])
        node_fig.add_trace(
            go.Scatter(x=node_frame["timestamp_dt"], y=node_frame["gpu_util_pct"], mode="lines", name="GPU util", line=dict(color="#4f8ef7", width=2)),
            row=1,
            col=1,
        )
        node_fig.add_trace(
            go.Scatter(x=node_frame["timestamp_dt"], y=node_frame["gpu_mem_util_pct"], mode="lines", name="GPU mem %", line=dict(color="#0f766e", width=2)),
            row=1,
            col=1,
        )
        if not node_events.empty:
            node_fig.add_trace(
                go.Scatter(x=node_events["timestamp_dt"], y=node_events["gpu_util_pct"], mode="markers", name="Incident", marker=dict(size=8, color="#ef4444")),
                row=1,
                col=1,
            )
        node_fig.add_trace(
            go.Scatter(x=node_frame["timestamp_dt"], y=node_frame["network_total_mb_s"], mode="lines", name="Network MB/s", line=dict(color="#14b8a6", width=2)),
            row=2,
            col=1,
        )
        node_fig.add_trace(
            go.Scatter(x=node_frame["timestamp_dt"], y=node_frame["wait_pressure"], mode="lines", name="IB pressure", line=dict(color="#8b5cf6", width=1.7, dash="dot")),
            row=2,
            col=1,
        )
        node_fig.update_layout(height=520, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=10, r=10, t=40, b=10))
        node_fig.update_yaxes(title_text="Util / mem", row=1, col=1)
        node_fig.update_yaxes(title_text="Throughput / pressure", row=2, col=1)
        node_fig.update_xaxes(title_text="Time", row=2, col=1)
        st.plotly_chart(node_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown('<div class="panel"><h3>Node metrics snapshot</h3><div class="subtle">A concise operational summary for the selected node.</div>', unsafe_allow_html=True)
            snapshot = pd.DataFrame(
                {
                    "metric": ["Samples", "Average GPU util", "Average GPU memory", "Average throughput", "Incident candidates"],
                    "value": [
                        len(node_frame),
                        f"{node_frame['gpu_util_pct'].mean():.1f}%",
                        f"{node_frame['gpu_mem_util_pct'].mean():.1f}%",
                        f"{node_frame['network_total_mb_s'].mean():.2f} MB/s",
                        int(node_events.shape[0]),
                    ],
                }
            )
            st.dataframe(snapshot, hide_index=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="panel"><h3>Incident log</h3><div class="subtle">How an admin can interpret the node window.</div>', unsafe_allow_html=True)
            _render_event_log(node_events)
            st.markdown('</div>', unsafe_allow_html=True)

    with insights_tab:
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown('<div class="panel"><h3>Correlation map</h3><div class="subtle">Shows which telemetry fields move together.</div>', unsafe_allow_html=True)
            st.plotly_chart(_build_correlation_heatmap(filtered), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="panel"><h3>Label summary</h3><div class="subtle">Counts from the labeled files currently available.</div>', unsafe_allow_html=True)
            _render_label_summary(filtered)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><h3>Cohort / node table</h3><div class="subtle">Operational summary for each source file.</div>', unsafe_allow_html=True)
        _render_node_table(filtered)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><h3>Quick interpretation</h3><div class="subtle">What this dashboard is designed to help an admin understand.</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - Stable GPU utilization with abrupt dips usually means the job is stalled or waiting on synchronization.
            - Rising IB wait / timeout counters often point to congestion before outright errors appear.
            - Comparing the two cohorts helps separate the baseline run from the annotated run.
            - The current labeled files are healthy-only, so the dashboard uses telemetry rules to surface possible degraded windows.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
