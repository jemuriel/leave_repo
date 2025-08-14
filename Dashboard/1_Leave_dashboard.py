# app.py
import os
import sys

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from pathlib import Path
import calendar
import plotly.express as px
import plotly.graph_objects as go

MONTH_ORDER = list(calendar.month_abbr)[1:]  # ['Jan','Feb',...,'Dec']
WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Ensure the repo root is in sys.path so local modules can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# -----------------------------
# Config & constants
# -----------------------------
st.set_page_config(page_title="Leave Pattern Explorer", layout="wide")

TYPE_MAP = {
    "DFP": "DFP",       # day without pay
    "PER_L": "PER_L",   # personal leave
    "RDO": "RDO",       # annual leave / RDO
    "UNK": "UNK",       # sick leave
    "WORK": "WORK",     # normal work
}

# Patterns of interest (current_type -> next_type)
PATTERNS = {("UNK", "RDO"), ("UNK", "PER_L"), ("RDO", "UNK"), ("PER_L", "UNK")}

WEEKDAY_LABELS = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
MONTH_LABELS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul",
                8: "Aug", 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Create a proper date
    if not {"YEAR", "MONTH", "DAY"}.issubset(df.columns):
        raise ValueError("Expected YEAR, MONTH, DAY columns.")
    df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]])
    # Normalise TYPE
    df["TYPE"] = df["TYPE"].map(TYPE_MAP).fillna(df["TYPE"])
    df = df[df['TYPE']!='WORK']
    # Tidy weekday labels if present (assuming 1=Mon .. 7=Sun)
    if "WEEKDAY" in df.columns:
        df["WEEKDAY_NAME"] = df["WEEKDAY"].map(WEEKDAY_LABELS).fillna(df["WEEKDAY"])
    else:
        # If WEEKDAY not present, derive it (Monday=1..Sunday=7)
        df["WEEKDAY"] = df["DATE"].dt.weekday + 1
        df["WEEKDAY_NAME"] = df["WEEKDAY"].map(WEEKDAY_LABELS)

    df['MONTH_NAME'] = df['MONTH'].map(MONTH_LABELS)
    return df

def compute_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each NAME, sort by DATE and compute next day's TYPE and metadata.
    Returns only rows where (TYPE, NEXT_TYPE) is in PATTERNS.
    """
    df = df.sort_values(["NAME", "DATE"]).copy()
    df["NEXT_TYPE"] = df.groupby("NAME")["TYPE"].shift(-1)
    df["NEXT_DATE"] = df.groupby("NAME")["DATE"].shift(-1)
    df["NEXT_DAY"] = df["NEXT_DATE"].dt.day
    df["NEXT_MONTH"] = df["NEXT_DATE"].dt.month
    df["NEXT_YEAR"] = df["NEXT_DATE"].dt.year
    df["NEXT_WEEKDAY"] = df["NEXT_DATE"].dt.weekday + 1
    df["NEXT_WEEKDAY_NAME"] = df["NEXT_WEEKDAY"].map(WEEKDAY_LABELS)

    # Keep only target patterns
    mask = list(zip(df["TYPE"], df["NEXT_TYPE"]))
    df_patterns = df[np.isin(mask, list(PATTERNS))].copy()
    return df_patterns

import calendar
import plotly.express as px
import plotly.graph_objects as go

MONTH_ORDER = list(calendar.month_abbr)[1:]  # ['Jan','Feb',...,'Dec']

def make_histogram_month(transitions: pd.DataFrame, barmode: str = "stack") -> go.Figure:
    """
    Bar chart of % within each (Depot, Month), bars divided by transition type.
    Uses wider facet wrap and tiny row spacing to avoid the vertical-spacing error.
    """
    if transitions.empty:
        return go.Figure()

    ALLOWED = set(PATTERNS)

    base = (
        transitions
        .assign(
            MONTH_NUM=lambda d: d["DATE"].dt.month.astype(int),
            MONTH_LABEL=lambda d: pd.Categorical(d["DATE"].dt.strftime("%b"),
                                                 categories=MONTH_ORDER, ordered=True),
            TRANSITION=lambda d: np.where(
            [(t, n) in ALLOWED for t, n in zip(d["TYPE"], d["NEXT_TYPE"])],
            d["TYPE"] + " → " + d["NEXT_TYPE"],
            np.nan
            )
        )
        .groupby(["MONTH_NUM", "MONTH_LABEL", "TRANSITION"], as_index=False)
        .size()
        .rename(columns={"size": "COUNT"})
    )

    # % within each (DEPO, MONTH)
    base["MONTH_TOTAL"] = base.groupby(["MONTH_NUM"])["COUNT"].transform("sum")
    base["PCT"] = base["COUNT"] / base["MONTH_TOTAL"] * 100

    # Fewer rows: wrap more columns; smaller row spacing to satisfy Plotly's constraint
    fig = px.bar(
        base,
        x="MONTH_LABEL",
        y="PCT",
        color="TRANSITION",
        # facet_col="DEPO_NAME",
        facet_col_wrap=6,            # <-- wider wrap (fewer facet rows)
        facet_row_spacing=0.01,      # <-- tiny vertical spacing (avoids the error)
        facet_col_spacing=0.03,
        category_orders={"MONTH_LABEL": MONTH_ORDER},
        labels={"MONTH_LABEL": "Month", "PCT": "Share of month (%)", "TRANSITION": "Transition Type"},
        hover_data={"PCT": ":.1f", "COUNT": True, "MONTH_LABEL": True, "TRANSITION": True},
    )
    fig.update_layout(barmode=barmode, yaxis_title="Share of month (%)", legend_title_text="Transition Type")

    # Lock all y-axes to 0–100 across facets
    fig.for_each_yaxis(lambda ax: ax.update(range=[0, 100]))

    # Optional: scale height with number of facet rows so subplots remain readable
    # n_depos = base["DEPO_NAME"].nunique()
    # rows = int(np.ceil(n_depos / 6)) if n_depos else 1
    # fig.update_layout(height=max(350, 250 * rows))

    return fig


def make_histogram_depo(transitions: pd.DataFrame, barmode: str = "stack") -> go.Figure:
    """
    Bar chart of % within each (Depot, Month), bars divided by transition type.
    Uses wider facet wrap and tiny row spacing to avoid the vertical-spacing error.
    """
    if transitions.empty:
        return go.Figure()

    ALLOWED = set(PATTERNS)

    base = (
        transitions
        .assign(
            MONTH_NUM=lambda d: d["DATE"].dt.month.astype(int),
            MONTH_LABEL=lambda d: pd.Categorical(d["DATE"].dt.strftime("%b"),
                                                 categories=MONTH_ORDER, ordered=True),
            TRANSITION=lambda d: np.where(
            [(t, n) in ALLOWED for t, n in zip(d["TYPE"], d["NEXT_TYPE"])],
            d["TYPE"] + " → " + d["NEXT_TYPE"],
            np.nan
            )
        )
        .groupby(["DEPO_NAME", "TRANSITION"], as_index=False)
        .size()
        .rename(columns={"size": "COUNT"})
    )

    # % within each (DEPO, MONTH)
    base["DEPO_TOTAL"] = base.groupby(["DEPO_NAME"])["COUNT"].transform("sum")
    base["PCT"] = base["COUNT"] / base["DEPO_TOTAL"] * 100

    # Fewer rows: wrap more columns; smaller row spacing to satisfy Plotly's constraint
    fig = px.bar(
        base,
        x="DEPO_NAME",
        y="PCT",
        color="TRANSITION",
        # facet_col="DEPO_NAME",
        facet_col_wrap=6,            # <-- wider wrap (fewer facet rows)
        facet_row_spacing=0.01,      # <-- tiny vertical spacing (avoids the error)
        facet_col_spacing=0.03,
        category_orders={"DEPO_NAME": MONTH_ORDER},
        labels={"DEPO_NAME": "Depo", "PCT": "Share of Depo (%)", "TRANSITION": "Transition Type"},
        hover_data={"PCT": ":.1f", "COUNT": True, "DEPO_NAME": True, "TRANSITION": True},
    )
    fig.update_layout(barmode=barmode, yaxis_title="Share of month (%)", legend_title_text="Transition Type")

    # Lock all y-axes to 0–100 across facets
    fig.for_each_yaxis(lambda ax: ax.update(range=[0, 100]))

    # Optional: scale height with number of facet rows so subplots remain readable
    # n_depos = base["DEPO_NAME"].nunique()
    # rows = int(np.ceil(n_depos / 6)) if n_depos else 1
    # fig.update_layout(height=max(350, 250 * rows))

    return fig


def make_day_month_heatmap(transitions: pd.DataFrame) -> go.Figure:
    if transitions.empty:
        return go.Figure()


    # Current & next day tall table
    part_current = transitions.assign(
        MONTH=lambda d: d["DATE"].dt.month.astype(int),
        DAY=lambda d: d["DATE"].dt.day.astype(int)
    )[["MONTH", "DAY"]]
    part_next = transitions[["NEXT_MONTH", "NEXT_DAY"]].rename(columns={"NEXT_MONTH": "MONTH", "NEXT_DAY": "DAY"}).dropna()
    part_next["MONTH"] = part_next["MONTH"].astype(int)
    part_next["DAY"] = part_next["DAY"].astype(int)

    tall = pd.concat([part_current, part_next], ignore_index=True)
    agg = tall.groupby(["MONTH", "DAY"], as_index=False).size().rename(columns={"size": "COUNT"})

    # Full grid and month labels
    grid = pd.MultiIndex.from_product([range(1, 13), range(1, 32)], names=["MONTH", "DAY"]).to_frame(index=False)
    heat = grid.merge(agg, on=["MONTH", "DAY"], how="left").fillna({"COUNT": 0})
    heat["MONTH_LABEL"] = pd.Categorical([calendar.month_abbr[m] for m in heat["MONTH"]], categories=MONTH_ORDER, ordered=True)

    # Pivot to matrix (y=day, x=month label)
    mat = heat.pivot(index="DAY", columns="MONTH_LABEL", values="COUNT").reindex(index=range(1, 32), columns=MONTH_ORDER)

    fig = go.Figure(
        data=go.Heatmap(
            z=mat.values,
            x=mat.columns,
            y=mat.index,
            coloraxis="coloraxis",
            hovertemplate="Month=%{x}<br>Day=%{y}<br>Occurrences=%{z}<extra></extra>"
        )
    )
    fig.update_layout(
        coloraxis=dict(colorbar_title="Occurrences"),
        xaxis_title="Month",
        yaxis_title="Day of Month",
        yaxis=dict(autorange="reversed")  # optional; remove if you want day 1 at bottom
    )
    return fig


def make_weekday_month_heatmap(transitions: pd.DataFrame) -> go.Figure:
    if transitions.empty:
        return go.Figure()

    part_current = transitions.assign(
        MONTH=lambda d: d["DATE"].dt.month.astype(int),
        WEEKDAY_NAME=lambda d: d["WEEKDAY_NAME"]
    )[["MONTH", "WEEKDAY_NAME"]]
    part_next = transitions[["NEXT_MONTH", "NEXT_WEEKDAY_NAME"]].rename(
        columns={"NEXT_MONTH": "MONTH", "NEXT_WEEKDAY_NAME": "WEEKDAY_NAME"}
    ).dropna()
    part_next["MONTH"] = part_next["MONTH"].astype(int)

    tall = pd.concat([part_current, part_next], ignore_index=True)
    agg = tall.groupby(["MONTH", "WEEKDAY_NAME"], as_index=False).size().rename(columns={"size": "COUNT"})
    agg["MONTH_LABEL"] = pd.Categorical([calendar.month_abbr[m] for m in agg["MONTH"]], categories=MONTH_ORDER, ordered=True)

    mat = (
        agg.pivot(index="WEEKDAY_NAME", columns="MONTH_LABEL", values="COUNT")
        .reindex(index=WEEKDAY_ORDER, columns=MONTH_ORDER)
        .fillna(0)
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=mat.values,
            x=mat.columns,
            y=mat.index,
            coloraxis="coloraxis",
            hovertemplate="Month=%{x}<br>Weekday=%{y}<br>Occurrences=%{z}<extra></extra>"
        )
    )
    fig.update_layout(
        coloraxis=dict(colorbar_title="Occurrences"),
        xaxis_title="Month",
        yaxis_title="Weekday"
    )
    return fig

import plotly.graph_objects as go

def make_pareto_by_depot_plotly(transitions: pd.DataFrame, top_n: int | None = None) -> go.Figure:
    """
    Pareto chart of transition events by depot (DEPO_NAME).
    - Bars: event counts per depot (sorted descending)
    - Line: cumulative percentage (% of total events)
    Set top_n to limit to the top N depots (e.g., 15) for readability.
    """
    if transitions.empty:
        return go.Figure()

    # Aggregate event counts per depot
    counts = (
        transitions.groupby("DEPO_NAME", as_index=False)
        .size()
        .rename(columns={"size": "COUNT"})
        .sort_values("COUNT", ascending=False)
    )

    # Optional: limit to top N depots
    if top_n is not None and top_n > 0:
        counts = counts.head(top_n)

    # Cumulative percentage
    counts["CUM_COUNT"] = counts["COUNT"].cumsum()
    total = counts["COUNT"].sum()
    counts["CUM_PCT"] = counts["CUM_COUNT"] / total * 100.0

    fig = go.Figure()

    # Bars (counts)
    fig.add_bar(
        x=counts["DEPO_NAME"],
        y=counts["COUNT"],
        name="Events",
        hovertemplate="Depot=%{x}<br>Events=%{y:,}<extra></extra>",
    )

    # Line (cumulative %)
    fig.add_scatter(
        x=counts["DEPO_NAME"],
        y=counts["CUM_PCT"],
        name="Cumulative %",
        mode="lines+markers",
        yaxis="y2",
        hovertemplate="Depot=%{x}<br>Cumulative=%{y:.1f}%<extra></extra>",
    )

    fig.update_layout(
        title="Pareto of Transition Events by Depot",
        xaxis_title="Depot",
        yaxis=dict(title="Events (count)"),
        yaxis2=dict(
            title="Cumulative %",
            overlaying="y",
            side="right",
            range=[0, 100],
            tickformat=".0f%%",
        ),
        legend_title_text="",
        margin=dict(l=40, r=40, t=60, b=80),
    )
    fig.update_xaxes(tickangle=-45)  # improves readability for long depot names
    return fig

# -----------------------------
# UI
# -----------------------------
st.title("Leave Pattern Explorer")
st.caption("Analyses day-to-day leave transitions for patterns of interest.")

st.markdown("""
### About this dashboard

This dashboard explores **day-to-day leave transitions** to highlight patterns that may indicate behavioural or rostering effects.

**Types of leave:**  
  - `DFP` – Day without pay  
  - `PER_L` – Personal leave  
  - `RDO` – Annual leave
  - `UNK` – Sick leave  
  - `WORK` – Normal working day

**Patterns analysed (transitions between consecutive days for the same person)**
- `UNK → RDO`  (Sick leave → RDO)
- `UNK → PER_L` (Sick leave → Personal leave)
- `RDO → UNK`  (RDO → Sick leave)
- `PER_L → UNK` (Personal leave → Sick leave)

Only these four transitions are included in counts and visuals.
""")
st.divider()


# Data path input (defaults to the sample filename)
default_path = Path('csv/leave_data.csv')
# csv_path = st.sidebar.text_input("CSV path", value=str(default_path))

try:
    data = load_data(default_path)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

st.markdown("### Depot filter")
depos = sorted(data["DEPO_NAME"].dropna().astype(str).unique().tolist())

# Optional: remember selection across reruns
_default_depos = st.session_state.get("depo_top_selection", depos)

col_depo, col_btn_all, col_btn_clear = st.columns([6, 1, 1])
with col_depo:
    sel_depos = st.multiselect(
        "Depot(s)",
        options=depos,
        default=_default_depos,
        key="depo_top"
    )
with col_btn_all:
    if st.button("Select all", use_container_width=True):
        sel_depos = depos
        st.session_state["depo_top"] = depos
with col_btn_clear:
    if st.button("Clear", use_container_width=True):
        sel_depos = []
        st.session_state["depo_top"] = []

# Persist current selection for next rerun
st.session_state["depo_top_selection"] = sel_depos

# st.dataframe(data)
# Sidebar filters
st.sidebar.header("Filters")

# names = sorted(data["NAME"].dropna().unique().tolist())
months = sorted(data["MONTH_NAME"].dropna().unique().astype(str).tolist())
years = sorted(data["YEAR"].dropna().unique().astype(int).tolist())
depo = sorted(data["DEPO_NAME"].dropna().unique().astype(str).tolist())

# sel_names = st.sidebar.multiselect("Name(s)", options=names, default=names)
sel_months = st.sidebar.multiselect("Month(s)", options=months, default=months)
sel_years = st.sidebar.multiselect("Year(s)", options=years, default=years)
# sel_depo = st.sidebar.multiselect("Depo(s)", options=depo, default=depo)

# Apply filters
filtered = data[
    # data["NAME"].isin(sel_names) &
    data["MONTH_NAME"].isin(sel_months) &
    data["YEAR"].isin(sel_years) &
    data["DEPO_NAME"].isin(sel_depos)
].copy()

# Compute transitions on the filtered set
transitions = compute_transitions(filtered)

# Optional: show only the four patterns (already filtered), but keep a selector to toggle detail
# with st.expander("Show transition table (filtered to target patterns)"):
#     st.dataframe(
#         transitions[
#             [
#                 "DEPO_NAME", "NAME", "DATE", "TYPE",
#                 "NEXT_DATE", "NEXT_TYPE",
#                 "YEAR", "MONTH", "DAY", "WEEKDAY_NAME"
#             ]
#         ],
#         use_container_width=True,
#         hide_index=True
#     )

# -----------------------------
# Plots
# -----------------------------
# col1, col2 = st.columns([1, 1])
#
# with col1:
#     st.subheader("Frequency by Depot and Month")
#     st.altair_chart(make_histogram(transitions), use_container_width=True)
#
# with col2:
#     st.subheader("Day vs Month Heatmap (both days coloured per event)")
#     st.altair_chart(make_day_month_heatmap(transitions), use_container_width=True)
#
# st.subheader("Weekday vs Month Heatmap (both days coloured per event)")
# st.altair_chart(make_weekday_month_heatmap(transitions), use_container_width=True)

st.subheader("Pareto: Events by Depot")
top_n = st.slider("Show top N depots", min_value=5, max_value=40, value=15, step=1)
st.plotly_chart(make_pareto_by_depot_plotly(transitions, top_n=top_n), use_container_width=True)

st.subheader("Frequency by Month")
barmode = st.radio("Bar layout", ["stack", "group"], horizontal=True, key="bar_layout")
st.plotly_chart(make_histogram_month(transitions, barmode=barmode), use_container_width=True)

st.subheader("Frequency by Depot")
st.plotly_chart(make_histogram_depo(transitions, barmode=barmode), use_container_width=True)

# st.subheader("Day × Month Heatmap (both days coloured per event)")
# st.plotly_chart(make_day_month_heatmap(transitions), use_container_width=True)
#
st.subheader("Weekday × Month Heatmap (both days coloured per event)")
st.plotly_chart(make_weekday_month_heatmap(transitions), use_container_width=True)


# st.markdown("""
# *Notes*
# - Patterns included: **UNK → RDO**, **UNK → PER_L**, **RDO → UNK**, **PER_L → UNK**.
# - Each occurrence adds one count to the cell for the day it happened **and** the following day.
# - If your weekday numbers differ, adjust `WEEKDAY_LABELS` at the top.
# """)
