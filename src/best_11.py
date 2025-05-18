"""
Streamlit front-end for the Best-XI optimiser.

Launch:
    streamlit run src/app_best_xi.py
"""

from pathlib import Path
import streamlit as st
import pandas as pd

# import helper functions from your selector module
from select_best_11 import build_player_table, select_best_xi

# ────────────────────────────────────────────────────────────
# Load raw data once (for dropdown options)
# ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

matches = pd.read_csv(RAW_DIR / "matches.csv")

all_teams  = sorted(
    pd.unique(
        pd.concat([matches["team1"], matches["team2"]]).dropna()
    )
)
all_venues = sorted(matches["venue"].dropna().unique())
all_seasons = sorted(matches["season"].unique())


# ────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────
st.set_page_config(page_title="IPL Best XI Selector", page_icon="🏏")
st.title("🏏  IPL – Build the Best Combined XI")

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team A", all_teams, index=all_teams.index("Mumbai Indians") if "Mumbai Indians" in all_teams else 0)
with col2:
    team2 = st.selectbox("Team B", all_teams, index=all_teams.index("Chennai Super Kings") if "Chennai Super Kings" in all_teams else 1)

venue = st.selectbox("Venue (optional)", [""] + all_venues)
season_sel = st.multiselect("Seasons (empty = all)", all_seasons, default=[])

st.markdown("---")

if st.button("Compute Best XI ▶"):
    st.info("Crunching numbers…")

    tbl = build_player_table(
        team1,
        team2,
        seasons=season_sel if season_sel else None,
        venue=venue or None,
    )
    xi = select_best_xi(tbl)

    # display results
    st.subheader(f"Best XI combining **{team1}** & **{team2}**")
    if venue:
        st.caption(f"*Venue filter*: {venue}")
    if season_sel:
        st.caption(f"*Seasons*: {', '.join(map(str, season_sel))}")

    st.dataframe(
        xi[["role", "value", "bat_VR", "bowl_VR", "field_VR"]].round(2),
        height=400,
    )

    st.metric(
        "Projected net run differential",
        f"{xi['value'].sum():.1f}",
        help="Sum of composite value metrics for the selected XI."
    )
