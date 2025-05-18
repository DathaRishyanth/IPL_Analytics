

"""
Streamlit dashboard with TWO features:

1) Real-time win-probability slider (LightGBM model you trained).
2) Best-XI optimiser that picks the best combined XI from two teams.

Prereqs
-------
• models/lgbm_winprob.joblib              (from `python src/train.py`)
• data/raw/matches.csv + deliveries.csv   (original IPL CSVs)
• src/select_best_11.py                   (patched version with fallback)
• src/features.py                         (for win-prob features)
"""

from pathlib import Path
from typing import List

import joblib
import pandas as pd
import streamlit as st

from features import add_features, design_matrix
from select_best_11 import build_player_table, select_best_xi

# ────────────────────────────────────────────────────────────
# Paths & data
# ────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
RAW_DIR  = ROOT / "data" / "raw"
MODEL    = ROOT / "models" / "lgbm_winprob.joblib"
REPORTS  = ROOT / "reports"

# Load model artefact for WinProb tab
artefact = joblib.load(MODEL)
model    = artefact["model"]
columns  = artefact["columns"]

# Load matches once for dropdowns in Best-XI tab
matches_df = pd.read_csv(RAW_DIR / "matches.csv")
teams = sorted(pd.unique(pd.concat([matches_df["team1"], matches_df["team2"]]).dropna()))
venues = sorted(matches_df["venue"].dropna().unique())
seasons = sorted(matches_df["season"].unique())

# ────────────────────────────────────────────────────────────
# Streamlit layout
# ────────────────────────────────────────────────────────────
st.set_page_config(page_title="IPL Analytics Hub", page_icon="🏏", layout="wide")
st.title("🏏  IPL Analytics Hub")

tab1, tab2 = st.tabs(["Win-Probability", "Best XI & Stats"])

# ===========================================================
# TAB 1 – Real-time Win-Prob
# ===========================================================
with tab1:
    st.header("Real-time Win-Probability Engine")

    # Sidebar inputs
    st.sidebar.header("Match State")
    over   = st.sidebar.slider("Over (0-19)", 0, 19, 10)
    ball   = st.sidebar.slider("Ball (1-6)", 1, 6, 3)
    runs   = st.sidebar.number_input("Current score", 0, 300, 90)
    wkts   = st.sidebar.slider("Wickets fallen", 0, 10, 3)
    target = st.sidebar.number_input("Target runs (0 if 1st innings)", 0, 300, 150)

    row = {
        "match_id":    0,
        "inning":      2 if target else 1,
        "over":        over,
        "ball":        ball,
        "runs_cum":    runs,
        "wickets":     wkts,
        "ball_number": over * 6 + (ball - 1),
        "target_runs": target if target else None,
        "batting_team": "teamA",
        "bowling_team": "teamB",
    }
    df_feat = add_features(pd.DataFrame([row]))
    X       = design_matrix(df_feat, existing_cols=columns)
    prob    = model.predict(X)[0]

    st.metric("Win probability", f"{prob*100:.1f} %")

    # Optional report gallery
    with st.expander("Season-overview plots"):
        for img in sorted(REPORTS.glob("*.png")):
            st.image(str(img), caption=img.stem)

# ===========================================================
# TAB 2 – Best XI optimiser
# ===========================================================
with tab2:
    st.header("Best XI Selector")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team A", teams, index=teams.index("Mumbai Indians") if "Mumbai Indians" in teams else 0)
    with col2:
        team2 = st.selectbox("Team B", teams, index=teams.index("Chennai Super Kings") if "Chennai Super Kings" in teams else 1)

    venue_sel   = st.selectbox("Venue (optional)", [""] + venues)
    season_mult = st.multiselect("Seasons (blank = all)", seasons)

    if team1 == team2:
        st.warning("Please pick two *different* teams.")
    elif st.button("Compute Best XI ▶"):
        try:
            tbl = build_player_table(
                team1,
                team2,
                seasons=season_mult if season_mult else None,
                venue=venue_sel or None,
            )
            xi = select_best_xi(tbl)

            st.success("Optimised XI ready!")
            st.dataframe(
                xi[["role", "value", "bat_VR", "bowl_VR", "field_VR"]].round(2),
                use_container_width=True,
            )
            st.metric("Projected run differential", f"{xi['value'].sum():.1f}")

        except ValueError as e:
            st.error(str(e))
            


