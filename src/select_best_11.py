"""
select_best_11.py  –  choose the best XI from two IPL teams
------------------------------------------------------------
Run examples
------------
python src/select_best_11.py "Mumbai Indians" "Chennai Super Kings"
python src/select_best_11.py "RCB" "RR" --venue "M Chinnaswamy Stadium" --seasons 2020 2021 2022
"""

from __future__ import annotations
from pathlib import Path
from typing import List

import argparse
import numpy as np
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize

# ────────────────────────────────────────────────────────────
# 0.  Load raw CSVs
# ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

matches = pd.read_csv(RAW_DIR / "matches.csv")
deliver = pd.read_csv(RAW_DIR / "deliveries.csv")

balls = deliver.merge(
    matches[["id", "season", "venue"]],
    left_on="match_id",
    right_on="id",
    how="left",
)

# ────────────────────────────────────────────────────────────
# 1.  Helper: baseline run-rate per season & venue
# ────────────────────────────────────────────────────────────
def season_venue_rr(df: pd.DataFrame, run_col: str) -> pd.Series:
    """
    Return a Series indexed by (season, venue) with runs per ball ×6 (i.e. RR).
    """
    grouped = df.groupby(["season", "venue"])[run_col].agg(["sum", "count"])
    rr = grouped["sum"] / grouped["count"] * 6
    return rr.rename("baseline_rr")


# ────────────────────────────────────────────────────────────
# 2.  Player value metrics
# ────────────────────────────────────────────────────────────
def batting_value(df: pd.DataFrame) -> pd.Series:
    base = season_venue_rr(df, "batsman_runs")
    merged = df.join(base, on=["season", "venue"])
    # baseline runs per ball  = RR / 6
    merged["diff"] = merged["batsman_runs"] - merged["baseline_rr"] / 6
    value = merged.groupby("batter")["diff"].sum()
    balls_faced = df.groupby("batter").size().clip(lower=1)
    return (value / balls_faced * 120).rename("bat_VR")  # per-120-ball value


def bowling_value(df: pd.DataFrame) -> pd.Series:
    base = season_venue_rr(df, "total_runs")
    merged = df.join(base, on=["season", "venue"])
    merged["diff"] = merged["baseline_rr"] / 6 - merged["total_runs"]  # runs saved
    value = merged.groupby("bowler")["diff"].sum()

    overs = (
        df.groupby(["bowler", "match_id", "over"])
        .size()
        .groupby("bowler")
        .count()
        .clip(lower=1)
    )
    return (value / overs).rename("bowl_VR")  # value per over


def fielding_value(df: pd.DataFrame) -> pd.Series:
    mask = df["dismissal_kind"].isin(["caught", "run out"])
    return df[mask].groupby("fielder")["is_wicket"].sum().rename("field_VR")


# ────────────────────────────────────────────────────────────
# 3.  Build player table for two teams + filters
# ────────────────────────────────────────────────────────────
def build_player_table(
    team1: str,
    team2: str,
    seasons: List[int] | None = None,
    venue: str | None = None,
) -> pd.DataFrame:
    df = balls.copy()
    if seasons:
        df = df[df["season"].isin(seasons)]
    if venue:
        df = df[df["venue"] == venue]

    players_pool = pd.unique(
        pd.concat(
            [
                df.loc[df["batting_team"] == team1, "batter"],
                df.loc[df["bowling_team"] == team1, "bowler"],
                df.loc[df["batting_team"] == team2, "batter"],
                df.loc[df["bowling_team"] == team2, "bowler"],
            ]
        )
    )

    bat   = batting_value(df[df["batter"].isin(players_pool)])
    bowl  = bowling_value(df[df["bowler"].isin(players_pool)])
    field = fielding_value(df[df["fielder"].isin(players_pool)])

    tbl = pd.concat([bat, bowl, field], axis=1).fillna(0)

    tbl["is_bowler"] = tbl["bowl_VR"] > 0
    tbl["is_keeper"] = tbl.index.to_series().str.contains(
        r"Dhoni|Pant|Saha|Parthiv|Karthik|de Kock|Samson"
    )

    tbl["value"] = tbl["bat_VR"] + tbl["bowl_VR"] + 0.2 * tbl["field_VR"]
    return tbl.sort_values("value", ascending=False)


# ────────────────────────────────────────────────────────────
# 4.  Optimiser – ILP
# ────────────────────────────────────────────────────────────
def select_best_xi(tbl: pd.DataFrame) -> pd.DataFrame:
    prob = LpProblem("Best_XI", LpMaximize)
    x = {p: LpVariable(name=p, cat="Binary") for p in tbl.index}

    prob += lpSum(tbl.loc[p, "value"] * x[p] for p in tbl.index)
    prob += lpSum(x.values()) == 11
    prob += lpSum((1 - int(tbl.loc[p, "is_bowler"])) * x[p] for p in tbl.index) >= 2
    prob += lpSum(int(tbl.loc[p, "is_bowler"]) * x[p] for p in tbl.index) >= 5
    prob += lpSum(int(tbl.loc[p, "is_keeper"]) * x[p] for p in tbl.index) >= 1

    is_overseas = tbl.index.to_series().str.contains(r"^[A-Z][a-z]+\s[A-Z]")
    prob += lpSum(int(flag) * x[p] for p, flag in zip(tbl.index, is_overseas)) <= 4

    prob.solve()

    chosen = [p for p, var in x.items() if var.value() == 1]
    xi = tbl.loc[chosen].sort_values("value", ascending=False)
    xi["role"] = xi.apply(
        lambda r: "WK" if r["is_keeper"] else "Bowler" if r["is_bowler"] else "Batter",
        axis=1,
    )
    return xi


# ────────────────────────────────────────────────────────────
# 5.  CLI
# ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("team1")
    ap.add_argument("team2")
    ap.add_argument("--venue")
    ap.add_argument("--seasons", nargs="*", type=int)
    args = ap.parse_args()

    tbl = build_player_table(args.team1, args.team2, args.seasons, args.venue)
    xi  = select_best_xi(tbl)

    hdr = f"Best XI: {args.team1} + {args.team2}"
    if args.venue:
        hdr += f" @ {args.venue}"
    if args.seasons:
        hdr += f" | Seasons: {', '.join(map(str, args.seasons))}"
    print("\n" + hdr)
    print(
        xi[["value", "bat_VR", "bowl_VR", "field_VR", "role"]]
        .round(2)
        .to_string()
    )
    print(f"\nProjected net run differential: {xi['value'].sum():.1f}")


if __name__ == "__main__":
    main()
