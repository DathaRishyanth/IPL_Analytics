"""
Generate IPL summary plots (batting, bowling, fielding, toss, venues…).

Usage
-----
python src/plot_stats.py
"""

from pathlib import Path
import glob
import re
import textwrap
import pandas as pd
import matplotlib.pyplot as plt

RAW_DIR     = Path(__file__).resolve().parents[1] / "data" / "raw"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)


# --------------------------------------------------------------------------- #
# 0. Load data (all seasons present)
# --------------------------------------------------------------------------- #
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    m_files = glob.glob(str(RAW_DIR / "matches.csv"))
    b_files = glob.glob(str(RAW_DIR / "deliveries.csv"))

    if not m_files or not b_files:
        raise FileNotFoundError("Add matches_*.csv & balls_*.csv to data/raw/ first.")

    matches = pd.concat((pd.read_csv(f) for f in m_files), ignore_index=True)
    balls   = pd.concat((pd.read_csv(f) for f in b_files), ignore_index=True)

    # attach match meta to balls
    meta_cols = ["id", "winner", "venue", "city", "toss_winner",
                 "toss_decision", "team1", "team2"]
    balls = balls.merge(matches[meta_cols], left_on="match_id", right_on="id", how="left")
    return matches, balls


matches, balls = load()


# --------------------------------------------------------------------------- #
# Helper for tidy file names
# --------------------------------------------------------------------------- #
def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text.lower()).strip("_")


def save_fig(title: str):
    fname = REPORTS_DIR / f"{slugify(title)}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()
    return fname.relative_to(REPORTS_DIR.parent)


saved = []  # list of relative paths we’ll print at the end


# --------------------------------------------------------------------------- #
# 1. Batting plots
# --------------------------------------------------------------------------- #
runs = balls.groupby("batter")["batsman_runs"].sum()
runs_top = runs.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
plt.barh(runs_top.index[::-1], runs_top.values[::-1])
plt.title("Top 10 Run-Scorers (All IPL Seasons)")
plt.xlabel("Runs")
saved.append(save_fig("top10_batters_runs"))


# --------------------------------------------------------------------------- #
# 2. Lowest scorers  (min ≥1 inning to avoid noise)
# --------------------------------------------------------------------------- #
innings_played = balls.groupby("batter")["inning"].nunique()
valid = runs[(innings_played >= 1) & (runs > 0)].sort_values().head(10)

plt.figure(figsize=(8, 5))
plt.barh(valid.index, valid.values)
plt.title("Lowest Run-Scorers (≥1 run)")
plt.xlabel("Runs")
saved.append(save_fig("lowest_batters_runs"))


# --------------------------------------------------------------------------- #
# 3. Bowling stats (wickets, top-5 table style bar)
# --------------------------------------------------------------------------- #
bowler_dismissals = {
    "bowled", "caught", "lbw", "caught and bowled", "stumped",
    "hit wicket"
}
wkt_mask = (balls["is_wicket"] == 1) & balls["dismissal_kind"].isin(bowler_dismissals)
wickets = balls[wkt_mask].groupby("bowler")["is_wicket"].count()

wickets_top5 = wickets.sort_values(ascending=False).head(5)
plt.figure(figsize=(8, 5))
plt.barh(wickets_top5.index[::-1], wickets_top5.values[::-1])
plt.title("Top 5 Wicket-Takers")
plt.xlabel("Wickets")
saved.append(save_fig("top5_bowlers_wickets"))

# zero-wicket bowlers pie
zero_ct = (wickets == 0).sum()
plt.figure()
plt.pie([zero_ct, len(wickets) - zero_ct],
        labels=["0 wickets", "≥1 wicket"],
        autopct="%d")
plt.title("Bowlers with Zero vs Some Wickets")
saved.append(save_fig("zero_wicket_bowlers"))


# --------------------------------------------------------------------------- #
# 4. Fielding: catches & run-outs
# --------------------------------------------------------------------------- #
catches = balls[balls["dismissal_kind"] == "caught"].groupby("fielder")["is_wicket"].count()
runouts = balls[balls["dismissal_kind"] == "run out"].groupby("fielder")["is_wicket"].count()

for series, label in [(catches, "Catches"), (runouts, "Run-outs")]:
    s_top = series.sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    plt.barh(s_top.index[::-1], s_top.values[::-1])
    plt.title(f"Top 10 Fielders – {label}")
    plt.xlabel("Dismissals")
    saved.append(save_fig(f"top10_fielders_{label.lower()}"))


# Wicket mode distribution
total_caught = catches.sum()
total_runout = runouts.sum()
other = int(balls["is_wicket"].sum() - total_caught - total_runout)

plt.figure()
plt.pie([total_caught, total_runout, other],
        labels=["Caught", "Run-out", "Other"],
        autopct="%1.0f%%")
plt.title("Wicket Modes")
saved.append(save_fig("wicket_mode_distribution"))


# --------------------------------------------------------------------------- #
# 5. Toss stats per team & advantage
# --------------------------------------------------------------------------- #
toss_cnt = matches["toss_winner"].value_counts()
toss_cnt_top = toss_cnt.head(10)

plt.figure(figsize=(8, 5))
plt.barh(toss_cnt_top.index[::-1], toss_cnt_top.values[::-1])
plt.title("Tosses Won – Top 10 Teams")
plt.xlabel("Toss wins")
saved.append(save_fig("toss_wins_by_team"))

matches["won_after_toss"] = matches["toss_winner"] == matches["winner"]
advantage = matches["won_after_toss"].mean()

plt.figure(figsize=(4, 4))
plt.bar(["Won after toss", "Lost after toss"],
        [matches["won_after_toss"].sum(),
         len(matches) - matches["won_after_toss"].sum()])
plt.title(f"Match Outcome after Winning Toss\n(Win rate = {advantage:.1%})")
saved.append(save_fig("toss_advantage"))


# --------------------------------------------------------------------------- #
# 6. Team success & boundaries
# --------------------------------------------------------------------------- #
wins = matches["winner"].value_counts().head(10)
plt.figure(figsize=(8, 5))
plt.barh(wins.index[::-1], wins.values[::-1])
plt.title("Most Successful Teams (Wins)")
plt.xlabel("Wins")
saved.append(save_fig("team_success"))


sixes_team = balls[balls["batsman_runs"] == 6].groupby("batting_team").size().sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
plt.barh(sixes_team.index[::-1], sixes_team.values[::-1])
plt.title("Sixes by Team – Top 10")
plt.xlabel("Sixes")
saved.append(save_fig("team_sixes"))

fours_team = balls[balls["batsman_runs"] == 4].groupby("batting_team").size().sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
plt.barh(fours_team.index[::-1], fours_team.values[::-1])
plt.title("Fours by Team – Top 10")
plt.xlabel("Fours")
saved.append(save_fig("team_fours"))


# --------------------------------------------------------------------------- #
# 7. Most common toss decisions (global & per team)
# --------------------------------------------------------------------------- #
decisions = matches["toss_decision"].value_counts()
plt.figure()
plt.bar(decisions.index, decisions.values)
plt.title("Toss Decisions – Overall")
plt.ylabel("Count")
saved.append(save_fig("toss_decision_overall"))


# decision heat-map style: bar for each team’s preferred decision
team_decision = matches.groupby(["toss_winner", "toss_decision"]).size().unstack(fill_value=0)
for team, row in team_decision.iterrows():
    plt.figure(figsize=(4, 3))
    plt.bar(row.index, row.values)
    plt.title(f"{team} – Toss Decisions")
    plt.ylabel("Count")
    saved.append(save_fig(f"toss_decision_{slugify(team)}"))


# --------------------------------------------------------------------------- #
# 8. Matches hosted by city
# --------------------------------------------------------------------------- #
city_counts = matches["city"].value_counts().head(15)
plt.figure(figsize=(8, 5))
plt.barh(city_counts.index[::-1], city_counts.values[::-1])
plt.title("Matches Hosted – Top 15 Cities")
plt.xlabel("Matches")
saved.append(save_fig("matches_by_city"))

# --------------------------------------------------------------------------- #
# 9. Lucky venues for the most successful team
# --------------------------------------------------------------------------- #
top_team = wins.idxmax()
venue_played = matches[(matches["team1"] == top_team) | (matches["team2"] == top_team)].groupby("venue").size()
venue_wins   = matches[matches["winner"] == top_team]["venue"].value_counts()
luck = (venue_wins / venue_played).dropna().sort_values(ascending=False).head(8)

plt.figure(figsize=(8, 5))
plt.barh(luck.index[::-1], (luck.values * 100)[::-1])
plt.title(f"‘Lucky’ Venues for {top_team} (%)")
plt.xlabel("Win percentage")
saved.append(save_fig("lucky_venues_top_team"))


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
print("\nSaved the following plots:")
for p in saved:
    print("  ", p)
print("\nAll images live in the 'reports/' folder – ready for your résumé, blog, or Streamlit gallery.")
