"""
ETL — merge match-level and ball-by-ball tables, add labels, save parquet.
Usage  :  python src/etl.py
"""

import pathlib
import pandas as pd

RAW_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def build_ball_frame(matches_df: pd.DataFrame, balls_df: pd.DataFrame) -> pd.DataFrame:
    # Merge match meta onto every ball
    m_cols = ["id", "winner", "target_runs"]
    df = balls_df.merge(matches_df[m_cols], left_on="match_id", right_on="id", how="left")

    df.sort_values(["match_id", "inning", "over", "ball"], inplace=True)
    df["runs_cum"] = df.groupby(["match_id", "inning"])["total_runs"].cumsum()
    df["wickets"] = df.groupby(["match_id", "inning"])["is_wicket"].cumsum()
    df["ball_number"] = df["over"] * 6 + (df["ball"] - 1)
    return df


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df["batting_side_won"] = (df["batting_team"] == df["winner"]).astype(int)
    return df


def main() -> None:
    matches = pd.read_csv(RAW_DIR / "matches.csv")
    balls = pd.read_csv(RAW_DIR / "deliveries.csv")

    df = add_label(build_ball_frame(matches, balls))
    df.to_parquet(OUT_DIR / "balls_with_features.parquet", index=False)
    print(f"✅ Saved {len(df):,} rows ➜ {OUT_DIR/'balls_with_features.parquet'}")


if __name__ == "__main__":
    main()
