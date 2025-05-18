"""
Feature-engineering utilities

• add_features(df)       → basic cricket match-state features
• design_matrix(df, cols)→ one-hot encode + clean + deduplicate column names
"""

from __future__ import annotations
import re
from typing import List, Sequence

import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Basic derived features
# --------------------------------------------------------------------------- #
PHASE_CUTS   = pd.IntervalIndex.from_tuples([(-1, 5), (5, 15), (15, 20)])
PHASE_LABELS = ["powerplay", "middle", "death"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add core numeric / categorical match-state features."""
    df = df.copy()

    df["runs_remaining"]  = (df["target_runs"].fillna(0) - df["runs_cum"]).clip(lower=0)
    df["balls_remaining"] = (120 - df["ball_number"]).clip(lower=0)
    df["wickets_in_hand"] = 10 - df["wickets"]

    df["current_rr"]  = df["runs_cum"] / ((df["ball_number"] + 1) / 6)
    df["required_rr"] = df["runs_remaining"] / (df["balls_remaining"] / 6).replace(0, 1)

    df["phase"] = pd.cut(df["over"], PHASE_CUTS, labels=PHASE_LABELS)

    return df


FEATURE_COLS: List[str] = [
    "runs_remaining",
    "balls_remaining",
    "wickets_in_hand",
    "current_rr",
    "required_rr",
    "phase",
]
LABEL_COL = "batting_side_won"

# --------------------------------------------------------------------------- #
# 2. Safe design-matrix helper
# --------------------------------------------------------------------------- #
_bad_json = re.compile(r"[^A-Za-z0-9_]+")  # we allow only letters, digits, underscore


def _clean_strings(index_like: Sequence[str]) -> pd.Index:
    """
    Replace disallowed chars AND ensure uniqueness (LightGBM needs that too).
    """
    cleaned = [re.sub(_bad_json, "_", s) for s in index_like]

    # Deduplicate while preserving order
    out = []
    seen = {}
    for col in cleaned:
        if col not in seen:
            seen[col] = 0
            out.append(col)
        else:
            seen[col] += 1
            out.append(f"{col}_{seen[col]}")
    return pd.Index(out)


def design_matrix(df: pd.DataFrame, existing_cols: Sequence[str] | None = None) -> pd.DataFrame:
    """
    One-hot encode categorical features, clean column names, and (optionally)
    align to an existing training column order.
    """
    X = pd.get_dummies(df[FEATURE_COLS])
    X.columns = _clean_strings(X.columns)

    if existing_cols is not None:
        X = X.reindex(columns=existing_cols, fill_value=0)

    return X
