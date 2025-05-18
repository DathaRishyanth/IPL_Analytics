# """
# Train a LightGBM win-probability model and save artefacts + metrics.
# Usage:  python src/train.py
# """

# from pathlib import Path
# import joblib
# import lightgbm as lgb
# import pandas as pd
# from sklearn.metrics import roc_auc_score, brier_score_loss
# from sklearn.model_selection import train_test_split

# from features import add_features, FEATURE_COLS, LABEL_COL


# # --------------------------------------------------------------------------- #
# # Paths
# # --------------------------------------------------------------------------- #
# DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "balls_with_features.parquet"
# MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
# MODEL_DIR.mkdir(exist_ok=True, parents=True)


# # --------------------------------------------------------------------------- #
# # Training pipeline
# # --------------------------------------------------------------------------- #
# def main() -> None:
#     # 1. Load + engineer ----------------------------------------------------- #
#     df = pd.read_parquet(DATA_PATH)
#     df = add_features(df)

#     # 2. Match-level split to avoid leakage --------------------------------- #
#     train_ids, test_ids = train_test_split(
#         df["match_id"].unique(), test_size=0.20, random_state=42
#     )
#     train_df = df[df["match_id"].isin(train_ids)]
#     test_df  = df[df["match_id"].isin(test_ids)]

#     # 3. One-hot encode categoricals ---------------------------------------- #
#     X_train = pd.get_dummies(train_df[FEATURE_COLS])
#     y_train = train_df[LABEL_COL]

#     X_test  = (
#         pd.get_dummies(test_df[FEATURE_COLS])
#         .reindex(columns=X_train.columns, fill_value=0)
#     )
#     y_test = test_df[LABEL_COL]

#     lgb_train = lgb.Dataset(X_train, label=y_train)
#     lgb_eval  = lgb.Dataset(X_test,  label=y_test, reference=lgb_train)

#     # 4. Hyper-parameters ---------------------------------------------------- #
#     params = dict(
#         objective="binary",
#         metric="binary_logloss",
#         learning_rate=0.05,
#         num_leaves=64,
#         feature_fraction=0.80,
#         bagging_fraction=0.80,
#         bagging_freq=5,
#         seed=42,
#         verbose=-1,           # silence core LightGBM info lines
#     )

#     # 5. Train with early stopping (LightGBM ≥ 4.x uses callbacks) ---------- #
#     model = lgb.train(
#         params=params,
#         train_set=lgb_train,
#         num_boost_round=800,
#         valid_sets=[lgb_train, lgb_eval],
#         callbacks=[
#             lgb.log_evaluation(period=50),
#             lgb.early_stopping(stopping_rounds=50, verbose=False),
#         ],
#     )

#     # 6. Evaluate ----------------------------------------------------------- #
#     y_pred = model.predict(X_test, num_iteration=model.best_iteration)
#     print(f"ROC-AUC : {roc_auc_score(y_test, y_pred):.3f}")
#     print(f"Brier   : {brier_score_loss(y_test, y_pred):.4f}")

#     # 7. Persist model + column order -------------------------------------- #
#     artefact_path = MODEL_DIR / "lgbm_winprob.joblib"
#     joblib.dump({"model": model, "columns": X_train.columns.tolist()}, artefact_path)
#     print(f"💾  Model saved to {artefact_path}")


# # --------------------------------------------------------------------------- #
# if __name__ == "__main__":
#     main()

from pathlib import Path
import joblib, lightgbm as lgb, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

from features import add_features, design_matrix, LABEL_COL

DATA_PATH = Path(__file__).resolve().parents[1] / "data/processed/balls_with_features.parquet"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df = add_features(df)

    train_ids, test_ids = train_test_split(df["match_id"].unique(), test_size=0.2, random_state=42)
    train_df, test_df = df[df["match_id"].isin(train_ids)], df[df["match_id"].isin(test_ids)]

    X_train = design_matrix(train_df)
    y_train = train_df[LABEL_COL]
    X_test  = design_matrix(test_df, existing_cols=X_train.columns)
    y_test  = test_df[LABEL_COL]

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval  = lgb.Dataset(X_test,  label=y_test, reference=lgb_train)

    params = dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.05,
        num_leaves=64,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        seed=42,
        verbose=-1,
    )

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=800,
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=50, verbose=False),
        ],
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    print(f"ROC-AUC : {roc_auc_score(y_test, y_pred):.3f}")
    print(f"Brier   : {brier_score_loss(y_test, y_pred):.4f}")

    joblib.dump({"model": model, "columns": X_train.columns.tolist()},
                MODEL_DIR / "lgbm_winprob.joblib")
    print("💾  Model saved ➜ models/lgbm_winprob.joblib")


if __name__ == "__main__":
    main()
