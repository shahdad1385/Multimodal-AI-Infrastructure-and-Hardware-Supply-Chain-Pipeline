import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database_connection import get_engine

engine = get_engine()


def temporal_split(df, train_ratio=0.85):
    unique_dates = np.sort(df["date"].unique())
    n_dates = len(unique_dates)
    cutoff_idx = int(n_dates * train_ratio)
    cutoff_date = pd.Timestamp(unique_dates[cutoff_idx])

    train_df = df[df["date"] <= cutoff_date].copy()
    test_df = df[df["date"] > cutoff_date].copy()

    print(f"  Cutoff: {cutoff_date.date()}")
    print(f"  Train: {len(train_df):,} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  Test:  {len(test_df):,} rows ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    assert train_df["date"].max() <= test_df["date"].min(), "Temporal leakage detected!"
    print(f"  No temporal leakage")

    return train_df, test_df, cutoff_date


def get_predictor_cols(train_df, target_col, numeric_cols):
    exclude = {"id", target_col, "date"}
    predictors = [c for c in numeric_cols if c not in exclude and train_df[c].notna().sum() > 100]
    return predictors


def train_lr_model(train_df, col, predictor_cols):
    train_mask = train_df[col].notna()
    X_train = train_df.loc[train_mask, predictor_cols].fillna(0)
    y_train = train_df.loc[train_mask, col]

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def fill_nulls_with_lr(train_df, test_df, col, predictor_cols, model):
    null_mask_train = train_df[col].isna()
    if null_mask_train.any():
        X_pred = train_df.loc[null_mask_train, predictor_cols].fillna(0)
        train_df.loc[null_mask_train, col] = model.predict(X_pred)

    null_mask_test = test_df[col].isna()
    if null_mask_test.any():
        X_pred = test_df.loc[null_mask_test, predictor_cols].fillna(0)
        test_df.loc[null_mask_test, col] = model.predict(X_pred)

    return train_df, test_df


def handle_nulls(train_df, test_df, numeric_cols):
    print("\n  Filling null values (train-fit, both-transform)...")
    null_cols = [c for c in numeric_cols if train_df[c].isnull().any()]

    if not null_cols:
        print("    No nulls found")
        return train_df, test_df

    print(f"    {len(null_cols)} columns with nulls in train:")
    for col in null_cols:
        n = train_df[col].isnull().sum()
        pct = n / len(train_df) * 100
        print(f"      {col}: {n} ({pct:.1f}%)")

    drop_cols = [c for c in null_cols if train_df[c].isnull().mean() > 0.50]
    if drop_cols:
        print(f"\n    Dropping {len(drop_cols)} columns (>50% nulls): {drop_cols}")
        train_df = train_df.drop(columns=drop_cols)
        test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
        null_cols = [c for c in null_cols if c not in drop_cols]

    for col in null_cols:
        predictors = get_predictor_cols(train_df, col, numeric_cols)
        predictors = [c for c in predictors if c != col]

        if len(predictors) >= 2 and train_df[col].notna().sum() >= 50:
            model = train_lr_model(train_df, col, predictors)
            train_df, test_df = fill_nulls_with_lr(train_df, test_df, col, predictors, model)
            remaining = train_df[col].isnull().sum()
            if remaining > 0:
                fallback_predictors = [c for c in predictors if train_df[c].notna().all() and test_df[c].notna().all()]
                if len(fallback_predictors) >= 2:
                    model2 = train_lr_model(train_df, col, fallback_predictors)
                    train_df, test_df = fill_nulls_with_lr(train_df, test_df, col, fallback_predictors, model2)
                    remaining = train_df[col].isnull().sum()
                if remaining > 0:
                    train_df[col] = train_df.groupby("ticker")[col].transform(
                        lambda x: x.interpolate(method="linear", limit_direction="both").ffill().bfill()
                    )
                    test_df[col] = test_df.groupby("ticker")[col].transform(
                        lambda x: x.interpolate(method="linear", limit_direction="both").ffill().bfill()
                    )
                    print(f"    {col}: LR ({len(predictors)} features) -> LR ({len(fallback_predictors)} features) -> interpolation")
                else:
                    print(f"    {col}: LR ({len(predictors)} features) -> LR ({len(fallback_predictors)} features)")
            else:
                print(f"    {col}: LR ({len(predictors)} features)")
        else:
            train_df[col] = train_df.groupby("ticker")[col].transform(
                lambda x: x.interpolate(method="linear", limit_direction="both").ffill().bfill()
            )
            test_df[col] = test_df.groupby("ticker")[col].transform(
                lambda x: x.interpolate(method="linear", limit_direction="both").ffill().bfill()
            )
            print(f"    {col}: interpolation (insufficient predictors)")

    remaining = train_df[numeric_cols].isnull().sum().sum()
    print(f"\n    Remaining nulls in train: {remaining}")
    assert remaining == 0, f"Still have {remaining} nulls in train!"

    return train_df, test_df


def normalize_features(train_df, test_df, numeric_cols):
    print("\n  Normalizing features (fit on train only)...")
    print("  - RobustScaler: price/volume/volatility (outlier-robust)")
    print("  - StandardScaler: ratios/returns (more normally distributed)")

    robust_cols = [
        "open", "high", "low", "close", "volume",
        "volatility_7d", "volatility_14d", "volatility_30d",
        "close_ma_7", "close_ma_14", "close_ma_30",
        "volume_ma_7", "volume_ratio", "volume_change",
        "high_low_ratio", "close_open_ratio",
    ]
    robust_cols = [c for c in robust_cols if c in train_df.columns]

    standard_cols = [
        "daily_return", "log_return", "price_range", "gap",
        "close_ma_ratio_7", "close_ma_ratio_14", "close_ma_ratio_30",
        "ewm_return_12", "ewm_return_26",
        "return_lag_1d", "return_lag_2d", "return_lag_3d",
        "return_lag_5d", "return_lag_10d",
    ]
    standard_cols = [c for c in standard_cols if c in train_df.columns]

    robust_scaler = RobustScaler()
    robust_scaler.fit(train_df[robust_cols])
    train_df[robust_cols] = robust_scaler.transform(train_df[robust_cols])
    test_df[robust_cols] = robust_scaler.transform(test_df[robust_cols])
    print(f"    RobustScaler applied to: {', '.join(robust_cols)}")

    standard_scaler = StandardScaler()
    standard_scaler.fit(train_df[standard_cols])
    train_df[standard_cols] = standard_scaler.transform(train_df[standard_cols])
    test_df[standard_cols] = standard_scaler.transform(test_df[standard_cols])
    print(f"    StandardScaler applied to: {', '.join(standard_cols)}")

    return train_df, test_df


def save_to_db(train_df, test_df):
    print("\n  Saving to DB...")
    with engine.connect() as conn:
        try:
            conn.execute(text("DROP TABLE IF EXISTS train_stock_prices"))
            conn.execute(text("DROP TABLE IF EXISTS test_stock_prices"))
            conn.commit()
        except Exception:
            pass

    train_df.to_sql("train_stock_prices", engine, if_exists="replace", index=False)
    test_df.to_sql("test_stock_prices", engine, if_exists="replace", index=False)

    with engine.connect() as conn:
        train_count = conn.execute(text("SELECT COUNT(*) FROM train_stock_prices")).scalar()
        test_count = conn.execute(text("SELECT COUNT(*) FROM test_stock_prices")).scalar()
    print(f"    train_stock_prices: {train_count:,} rows")
    print(f"    test_stock_prices: {test_count:,} rows")


def run_preprocessing():
    print("=" * 50)
    print("PREPROCESSING")
    print("=" * 50)

    print("\nLoading data from DB...")
    df = pd.read_sql("SELECT * FROM stock_prices ORDER BY ticker, date", engine, parse_dates=["date"])
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    print("\nStep 1: Temporal split (85/15)...")
    train_df, test_df, cutoff_date = temporal_split(df, train_ratio=0.85)

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "id"]

    print("\nStep 2: Fill null values (train-fit, both-transform)...")
    print("  Each column: Linear Regression on ALL available features (trained on train)")
    train_df, test_df = handle_nulls(train_df, test_df, numeric_cols)

    print("\nStep 3: Normalize features (fit on train only)...")
    train_df, test_df = normalize_features(train_df, test_df, numeric_cols)

    print("\nStep 4: Save to DB...")
    save_to_db(train_df, test_df)

    print(f"\n{'=' * 50}")
    print("Preprocessing complete!")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    run_preprocessing()
