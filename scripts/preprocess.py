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


def compute_cutoff_date(df, train_ratio=0.85):
    unique_dates = np.sort(df["date"].unique())
    n_dates = len(unique_dates)
    cutoff_idx = int(n_dates * train_ratio)
    cutoff_date = pd.Timestamp(unique_dates[cutoff_idx])
    return cutoff_date


def apply_split(df, cutoff_date):
    train_df = df[df["date"] <= cutoff_date].copy()
    test_df = df[df["date"] > cutoff_date].copy()
    return train_df, test_df


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


def handle_nulls(train_df, test_df, numeric_cols, group_col=None):
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
        print(f"    Dropping {len(drop_cols)} columns (>50% nulls): {drop_cols}")
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
                    if group_col:
                        train_df[col] = train_df.groupby(group_col)[col].transform(
                            lambda x: x.interpolate(method="linear", limit_direction="both").ffill().bfill()
                        )
                        test_df[col] = test_df.groupby(group_col)[col].transform(
                            lambda x: x.interpolate(method="linear", limit_direction="both").ffill().bfill()
                        )
                    else:
                        train_df[col] = train_df[col].interpolate(method="linear", limit_direction="both").ffill().bfill()
                        test_df[col] = test_df[col].interpolate(method="linear", limit_direction="both").ffill().bfill()
                    print(f"    {col}: LR ({len(predictors)} feats) -> LR ({len(fallback_predictors)} feats) -> interpolation")
                else:
                    print(f"    {col}: LR ({len(predictors)} feats) -> LR ({len(fallback_predictors)} feats)")
            else:
                print(f"    {col}: LR ({len(predictors)} feats)")
        else:
            if group_col:
                train_df[col] = train_df.groupby(group_col)[col].transform(
                    lambda x: x.interpolate(method="linear", limit_direction="both").ffill().bfill()
                )
                test_df[col] = test_df.groupby(group_col)[col].transform(
                    lambda x: x.interpolate(method="linear", limit_direction="both").ffill().bfill()
                )
            else:
                train_df[col] = train_df[col].interpolate(method="linear", limit_direction="both").ffill().bfill()
                test_df[col] = test_df[col].interpolate(method="linear", limit_direction="both").ffill().bfill()
            print(f"    {col}: interpolation (insufficient predictors)")

    return train_df, test_df


def normalize_features(train_df, test_df, numeric_cols):
    robust_cols = [
        "open", "high", "low", "close", "volume",
        "volatility_7d", "volatility_14d", "volatility_30d",
        "close_ma_7", "close_ma_14", "close_ma_30",
        "volume_ma_7", "volume_ratio", "volume_change",
        "high_low_ratio", "close_open_ratio",
        "indicator_vol_7d", "indicator_ma_7", "indicator_ma_30",
    ]
    robust_cols = [c for c in robust_cols if c in train_df.columns]

    standard_cols = [
        "daily_return", "log_return", "price_range", "gap",
        "close_ma_ratio_7", "close_ma_ratio_14", "close_ma_ratio_30",
        "ewm_return_12", "ewm_return_26",
        "return_lag_1d", "return_lag_2d", "return_lag_3d",
        "return_lag_5d", "return_lag_10d",
        "indicator_return", "indicator_log_return",
        "indicator_lag_1d", "indicator_lag_5d", "indicator_lag_10d",
        "indicator_ma_ratio", "indicator_ewm_12",
        "headline_len", "daily_article_count", "daily_source_count",
    ]
    standard_cols = [c for c in standard_cols if c in train_df.columns]

    applied = []
    if robust_cols:
        robust_scaler = RobustScaler()
        robust_scaler.fit(train_df[robust_cols])
        train_df[robust_cols] = robust_scaler.transform(train_df[robust_cols])
        test_df[robust_cols] = robust_scaler.transform(test_df[robust_cols])
        applied.append(f"RobustScaler: {', '.join(robust_cols)}")

    if standard_cols:
        standard_scaler = StandardScaler()
        standard_scaler.fit(train_df[standard_cols])
        train_df[standard_cols] = standard_scaler.transform(train_df[standard_cols])
        test_df[standard_cols] = standard_scaler.transform(test_df[standard_cols])
        applied.append(f"StandardScaler: {', '.join(standard_cols)}")

    for line in applied:
        print(f"    {line}")

    return train_df, test_df


def save_split(train_df, test_df, table_name):
    train_table = f"train_{table_name}"
    test_table = f"test_{table_name}"
    with engine.connect() as conn:
        try:
            conn.execute(text(f"DROP TABLE IF EXISTS {train_table}"))
            conn.execute(text(f"DROP TABLE IF EXISTS {test_table}"))
            conn.commit()
        except Exception:
            pass

    train_df.to_sql(train_table, engine, if_exists="replace", index=False)
    test_df.to_sql(test_table, engine, if_exists="replace", index=False)

    with engine.connect() as conn:
        tc = conn.execute(text(f"SELECT COUNT(*) FROM {train_table}")).scalar()
        tsc = conn.execute(text(f"SELECT COUNT(*) FROM {test_table}")).scalar()
    print(f"    {train_table}: {tc:,} rows | {test_table}: {tsc:,} rows")


def preprocess_stock_prices(cutoff_date):
    print("\n--- stock_prices ---")
    df = pd.read_sql("SELECT * FROM stock_prices ORDER BY ticker, date", engine, parse_dates=["date"])
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols")

    train_df, test_df = apply_split(df, cutoff_date)
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "id"]

    print("  Null handling:")
    train_df, test_df = handle_nulls(train_df, test_df, numeric_cols, group_col="ticker")

    print("  Normalizing:")
    train_df, test_df = normalize_features(train_df, test_df, numeric_cols)

    save_split(train_df, test_df, "stock_prices")


def preprocess_market_indicators(cutoff_date):
    print("\n--- market_indicators ---")
    df = pd.read_sql("SELECT * FROM market_indicators ORDER BY indicator, date", engine, parse_dates=["date"])
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols")

    train_df, test_df = apply_split(df, cutoff_date)
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "id"]

    print("  Null handling:")
    train_df, test_df = handle_nulls(train_df, test_df, numeric_cols, group_col="indicator")

    print("  Normalizing:")
    train_df, test_df = normalize_features(train_df, test_df, numeric_cols)

    save_split(train_df, test_df, "market_indicators")


def preprocess_news_articles(cutoff_date):
    print("\n--- news_articles ---")
    df = pd.read_sql("SELECT * FROM news_articles ORDER BY date", engine, parse_dates=["date"])
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols")

    train_df, test_df = apply_split(df, cutoff_date)
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "id"]

    print("  Null handling:")
    train_df, test_df = handle_nulls(train_df, test_df, numeric_cols)

    print("  Normalizing:")
    train_df, test_df = normalize_features(train_df, test_df, numeric_cols)

    save_split(train_df, test_df, "news_articles")


def preprocess_hf_news(cutoff_date):
    print("\n--- hf_news ---")
    df = pd.read_sql("SELECT * FROM hf_news ORDER BY date", engine, parse_dates=["date"])
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols")

    train_df, test_df = apply_split(df, cutoff_date)
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "id"]

    print("  Null handling:")
    train_df, test_df = handle_nulls(train_df, test_df, numeric_cols)

    print("  Normalizing:")
    train_df, test_df = normalize_features(train_df, test_df, numeric_cols)

    save_split(train_df, test_df, "hf_news")


def run_preprocessing():
    print("=" * 50)
    print("PREPROCESSING — ALL TABLES")
    print("=" * 50)

    print("\nStep 1: Compute cutoff from stock_prices...")
    stock_df = pd.read_sql("SELECT date FROM stock_prices", engine, parse_dates=["date"])
    cutoff_date = compute_cutoff_date(stock_df, train_ratio=0.85)
    print(f"  Cutoff: {cutoff_date.date()}")
    print(f"  (All tables will use this same cutoff)")

    preprocess_stock_prices(cutoff_date)
    preprocess_market_indicators(cutoff_date)
    preprocess_news_articles(cutoff_date)
    preprocess_hf_news(cutoff_date)

    print(f"\n{'=' * 50}")
    print("Preprocessing complete!")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    run_preprocessing()
