import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database_connection import get_engine

engine = get_engine()


def add_columns_if_missing(table, columns):
    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()}
        for col_name, col_type in columns:
            if col_name not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"))
                print(f"    + {col_name} ({col_type})")
        conn.commit()


def add_stock_features():
    print("📈 Adding features to stock_prices...")

    features = [
        ("daily_return", "REAL"), ("log_return", "REAL"),
        ("price_range", "REAL"), ("gap", "REAL"),
        ("volume_change", "REAL"), ("volume_ma_7", "REAL"), ("volume_ratio", "REAL"),
        ("return_lag_1d", "REAL"), ("return_lag_2d", "REAL"), ("return_lag_3d", "REAL"),
        ("return_lag_5d", "REAL"), ("return_lag_10d", "REAL"),
        ("volatility_7d", "REAL"), ("volatility_14d", "REAL"), ("volatility_30d", "REAL"),
        ("close_ma_7", "REAL"), ("close_ma_14", "REAL"), ("close_ma_30", "REAL"),
        ("close_ma_ratio_7", "REAL"), ("close_ma_ratio_14", "REAL"), ("close_ma_ratio_30", "REAL"),
        ("ewm_return_12", "REAL"), ("ewm_return_26", "REAL"),
        ("high_low_ratio", "REAL"), ("close_open_ratio", "REAL"),
        ("year", "INTEGER"), ("quarter", "INTEGER"), ("month", "INTEGER"),
        ("week_of_year", "INTEGER"), ("day_of_week", "INTEGER"),
        ("is_weekend", "INTEGER"), ("is_month_start", "INTEGER"),
        ("is_month_end", "INTEGER"), ("is_quarter_end", "INTEGER"),
        ("day_name", "TEXT"), ("month_name", "TEXT"), ("week_number", "INTEGER"),
        ("is_holiday", "INTEGER"),
        ("day_name_monday", "INTEGER"), ("day_name_tuesday", "INTEGER"),
        ("day_name_wednesday", "INTEGER"), ("day_name_thursday", "INTEGER"),
        ("day_name_friday", "INTEGER"), ("day_name_saturday", "INTEGER"), ("day_name_sunday", "INTEGER"),
        ("month_name_january", "INTEGER"), ("month_name_february", "INTEGER"),
        ("month_name_march", "INTEGER"), ("month_name_april", "INTEGER"),
        ("month_name_may", "INTEGER"), ("month_name_june", "INTEGER"),
        ("month_name_july", "INTEGER"), ("month_name_august", "INTEGER"),
        ("month_name_september", "INTEGER"), ("month_name_october", "INTEGER"),
        ("month_name_november", "INTEGER"), ("month_name_december", "INTEGER"),
        ("quarter_1", "INTEGER"), ("quarter_2", "INTEGER"),
        ("quarter_3", "INTEGER"), ("quarter_4", "INTEGER"),
    ]
    add_columns_if_missing("stock_prices", features)

    df = pd.read_sql("SELECT id, date, open, high, low, close, volume FROM stock_prices ORDER BY ticker, date", engine)
    df["date"] = pd.to_datetime(df["date"])

    updates = pd.DataFrame(index=df.index)

    updates["daily_return"] = df["close"].pct_change()
    updates["log_return"] = np.log(df["close"] / df["close"].shift(1))
    updates["price_range"] = (df["high"] - df["low"]) / df["close"]
    updates["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    updates["volume_change"] = df["volume"].pct_change()
    updates["volume_ma_7"] = df["volume"].shift(1).rolling(7).mean()
    updates["volume_ratio"] = df["volume"] / (updates["volume_ma_7"] + 1)

    for lag in [1, 2, 3, 5, 10]:
        updates[f"return_lag_{lag}d"] = updates["daily_return"].shift(lag)

    for w in [7, 14, 30]:
        shifted = updates["daily_return"].shift(1)
        updates[f"volatility_{w}d"] = shifted.rolling(w).std()
        updates[f"close_ma_{w}"] = df["close"].shift(1).rolling(w).mean()
        updates[f"close_ma_ratio_{w}"] = df["close"] / (updates[f"close_ma_{w}"] + 1e-8)

    shifted = updates["daily_return"].shift(1)
    updates["ewm_return_12"] = shifted.ewm(span=12).mean()
    updates["ewm_return_26"] = shifted.ewm(span=26).mean()

    updates["high_low_ratio"] = df["high"] / (df["low"] + 1e-8)
    updates["close_open_ratio"] = df["close"] / (df["open"] + 1e-8)

    d = df["date"]
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                   5: 'May', 6: 'June', 7: 'July', 8: 'August',
                   9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    updates["year"] = d.dt.year
    updates["quarter"] = d.dt.quarter
    updates["month"] = d.dt.month
    updates["week_of_year"] = d.dt.isocalendar().week.astype(int)
    updates["day_of_week"] = d.dt.dayofweek
    updates["is_weekend"] = d.dt.dayofweek.isin([5, 6]).astype(int)
    updates["is_month_start"] = d.dt.is_month_start.astype(int)
    updates["is_month_end"] = d.dt.is_month_end.astype(int)
    updates["is_quarter_end"] = d.dt.is_quarter_end.astype(int)
    updates["day_name"] = d.dt.dayofweek.map(day_names)
    updates["month_name"] = d.dt.month.map(month_names)
    updates["week_number"] = d.dt.isocalendar().week.astype(int)

    from sklearn.preprocessing import OneHotEncoder as OHE
    cat_df = pd.DataFrame({"day_name": updates["day_name"], "month_name": updates["month_name"],
                            "quarter": df["date"].dt.quarter if "date" in df.columns else d.dt.quarter})
    enc = OHE(sparse_output=False, handle_unknown="ignore")
    enc.fit(cat_df.fillna("unknown"))
    encoded = enc.transform(cat_df.fillna("unknown"))
    for i, col in enumerate(enc.get_feature_names_out()):
        updates[col.replace(" ", "_").lower()] = encoded[:, i].astype(int)

    import holidays as hol
    us_holidays = hol.US(years=range(2018, 2027))
    updates["is_holiday"] = df["date"].apply(lambda x: 1 if pd.to_datetime(x).date() in us_holidays else 0) if "date" in df.columns else 0

    updates = updates.where(pd.notnull(updates), None)

    with engine.connect() as conn:
        for idx in df.index:
            row_id = int(df.loc[idx, "id"])
            sets = []
            vals = {}
            for col in updates.columns:
                v = updates.loc[idx, col]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    sets.append(f"{col} = :{col}")
                    vals[col] = float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
            if sets:
                vals["id"] = row_id
                conn.execute(text(f"UPDATE stock_prices SET {', '.join(sets)} WHERE id = :id"), vals)
        conn.commit()

    print(f"    ✅ Updated {len(df):,} rows")


def add_indicator_features():
    print("📊 Adding features to market_indicators...")

    features = [
        ("indicator_return", "REAL"), ("indicator_log_return", "REAL"),
        ("indicator_lag_1d", "REAL"), ("indicator_lag_5d", "REAL"), ("indicator_lag_10d", "REAL"),
        ("indicator_vol_7d", "REAL"),
        ("indicator_ma_7", "REAL"), ("indicator_ma_30", "REAL"), ("indicator_ma_ratio", "REAL"),
        ("indicator_ewm_12", "REAL"),
    ]
    add_columns_if_missing("market_indicators", features)

    df = pd.read_sql("SELECT id, date, close, indicator FROM market_indicators ORDER BY indicator, date", engine)
    df["date"] = pd.to_datetime(df["date"])

    updates = pd.DataFrame(index=df.index)

    updates["indicator_return"] = df.groupby("indicator")["close"].pct_change()

    updates["indicator_log_return"] = df.groupby("indicator").apply(
        lambda g: np.log(g["close"] / g["close"].shift(1)), include_groups=False
    ).reset_index(level=0, drop=True)

    for lag in [1, 5, 10]:
        updates[f"indicator_lag_{lag}d"] = df.groupby("indicator")["close"].shift(lag)

    shifted = updates.groupby(df["indicator"])["indicator_return"].shift(1)
    updates["indicator_vol_7d"] = shifted.rolling(7).mean()

    updates["indicator_ma_7"] = df.groupby("indicator")["close"].shift(1).rolling(7).mean()
    updates["indicator_ma_30"] = df.groupby("indicator")["close"].shift(1).rolling(30).mean()
    updates["indicator_ma_ratio"] = df["close"] / (updates["indicator_ma_7"] + 1e-8)
    updates["indicator_ewm_12"] = shifted.ewm(span=12).mean()

    updates = updates.where(pd.notnull(updates), None)

    with engine.connect() as conn:
        for idx in df.index:
            row_id = int(df.loc[idx, "id"])
            sets = []
            vals = {}
            for col in updates.columns:
                v = updates.loc[idx, col]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    sets.append(f"{col} = :{col}")
                    vals[col] = float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
            if sets:
                vals["id"] = row_id
                conn.execute(text(f"UPDATE market_indicators SET {', '.join(sets)} WHERE id = :id"), vals)
        conn.commit()

    print(f"    ✅ Updated {len(df):,} rows")


def add_news_features():
    print("📰 Adding features to news_articles...")

    features = [
        ("headline_len", "INTEGER"),
        ("daily_article_count", "INTEGER"),
        ("daily_source_count", "INTEGER"),
    ]
    add_columns_if_missing("news_articles", features)

    df = pd.read_sql("SELECT id, date, headline, source FROM news_articles", engine)
    df["date"] = pd.to_datetime(df["date"])
    df["headline_len"] = df["headline"].str.len()

    daily_counts = df.groupby("date").size().rename("daily_article_count")
    daily_sources = df.groupby("date")["source"].nunique().rename("daily_source_count")
    df = df.join(daily_counts, on="date").join(daily_sources, on="date")

    with engine.connect() as conn:
        for idx in df.index:
            row_id = int(df.loc[idx, "id"])
            conn.execute(text(
                "UPDATE news_articles SET headline_len = :hl, daily_article_count = :dac, daily_source_count = :dsc WHERE id = :id"
            ), {"hl": int(df.loc[idx, "headline_len"]), "dac": int(df.loc[idx, "daily_article_count"]),
                "dsc": int(df.loc[idx, "daily_source_count"]), "id": row_id})
        conn.commit()

    print(f"    ✅ Updated {len(df):,} rows")


def add_hf_features():
    print("📚 Adding features to hf_news...")

    features = [
        ("headline_len", "INTEGER"),
        ("daily_article_count", "INTEGER"),
    ]
    add_columns_if_missing("hf_news", features)

    df = pd.read_sql("SELECT id, date, headline FROM hf_news", engine)
    df["date"] = pd.to_datetime(df["date"])
    df["headline_len"] = df["headline"].str.len()

    daily_counts = df.groupby("date").size().rename("daily_article_count")
    df = df.join(daily_counts, on="date")

    with engine.connect() as conn:
        for idx in df.index:
            row_id = int(df.loc[idx, "id"])
            conn.execute(text(
                "UPDATE hf_news SET headline_len = :hl, daily_article_count = :dac WHERE id = :id"
            ), {"hl": int(df.loc[idx, "headline_len"]), "dac": int(df.loc[idx, "daily_article_count"]), "id": row_id})
        conn.commit()

    print(f"    ✅ Updated {len(df):,} rows")


if __name__ == "__main__":
    print("🔧 Adding features to database tables...\n")
    add_stock_features()
    add_indicator_features()
    add_news_features()
    add_hf_features()
    print(f"\n{'='*50}")
    print("✅ All features added to database!")
    print("{'='*50}")
