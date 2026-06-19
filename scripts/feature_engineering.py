import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "data_samples", "cleaned")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_samples")


# ---------------------------------------------------------------------------
# Stock price features (per ticker)
# ---------------------------------------------------------------------------
def engineer_stock_features(df):
    print("  Building stock price features...")
    result = []

    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("date").copy()

        g["daily_return"] = g["close"].pct_change()
        g["log_return"] = np.log(g["close"] / g["close"].shift(1))

        g["price_range"] = (g["high"] - g["low"]) / g["close"]
        g["gap"] = (g["open"] - g["close"].shift(1)) / g["close"].shift(1)

        g["volume_change"] = g["volume"].pct_change()
        g["volume_ma_7"] = g["volume"].shift(1).rolling(7).mean()
        g["volume_ratio"] = g["volume"] / (g["volume_ma_7"] + 1)

        for lag in [1, 2, 3, 5, 10]:
            g[f"return_lag_{lag}d"] = g["daily_return"].shift(lag)

        for window in [7, 14, 30]:
            shifted = g["daily_return"].shift(1)
            g[f"volatility_{window}d"] = shifted.rolling(window).std()
            g[f"close_ma_{window}"] = g["close"].shift(1).rolling(window).mean()
            g[f"close_ma_ratio_{window}"] = g["close"] / (g[f"close_ma_{window}"] + 1e-8)

        shifted = g["daily_return"].shift(1)
        g["ewm_return_12"] = shifted.ewm(span=12).mean()
        g["ewm_return_26"] = shifted.ewm(span=26).mean()

        g["high_low_ratio"] = g["high"] / (g["low"] + 1e-8)
        g["close_open_ratio"] = g["close"] / (g["open"] + 1e-8)

        result.append(g)

    out = pd.concat(result, ignore_index=True)
    print(f"    ✅ {len(out):,} rows, {len(out.columns)} features")
    return out


# ---------------------------------------------------------------------------
# Market indicator features (per indicator)
# ---------------------------------------------------------------------------
def engineer_indicator_features(df):
    print("  Building market indicator features...")
    result = []

    for indicator, group in df.groupby("indicator"):
        g = group.sort_values("date").copy()
        safe_name = indicator.lower().replace(" ", "_").replace("-", "_")

        g[f"{safe_name}_return"] = g["close"].pct_change()
        g[f"{safe_name}_log_return"] = np.log(g["close"] / g["close"].shift(1))

        for lag in [1, 5, 10]:
            g[f"{safe_name}_lag_{lag}d"] = g["close"].shift(lag)

        shifted_ret = g[f"{safe_name}_return"].shift(1)
        g[f"{safe_name}_vol_7d"] = shifted_ret.rolling(7).std()
        g[f"{safe_name}_ma_7"] = g["close"].shift(1).rolling(7).mean()
        g[f"{safe_name}_ma_30"] = g["close"].shift(1).rolling(30).mean()
        g[f"{safe_name}_ma_ratio"] = g["close"] / (g[f"{safe_name}_ma_7"] + 1e-8)

        result.append(g)

    out = pd.concat(result, ignore_index=True)
    pivot_cols = [c for c in out.columns if c not in ["date", "indicator", "ticker",
                                                        "open", "high", "low", "close", "volume"]]
    pivoted = out.pivot_table(index="date", columns="indicator", values=pivot_cols, aggfunc="first")
    pivoted.columns = [f"{col}_{ind}" for col, ind in pivoted.columns]
    pivoted = pivoted.reset_index()

    print(f"    ✅ {len(pivoted):,} rows, {len(pivoted.columns)} features")
    return pivoted


# ---------------------------------------------------------------------------
# News features (per day)
# ---------------------------------------------------------------------------
def engineer_news_features(news_df, hf_df):
    print("  Building news features...")

    all_news = pd.concat([
        news_df[["date", "headline", "source", "headline_len"]].assign(dataset="live"),
        hf_df[["date", "headline", "source", "headline_len"]].assign(dataset="hf"),
    ], ignore_index=True)
    all_news["date"] = pd.to_datetime(all_news["date"])

    daily = all_news.groupby("date").agg(
        news_count=("headline", "count"),
        avg_headline_len=("headline_len", "mean"),
        max_headline_len=("headline_len", "max"),
        unique_sources=("source", "nunique"),
    ).reset_index()

    source_counts = all_news.groupby(["date", "source"]).size().unstack(fill_value=0)
    source_counts.columns = [f"news_src_{c.replace(' ', '_').lower()}" for c in source_counts.columns]
    source_counts = source_counts.reset_index()

    live_only = all_news[all_news["dataset"] == "live"]
    daily_live = live_only.groupby("date").size().reset_index(name="news_count_live")

    hf_only = all_news[all_news["dataset"] == "hf"]
    daily_hf = hf_only.groupby("date").size().reset_index(name="news_count_hf")

    result = daily.merge(source_counts, on="date", how="left")
    result = result.merge(daily_live, on="date", how="left")
    result = result.merge(daily_hf, on="date", how="left")
    result = result.fillna(0)

    print(f"    ✅ {len(result):,} days, {len(result.columns)} features")
    return result


# ---------------------------------------------------------------------------
# Calendar / time features
# ---------------------------------------------------------------------------
def add_calendar_features(df):
    print("  Adding calendar features...")
    d = df["date"].copy() if "date" in df.columns else df.copy()

    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                  4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                   5: 'May', 6: 'June', 7: 'July', 8: 'August',
                   9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    cal = pd.DataFrame({
        "year": d.dt.year,
        "quarter": d.dt.quarter,
        "month": d.dt.month,
        "week_of_year": d.dt.isocalendar().week.astype(int),
        "day_of_week": d.dt.dayofweek,
        "is_weekend": d.dt.dayofweek.isin([5, 6]).astype(int),
        "is_month_start": d.dt.is_month_start.astype(int),
        "is_month_end": d.dt.is_month_end.astype(int),
        "is_quarter_end": d.dt.is_quarter_end.astype(int),
        "day_name": d.dt.dayofweek.map(day_names),
        "month_name": d.dt.month.map(month_names),
        "week_number": d.dt.isocalendar().week.astype(int),
    })

    df = pd.concat([df.reset_index(drop=True), cal.reset_index(drop=True)], axis=1)
    print(f"    ✅ Added {len(cal.columns)} calendar features")
    return df


# ---------------------------------------------------------------------------
# Build unified feature matrix
# ---------------------------------------------------------------------------
def build_feature_matrix():
    print("\n🔧 Building feature matrix...\n")

    print("Loading cleaned data...")
    stock = pd.read_csv(os.path.join(CLEANED_DIR, "stock_prices_clean.csv"), parse_dates=["date"])
    indicators = pd.read_csv(os.path.join(CLEANED_DIR, "market_indicators_clean.csv"), parse_dates=["date"])
    news = pd.read_csv(os.path.join(CLEANED_DIR, "news_articles_clean.csv"), parse_dates=["date"])
    hf = pd.read_csv(os.path.join(CLEANED_DIR, "hf_news_clean.csv"), parse_dates=["date"])

    stock_features = engineer_stock_features(stock)
    indicator_features = engineer_indicator_features(indicators)
    news_features = engineer_news_features(news, hf)

    print("  Merging all features...")
    matrix = stock_features.merge(indicator_features, on="date", how="left")
    matrix = matrix.merge(news_features, on="date", how="left")

    matrix = add_calendar_features(matrix)

    matrix = matrix.sort_values(["ticker", "date"]).reset_index(drop=True)

    numeric_cols = matrix.select_dtypes(include=[np.number]).columns
    matrix[numeric_cols] = matrix.groupby("ticker")[numeric_cols].transform(lambda x: x.ffill())
    matrix = matrix.fillna(0)

    print(f"\n  Dropping low-variance columns...")
    zero_var = [c for c in matrix.columns if matrix[c].nunique() <= 1]
    matrix = matrix.drop(columns=zero_var, errors="ignore")
    print(f"    Removed {len(zero_var)} constant columns")

    corr = matrix.select_dtypes(include=[np.number]).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
    matrix = matrix.drop(columns=high_corr, errors="ignore")
    print(f"    Removed {len(high_corr)} highly correlated features (>0.95)")

    output_path = os.path.join(OUTPUT_DIR, "feature_matrix.csv")
    matrix.to_csv(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"✅ Feature matrix built!")
    print(f"📊 Shape: {matrix.shape[0]:,} rows x {matrix.shape[1]} features")
    print(f"📂 Saved to: {output_path}")
    print(f"{'='*50}")

    print(f"\nFeature columns:")
    for i, col in enumerate(matrix.columns, 1):
        print(f"  {i:3d}. {col}")

    return matrix


if __name__ == "__main__":
    build_feature_matrix()
