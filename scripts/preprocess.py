import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database_connection import get_engine

engine = get_engine()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data_samples")


def handle_missing_values():
    print("🔧 Step 1: Handling missing values...")
    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM stock_prices ORDER BY ticker, date", engine)
        print(f"    Rows before: {len(df):,}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "id"]

        before_nulls = df[numeric_cols].isnull().sum().sum()

        for col in numeric_cols:
            df[col] = df.groupby("ticker")[col].transform(lambda x: x.ffill().bfill())

        df[numeric_cols] = df[numeric_cols].fillna(0)

        after_nulls = df[numeric_cols].isnull().sum().sum()
        print(f"    Missing values: {before_nulls} → {after_nulls}")

        for idx in df.index:
            sets = []
            vals = {}
            for col in numeric_cols:
                v = df.loc[idx, col]
                if pd.notna(v):
                    sets.append(f"{col} = :{col}")
                    vals[col] = float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
            if sets:
                vals["id"] = int(df.loc[idx, "id"])
                conn.execute(text(f"UPDATE stock_prices SET {', '.join(sets)} WHERE id = :id"), vals)
        conn.commit()

    print(f"    ✅ Missing values handled")


def normalize_features():
    print("\n🔧 Step 2: Normalizing numeric features...")
    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM stock_prices ORDER BY ticker, date", engine)

        scale_cols = [
            "open", "high", "low", "close", "volume",
            "daily_return", "log_return", "price_range", "gap",
            "volume_change", "volume_ma_7", "volume_ratio",
            "volatility_7d", "volatility_14d", "volatility_30d",
            "close_ma_7", "close_ma_14", "close_ma_30",
            "close_ma_ratio_7", "close_ma_ratio_14", "close_ma_ratio_30",
            "ewm_return_12", "ewm_return_26",
            "high_low_ratio", "close_open_ratio",
        ]

        scale_cols = [c for c in scale_cols if c in df.columns]

        try:
            conn.execute(text("ALTER TABLE stock_prices ADD COLUMN normalized_close REAL"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE stock_prices ADD COLUMN normalized_volume REAL"))
        except Exception:
            pass
        try:
            conn.execute(text("ALTER TABLE stock_prices ADD COLUMN normalized_volatility REAL"))
        except Exception:
            pass

        for ticker in df["ticker"].unique():
            mask = df["ticker"] == ticker
            ticker_data = df.loc[mask]

            if len(ticker_data) < 2:
                continue

            scaler = RobustScaler()

            close_scaled = scaler.fit_transform(ticker_data[["close"]]).flatten()
            volume_scaled = scaler.fit_transform(ticker_data[["volume"]]).flatten()
            vol_scaled = scaler.fit_transform(ticker_data[["volatility_7d"]].fillna(0)).flatten()

            for i, idx in enumerate(ticker_data.index):
                conn.execute(text(
                    "UPDATE stock_prices SET normalized_close = :nc, normalized_volume = :nv, normalized_volatility = :nvol WHERE id = :id"
                ), {"nc": float(close_scaled[i]), "nv": float(volume_scaled[i]),
                    "nvol": float(vol_scaled[i]), "id": int(df.loc[idx, "id"])})

        conn.commit()
    print(f"    ✅ RobustScaler applied to close, volume, volatility_7d (per ticker)")


def remove_correlated_features():
    print("\n🔧 Step 3: Removing highly correlated features...")
    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM stock_prices", engine)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["id"]]

        corr = df[numeric_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = [col for col in upper.columns if any(upper[col] > 0.95)]

        print(f"    Features with >0.95 correlation: {len(high_corr)}")
        for col in high_corr:
            print(f"      - {col}")

        drop_cols = [c for c in high_corr if c not in [
            "open", "high", "low", "close", "volume",
            "daily_return", "volatility_7d", "volume_ratio",
        ]]

        for col in drop_cols:
            try:
                conn.execute(text(f"ALTER TABLE stock_prices DROP COLUMN {col}"))
                print(f"      Dropped: {col}")
            except Exception:
                pass

        conn.commit()
    print(f"    ✅ Removed {len(drop_cols)} highly correlated features")


def tfidf_news_headlines():
    print("\n🔧 Step 4: TF-IDF vectorization of news headlines...")
    with engine.connect() as conn:
        df = pd.read_sql("SELECT id, headline FROM news_articles WHERE headline IS NOT NULL", engine)

        if df.empty:
            print("    ⚠ No headlines found, skipping.")
            return

        tfidf = TfidfVectorizer(max_features=20, stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(df["headline"])
        feature_names = [f"tfidf_{f.replace(' ', '_')}" for f in tfidf.get_feature_names_out()]

        print(f"    Top TF-IDF features: {[f.replace(' ', '_') for f in tfidf.get_feature_names_out()]}")

        for name in feature_names:
            try:
                conn.execute(text(f"ALTER TABLE news_articles ADD COLUMN {name} REAL"))
            except Exception:
                pass

        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        tfidf_df["id"] = df["id"].values

        for idx in tfidf_df.index:
            sets = []
            vals = {}
            for col in feature_names:
                v = float(tfidf_df.loc[idx, col])
                if v > 0:
                    sets.append(f"{col} = :{col}")
                    vals[col] = v
            if sets:
                vals["id"] = int(tfidf_df.loc[idx, "id"])
                conn.execute(text(f"UPDATE news_articles SET {', '.join(sets)} WHERE id = :id"), vals)
        conn.commit()

    print(f"    ✅ {len(feature_names)} TF-IDF features added to news_articles")


def run_preprocessing():
    print("="*50)
    print("SECTION 2: Comprehensive Data Preprocessing")
    print("="*50)

    handle_missing_values()
    normalize_features()
    remove_correlated_features()
    tfidf_news_headlines()

    print(f"\n{'='*50}")
    print("✅ All preprocessing complete!")
    print("{'='*50}")


if __name__ == "__main__":
    run_preprocessing()
