import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database_connection import get_engine, get_session

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "data_samples", "cleaned")


def load_table(query, engine):
    return pd.read_sql(query, engine)


def clean_stock_prices(engine):
    df = load_table("SELECT * FROM stock_prices", engine)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "ticker"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df.groupby("ticker")[numeric_cols].transform(lambda x: x.ffill().bfill())
    df = df.dropna(subset=["close"])

    df["was_missing_volume"] = (df["volume"].isna()).astype(int)
    return df


def clean_market_indicators(engine):
    df = load_table("SELECT * FROM market_indicators", engine)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "indicator"])
    df = df.sort_values(["indicator", "date"]).reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df.groupby("indicator")[numeric_cols].transform(lambda x: x.ffill().bfill())
    df = df.dropna(subset=["close"])
    return df


def clean_news_articles(engine):
    df = load_table("SELECT * FROM news_articles", engine)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "headline"])
    df = df.drop_duplicates(subset=["headline", "date"])
    df["headline"] = df["headline"].str.strip()
    df["headline_len"] = df["headline"].str.len()
    df["source"] = df["source"].fillna("unknown")
    return df


def clean_hf_news(engine):
    df = load_table("SELECT * FROM hf_news", engine)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "headline"])
    df = df.drop_duplicates(subset=["headline", "date", "ticker"])
    df["headline"] = df["headline"].str.strip()
    df["headline_len"] = df["headline"].str.len()
    df["source"] = df["source"].fillna("unknown")
    return df


def clean_companies(engine):
    df = load_table("SELECT * FROM companies", engine)
    return df


def run_preprocessing():
    print("🧹 Running preprocessing...\n")
    engine = get_engine()

    os.makedirs(CLEANED_DIR, exist_ok=True)

    print("  Cleaning stock_prices...")
    stock = clean_stock_prices(engine)
    stock.to_csv(os.path.join(CLEANED_DIR, "stock_prices_clean.csv"), index=False)
    print(f"    ✅ {len(stock):,} rows")

    print("  Cleaning market_indicators...")
    indicators = clean_market_indicators(engine)
    indicators.to_csv(os.path.join(CLEANED_DIR, "market_indicators_clean.csv"), index=False)
    print(f"    ✅ {len(indicators):,} rows")

    print("  Cleaning news_articles...")
    news = clean_news_articles(engine)
    news.to_csv(os.path.join(CLEANED_DIR, "news_articles_clean.csv"), index=False)
    print(f"    ✅ {len(news):,} rows")

    print("  Cleaning hf_news...")
    hf = clean_hf_news(engine)
    hf.to_csv(os.path.join(CLEANED_DIR, "hf_news_clean.csv"), index=False)
    print(f"    ✅ {len(hf):,} rows")

    print("  Loading companies...")
    companies = clean_companies(engine)
    companies.to_csv(os.path.join(CLEANED_DIR, "companies_clean.csv"), index=False)
    print(f"    ✅ {len(companies):,} rows")

    print(f"\n{'='*50}")
    print(f"✅ Preprocessing complete!")
    print(f"📂 Cleaned data saved to: {CLEANED_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_preprocessing()
