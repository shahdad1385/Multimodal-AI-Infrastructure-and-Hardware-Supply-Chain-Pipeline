import os
import pandas as pd
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data_samples')

TARGET_TICKERS = [
    "NVDA", "META", "GOOGL", "TSM", "VRT", "MOD", "SMCI",
    "AMD", "INTC", "AVGO", "QCOM", "MRVL", "AMAT", "MU",
    "LRCX", "KLAC", "ASML", "ARM", "NXPI", "ON",
]

START_DATE = "2018-01-01"


def download_and_filter():
    print("📥 Downloading HuggingFace financial-news dataset...")
    print("   Source: ashraq/financial-news (~143MB Parquet)\n")

    ds = load_dataset("ashraq/financial-news", split="train")
    df = ds.to_pandas()
    print(f"   Total rows: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}")

    print(f"\n🔍 Filtering to target tickers ({len(TARGET_TICKERS)} tickers)...")
    df["stock"] = df["stock"].astype(str).str.upper().str.strip()
    df = df[df["stock"].isin(TARGET_TICKERS)]
    print(f"   After ticker filter: {len(df)} rows")

    print(f"📅 Filtering to date >= {START_DATE}...")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df = df[df["date"] >= START_DATE]
    print(f"   After date filter: {len(df)} rows")

    df = df.rename(columns={
        "headline": "headline",
        "date": "date",
        "publisher": "source",
        "stock": "ticker",
    })

    if "url" not in df.columns:
        df["url"] = ""

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    cols = ["date", "headline", "source", "ticker", "url"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    df.sort_values("date", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "hf_financial_news.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n📁 Saved: {output_path} ({len(df)} rows)")

    print("\n📊 Ticker distribution:")
    print(df["ticker"].value_counts().to_string())

    print(f"\n📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"📰 Unique sources: {df['source'].nunique()}")

    return df


if __name__ == "__main__":
    download_and_filter()
