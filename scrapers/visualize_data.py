import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data_samples')
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')

TARGET_KEYWORDS = [
    "ai", "artificial intelligence", "nvidia", "amd", "intel", "tsmc",
    "semiconductor", "chip", "gpu", "data center", "datacenter",
    "cooling", "power", "grid", "energy", "infrastructure",
]


def load_data():
    path = os.path.join(OUTPUT_DIR, "ai_infrastructure_news.csv")
    if not os.path.exists(path):
        print(f"❌ Dataset not found at {path}. Run the crawler first.")
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    return df


def chart_publication_frequency(df):
    daily = df.groupby(df["date"].dt.date).size().reset_index(name="count")
    daily["date"] = pd.to_datetime(daily["date"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(daily["date"], daily["count"], color="#2196F3", alpha=0.8)
    ax.set_title("Daily Article Publication Frequency", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Articles")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "publication_frequency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {path}")


def chart_articles_by_source(df):
    counts = df["source"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Set2", len(counts))
    counts.plot.barh(ax=ax, color=colors)
    ax.set_title("Articles by Source", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Articles")
    ax.set_ylabel("")
    for i, v in enumerate(counts):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=10)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "articles_by_source.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {path}")


def chart_keyword_frequency(df):
    text_blob = " ".join(
        (str(row["headline"]) + " " + str(row.get("summary", "")))
        for _, row in df.iterrows()
    ).lower()

    freq = {}
    for kw in TARGET_KEYWORDS:
        count = len(re.findall(rf"\b{re.escape(kw)}\b", text_blob))
        if count > 0:
            freq[kw] = count

    if not freq:
        print("  ⚠ No keyword matches found — skipping keyword chart.")
        return

    freq_df = pd.DataFrame(list(freq.items()), columns=["keyword", "count"])
    freq_df.sort_values("count", ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(freq_df))
    ax.barh(freq_df["keyword"], freq_df["count"], color=colors)
    ax.set_title("Keyword Frequency in Headlines & Summaries", fontsize=14, fontweight="bold")
    ax.set_xlabel("Frequency")
    for i, v in enumerate(freq_df["count"]):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=10)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "keyword_frequency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {path}")


def chart_source_timeline(df):
    daily_source = df.groupby([df["date"].dt.date, "source"]).size().reset_index(name="count")
    daily_source["date"] = pd.to_datetime(daily_source["date"])

    pivot = daily_source.pivot_table(index="date", columns="source", values="count", fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    pivot.plot.area(ax=ax, alpha=0.7, colormap="Set2")
    ax.set_title("Article Volume by Source Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Articles")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "source_timeline.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {path}")


def chart_headline_length(df):
    df = df.copy()
    df["headline_len"] = df["headline"].str.len()

    fig, ax = plt.subplots(figsize=(10, 5))
    for source in df["source"].unique():
        subset = df[df["source"] == source]
        ax.hist(subset["headline_len"], bins=20, alpha=0.5, label=source)
    ax.set_title("Headline Length Distribution by Source", fontsize=14, fontweight="bold")
    ax.set_xlabel("Character Length")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "headline_length_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {path}")


import re


def generate_all_charts():
    print("📊 Generating visualizations...\n")
    df = load_data()
    if df is None or df.empty:
        return

    os.makedirs(CHARTS_DIR, exist_ok=True)

    chart_publication_frequency(df)
    chart_articles_by_source(df)
    chart_keyword_frequency(df)
    chart_source_timeline(df)
    chart_headline_length(df)

    print(f"\n✅ All charts saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    generate_all_charts()
