# Multimodal AI Infrastructure & Hardware Supply Chain Pipeline

### Phase 1: High-Performance Alternative Data Acquisition Engine

### 📋 Executive Overview

Predicting market valuation shifts for Big Tech AI market leaders requires data beyond traditional daily closing prices. This project implements a robust, multimodal ingestion pipeline designed to align standard financial time-series indicators with structured, alternative textual signals scraped from critical hardware supply chain and grid infrastructure networks.

The core objective is to map lag-correlations between physical constraints—such as grid capacity, high-velocity liquid cooling breakthroughs, and chip-foundry expansions—and the downstream valuation fluctuations of market cap leaders.

### 🛠️ System Architecture & Data Flow

```text
                   ┌───────────────────────────────┐
                   │    Target Infrastructure      │
                   └───────────────┬───────────────┘
                                   │
          ┌────────────────────────┴────────────────────────┐
          ▼                                                 ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│       Yahoo Finance Web API     │   │   Multi-Source News Collector   │
├─────────────────────────────────┤   ├─────────────────────────────────┤
│ • yfinance Library Ingestion    │   │ • TechCrunch AI (RSS)          │
│ • Maximum Time Horizon (2018+)  │   │ • Semiconductor Engineering    │
│ • Standardized Numerical CSV    │   │ • The Register (Atom feed)     │
└─────────────────────────────────┘   │ • Tom's Hardware (RSS)         │
                                      │ • SemiAnalysis (RSS)           │
                                      │ • DataCenterDynamics (HTML)    │
                                      │ • Keyword Matrix Filtering     │
                                      │ • Cross-source Deduplication   │
                                      └─────────────────────────────────┘
                         \                           /
                          \─────────────┬───────────/
                                        ▼
                           ┌─────────────────────────────┐
                           │    /data_samples Output     │
                           ├─────────────────────────────┤
                           │  • Market Time Series Data  │
                           │  • Filtered AI News Dataset │
                           │  • Raw All-News Dataset     │
                           │  • Visualization Charts     │
                           └─────────────────────────────┘
```

### 1. Financial Ingestion Pipeline (`scrapers/AI_Stock.py`)

Scope: Captures full-depth, historical trading indicators (Open, High, Low, Close, Volume) since 2018-01-01.

**Target Ledger:**
* **Semiconductor Layer:** TSMC (`TSM`)
* **Compute/AI Layer:** NVIDIA (`NVDA`), META (`META`), Alphabet (`GOOGL`)
* **Cooling & Rack Infrastructure:** Vertiv (`VRT`), Modine Manufacturing (`MOD`), Super Micro Computer (`SMCI`)

### 2. Multi-Source News Collector (`scrapers/collect_infrastructure_news.py`)

Scope: Collects AI infrastructure and semiconductor news from 6 sources using RSS feeds and HTML scraping.

**News Sources:**

| Source | Method | Focus Area |
|---|---|---|
| Reuters | Google News RSS | Technology, AI, semiconductor finance |
| TechCrunch AI | RSS | Artificial intelligence, tech |
| Semiconductor Engineering | RSS | Semiconductor, chips |
| The Register | Atom RSS | Data center, infrastructure |
| Tom's Hardware | RSS | Hardware, semiconductor |
| SemiAnalysis | RSS | Semiconductor analysis |
| DataCenterDynamics | HTML scraping | Data center infrastructure |

**Keyword Filtering:** Articles are filtered against a matrix of AI infrastructure terms including: `ai`, `semiconductor`, `nvidia`, `tsmc`, `data center`, `cooling`, `power`, `grid`, `gpu`, and more.

### 3. Data Visualization (`scrapers/visualize_data.py`)

Scope: Generates publication frequency, source distribution, keyword frequency, timeline, and headline length charts from collected data.

---

### 📂 Repository Structure

```text
├── scrapers/                           # Core collection engine scripts
│   ├── AI_Stock.py                     # Historical market data ingestion
│   ├── collect_infrastructure_news.py  # Multi-source news collector
│   └── visualize_data.py               # Data visualization charts
├── data_samples/                       # Output data warehouse
│   ├── ai_infrastructure_stock_data.csv
│   ├── ai_infrastructure_news.csv      # Filtered AI news dataset
│   ├── all_news_raw.csv                # All scraped news (unfiltered)
│   └── charts/                         # Generated visualization PNGs
├── requirements.txt
└── README.md
```

### 🚀 Installation & Local Environment Setup

Ensure you have Python 3.10+ installed.

#### 1. Clone the Workspace

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

#### 2. Install Package Dependencies

```bash
pip install -r requirements.txt
```

### 🏃 Execution Manual

#### Step 1: Ingest Stock Data

```bash
python3 scrapers/AI_Stock.py
```

#### Step 2: Collect News from All Sources

```bash
python3 scrapers/collect_infrastructure_news.py
```

#### Step 3: Generate Visualizations

```bash
python3 scrapers/visualize_data.py
```

---

### 📊 Dataset Metadata & Features

#### Financial Time Series (`ai_infrastructure_stock_data.csv`)

| Feature | Data Type | Representation |
| :--- | :--- | :--- |
| **Date** | Timestamp | Time-zone naive date marker (YYYY-MM-DD) |
| **Open / Close** | Float64 | Dollar denomination trading values |
| **Volume** | Int64 | Consolidated share volume traded during session |
| **Company / Ticker** | String | Unique target asset identifiers |

#### AI Infrastructure News (`ai_infrastructure_news.csv`)

| Feature | Data Type | Representation |
| :--- | :--- | :--- |
| **date** | Timestamp | Article publication date (YYYY-MM-DD) |
| **headline** | String | Article headline text |
| **summary** | String | Article summary/description |
| **url** | String | Canonical article URL |
| **source** | String | News source name |
| **source_categories** | String | Source topic categories |
| **tags** | String | Article tags from feed |

---

### 🔮 Downstream Pipeline Context (Phase 2 & Phase 3)

The collected headlines and summaries provide the foundation for downstream data science tasks:

* **Natural Language Processing (Phase 2):** Passing extracted text into transformer models (e.g., FinBERT) to turn news stories into quantitative numeric scalar indicators:

$$\text{Sentiment Score} = P(\text{Positive}) - P(\text{Negative})$$

* **Feature Alignment:** Aggregating alternative scores daily and evaluating predictive lead-lag characteristics against hardware market cap metrics.
