# Multimodal AI Infrastructure & Hardware Supply Chain Pipeline

### Phase 1: High-Performance Alternative Data Acquisition Engine

### 📋 Executive Overview

Predicting market valuation shifts for Big Tech AI market leaders requires data beyond traditional daily closing prices. This project implements a robust, multimodal ingestion pipeline designed to align standard financial time-series indicators with structured, alternative textual signals scraped from critical hardware supply chain and grid infrastructure networks.

The core objective is to map lag-correlations between physical constraints—such as grid capacity, high-velocity liquid cooling breakthroughs, and chip-foundry expansions—and the downstream valuation fluctuations of market cap leaders.

### 🛠️ System Architecture & Data Flow

The collection ecosystem operates through two distinct pipelines engineered to minimize anti-bot latency and handle JavaScript-heavy DOM rendering.

```text
                  ┌───────────────────────────────┐
                  │    Target Infrastructure      │
                  └───────────────┬───────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌─────────────────────────────────┐               ┌─────────────────────────────────┐
│       Yahoo Finance Web API     │               │  DataCenterDynamics News Portal │
├─────────────────────────────────┤               ├─────────────────────────────────┤
│ • Requests Library Ingestion    │               │ • Headless Selenium Engine      │
│ • Maximum Time Horizon (2018+)  │               │ • Dynamic Scroll & Render Loop  │
│ • Standardized Numerical CSV    │               │ • Keyword Matrix Soft-Filtering │
└────────────────\────────────────┘               └────────────────/────────────────┘
                  \                                               /
                   \──────────────────────┬──────────────────────/
                                          ▼
                           ┌─────────────────────────────┐
                           │    /data_samples Output     │
                           ├─────────────────────────────┤
                           │  • Market Time Series Data  │
                           │  • Raw Alternative News Data│
                           └─────────────────────────────┘


```
### 1. Financial Ingestion Pipeline (`scraper/collect_market_data.py`)

Scope: Captures full-depth, historical trading indicators (Open, High, Low, Close, Volume) since 2018-01-01.

**Target Ledger:**
* **Semiconductor Layer:** TSMC (`TSM`)
* **Compute/AI Layer:** NVIDIA (`NVDA`), META (`META`), Alphabet (`GOOGL`)
* **Cooling & Rack Infrastructure:** Vertiv (`VRT`), Modine Manufacturing (`MOD`), Super Micro Computer (`SMCI`)

### 2. Alternative Textual Scraper (`scraper/collect_infrastructure_news.py`)

Scope: Bypasses deep cloud security firewalls via native browser automation to isolate infrastructure-specific signals.

**Feature Extraction Matrix:** Evaluates daily press feeds against a targeted boolean token index:
`["cooling", "liquid", "immersion", "datacenter", "tsmc", "semiconductor", "power plant", "grid", "electricity"]`

---

### 📂 Repository Structure

```text
├── scraper/                        # Core collection engine scripts
│   ├── collect_market_data.py      # Daily historical market data ingestion script
│   └── collect_infrastructure_news.py # Headless browser alternative text extraction script
├── data_samples/                   # Public data warehouse location for Phase 1 verification
│   ├── ai_infrastructure_stock_data.csv # Output time-series data
│   └── raw_alternative_news_data.csv    # Filtered project-relevant news indicators
└── README.md                       # Core pipeline documentation
```

### 🚀 Installation & Local Environment Setup

Ensure you have Python 3.10+ installed. This environment utilizes webdriver-manager to instantiate and manage headless binary components automatically.

#### 1. Clone the Workspace

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

#### 2. Install Package Dependencies

```bash
pip install yfinance pandas requests selenium webdriver-manager scrapy
```

### 🏃 Execution Manual

### Ingest Stock Performance Indices

To run the automated market acquisition loop and output to the unified CSV index, execute:

```bash
python3 collect_market_data.py
```

### Ingest Grid & Cooling Alternative Indicators

To spawn the automated scraping sequence, invoke the headless browser engine using:

```bash
python3 collect_infrastructure_news.py
```

### 📊 Dataset Metadata & Features

#### Financial Time Series Structure (`ai_infrastructure_stock_data.csv`)

| Feature | Data Type | Representation |
| :--- | :--- | :--- |
| **Date** | Timestamp | Time-zone naive date marker (YYYY-MM-DD) |
| **Open / Close** | Float64 | Dollar denomination trading values |
| **Volume** | Int64 | Consolidated share volume traded during session |
| **Company / Ticker** | String | Unique target asset identifiers |

#### Alternative Text Data Structure (`raw_alternative_news_data.csv`)

| Feature | Data Type | Representation |
| :--- | :--- | :--- |
| **date** | Timestamp | Date extracted or assigned at timestamp of scraper runtime |
| **headline** | String | Raw text scraped from article structural header element |
| **url** | String | Canonical URL path referencing source validation anchor |

---

### 🔮 Downstream Pipeline Context (Phase 2 & Phase 3)

The unstructured headlines gathered by this script provide the foundation for downstream data science tasks:

* **Natural Language Processing (Phase 2):** Passing extracted text into transformer models (e.g., FinBERT) to turn news stories into quantitative numeric scalar indicators:

$$\text{Sentiment Score} = P(\text{Positive}) - P(\text{Negative})$$

* **Feature Alignment:** Aggregating alternative scores daily and evaluating predictive lead-lag characteristics against hardware market cap metrics.
