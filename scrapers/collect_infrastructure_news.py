import os
import time
from datetime import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data_samples')


def run_advanced_news_scraper():
    print("🚀 Initializing Selenium WebDriver for AI Infrastructure Data Extraction...")

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    target_url = "https://www.datacenterdynamics.com/en/news/"
    filtered_data = []
    all_news_data = []

    target_keywords = [
        "cooling", "liquid", "immersion", "ai", "datacenter", "center", "nvidia",
        "tsmc", "taiwan", "china", "semiconductor", "chip", "gpus", "gpu",
        "power", "plant", "electricity", "energy", "grid", "utilities", "infrastructure"
    ]

    try:
        print(f"🔎 Navigating to target portal: {target_url}")
        driver.get(target_url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "a"))
        )

        print("📥 Executing dynamic scroll requests to load archival dataset history...")
        for _ in range(2):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)

        print("📊 Extracting text and payload links from DOM anchor elements...")
        articles = driver.find_elements(By.TAG_NAME, "a")
        print(f"🔎 Total raw anchor links intercepted in memory: {len(articles)}")

        for idx, article in enumerate(articles):
            try:
                title_text = article.text
                link_url = article.get_attribute("href")

                if not title_text or len(title_text.strip()) < 15 or not link_url:
                    continue

                title_clean = title_text.strip().replace('\n', ' ')
                title_lower = title_clean.lower()

                item = {
                    "date": datetime.today().strftime('%Y-%m-%d'),
                    "headline": title_clean,
                    "url": link_url
                }

                all_news_data.append(item)

                match_found = any(keyword in title_lower for keyword in target_keywords)
                if match_found:
                    filtered_data.append(item)
                    print(f"✅ [MATCH FOUND - Element {idx}]: {title_clean[:60]}...")

            except Exception:
                continue

    except Exception as e:
        print(f"❌ Critical breakdown in core scraper thread architecture: {e}")
    finally:
        print("🔒 Terminating browser socket connection securely...")
        driver.quit()

    print("\n💾 Commencing CSV output write sequence...")

    if all_news_data:
        df_all = pd.DataFrame(all_news_data)
        df_all.drop_duplicates(subset=["headline"], inplace=True)
        df_all = df_all[df_all['url'].str.contains('/news/|/features/|/analysis/', case=False, na=False)]

        debug_path = os.path.join(OUTPUT_DIR, "all_found_news_debug.csv")
        df_all.to_csv(debug_path, index=False, encoding="utf-8-sig")
        print(f"📁 Debug dump file compiled successfully at: {os.path.abspath(debug_path)} (Records: {len(df_all)})")
    else:
        print("❌ Data payload processing aborted. No textual elements parsed. Verify proxy or network socket availability.")

    if filtered_data:
        df_filtered = pd.DataFrame(filtered_data)
        df_filtered.drop_duplicates(subset=["headline"], inplace=True)
        df_filtered = df_filtered[df_filtered['url'].str.contains('/news/|/features/|/analysis/', case=False, na=False)]

        if not df_filtered.empty:
            filtered_path = os.path.join(OUTPUT_DIR, "raw_alternative_news_data.csv")
            df_filtered.to_csv(filtered_path, index=False, encoding="utf-8-sig")
            print(f"✨ Target project dataset generated successfully at: {os.path.abspath(filtered_path)} (Records: {len(df_filtered)})")
        else:
            print("⚠️ Extracted elements did not match the required canonical news URL pathways.")
    else:
        print("⚠️ Pipeline complete, but no headlines intersected with current thematic keyword matrix array.")

if __name__ == "__main__":
    run_advanced_news_scraper()
