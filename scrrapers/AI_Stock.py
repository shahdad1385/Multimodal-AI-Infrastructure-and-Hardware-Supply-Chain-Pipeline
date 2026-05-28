import os
import pandas as pd
import yfinance as yf
from datetime import datetime

def fetch_ai_ecosystem_data():
    print("🚀 Starting AI Infrastructure Data Collection Pipeline...")
    
    # 1. Define the ticker mapping (Publicly Traded Targets only)8
    ticker_dict = {
        # Core AI / Big Tech
        "NVIDIA": "NVDA",
        "META": "META",
        "GOOGLE": "GOOGL",
        # Semiconductor / Foundry Layer
        "TSMC": "TSM",  # US ADR ticker for TSMC
        # Top AI Data Center Cooling & Infrastructure Giants
        "VERTIV": "VRT",
        "MODINE_COOLING": "MOD",
        "SUPER_MICRO_COOLING": "SMCI"
    }
    
    # Define time horizon (
        # Start Date is almost before the raise of AI
    start_date = "2018-01-01" 
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    master_df = pd.DataFrame()
    
    # 2. Fetch and loop through each ticker
    for company_name, ticker in ticker_dict.items():
        print(f"📥 Fetching historical data for {company_name} ({ticker})...")
        try:
            # Initialize ticker object
            stock = yf.Ticker(ticker)
            
            # Fetch maximum available historical data within our timeframe
            df = stock.history(start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                print(f"⚠️ No data found for {ticker}. Skipping.")
                continue
                
            # Reset index to bring Date in as a column
            df = df.reset_index()
            
            # Clean/Standardize column names
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Add metadata columns to identify the asset
            df['Company'] = company_name
            df['Ticker'] = ticker
            
            # Ensure Date is timezone-naive to prevent merge/export formatting conflicts
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            
            # Append to master dataframe
            master_df = pd.concat([master_df, df], ignore_index=True)
            print(f"✅ Successfully collected {len(df)} rows for {company_name}.")
            
        except Exception as e:
            print(f"❌ Failed to fetch data for {ticker}. Error: {e}")
            
    # 3. Export Pipeline Output
    if not master_df.empty:
        output_filename = "ai_infrastructure_stock_data.csv"
        
        # Sort data logically by Date and Company
        master_df = master_df.sort_values(by=['Date', 'Company']).reset_index(drop=True)
        
        # Save to CSV
        master_df.to_csv(output_filename, index=False)
        
        print("\n" + "="*50)
        print("🎉 DATA COLLECTION SUCCESSFUL!")
        print(f"📊 Total Dataset Rows: {len(master_df)}")
        print(f"📂 Saved file as: {os.path.abspath(output_filename)}")
        print("="*50)
        
        # Display a quick structural sample for verification
        print("\n👀 Dataset Sample Preview:")
        print(master_df.head(10))
    else:
        print("❌ Pipeline completed, but no data was collected.")

if __name__ == "__main__":
    fetch_ai_ecosystem_data()