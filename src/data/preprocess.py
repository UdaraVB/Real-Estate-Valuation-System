import pandas as pd
import re

def clean_pakistan_real_estate(df):
    # 1. Standardize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # 2. Enhanced Price Conversion Logic
    def convert_price(price_str):
        price_str = str(price_str).lower().strip().replace(',', '')
        number_match = re.search(r"(\d+\.?\d*)", price_str)
        if not number_match:
            return 0
        
        val = float(number_match.group(1))
        if 'crore' in price_str:
            return int(val * 10_000_000)
        elif 'lakh' in price_str:
            return int(val * 100_000)
        return int(val)

    df['price'] = df['price'].apply(convert_price)

    # 3. Fixed Area Logic
    # We use .str.extract to grab ONLY the first number (e.g., '120' or '120.5')
    # This ignores the 'Sq. Yd.' dots entirely!
    df['area'] = df['area'].astype(str).str.replace(',', '')
    df['area'] = df['area'].str.extract(r'(\d+\.?\d*)')[0].astype(float)

    return df