import pandas as pd
import re

def clean_pakistan_real_estate(df):
    # 1. Standardize columns
    df.columns = [c.lower().strip() for c in df.columns]
    
    # 2. Helper function for the math
    def convert_price(price_str):
        price_str = str(price_str).lower().strip()
        # Extract the number part (e.g., "1.5" from "1.5 Crore")
        number_match = re.search(r"(\d+\.?\d*)", price_str)
        if not number_match:
            return 0
            
        val = float(number_match.group(1))
        
        # Multiply based on unit
        if 'crore' in price_str:
            return int(val * 10_000_000)
        elif 'lakh' in price_str:
            return int(val * 100_000)
        return int(val)

    # 3. Apply the conversion
    df['price'] = df['price'].apply(convert_price)
    
    # 4. Clean up area (removing 'Sq. Yd.' and commas)
    df['area'] = df['area'].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
    
    return df