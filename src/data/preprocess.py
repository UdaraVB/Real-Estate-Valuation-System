import pandas as pd 
import re

def clean_pakistan_real_estate(df):

    df['Area'] = df['Area'].astype(str).str.replace(',', '')
    df['Area'] = df['Area'].str.extract(r'(\d+\.?\d*)').astype(float)

    def convert_price(price_str):
        price_str = str(price_str).lower().replace(',', '')
        match = re.search(r'(\d+\.?\d*)', price_str)
        if not match:
            return None
        val = float(match.group(1))
        if 'crore' in price_str:
            return val * 10_000_000
        elif 'lakh' in price_str:
            return val * 100_000
        return val

    df['Price'] = df['Price'].apply(convert_price)
    df['Baths'] = pd.to_numeric(df['Baths'], errors='coerce')
    df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')

    return df.dropna(subset=['Area', 'Baths', 'Bedrooms', 'Price'])

if __name__ == "__main__":
    import pandas as pd
    test_data = pd.DataFrame({'Price': ['2 Crore'], 'Area': ['500'], 'Baths': ['4'], 'Bedrooms': ['5']})
    print("Testing the cleaning function...")
    print(clean_pakistan_real_estate(test_data))