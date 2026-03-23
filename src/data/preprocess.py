def clean_pakistan_real_estate(df):
    # 1. Standardize column names to lowercase and remove extra spaces
    df.columns = [c.lower().strip() for c in df.columns]
    
    # 2. Now use lowercase 'area' instead of 'Area'
    df['area'] = df['area'].astype(str).str.replace(',', '')
    
    # 3. Do the same for 'price' (make sure it's lowercase 'p')
    # ... rest of your cleaning logic using lowercase column names ...
    
    return df