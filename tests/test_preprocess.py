import pandas as pd
import pytest
from src.data.preprocess import clean_pakistan_real_estate

def test_price_conversion_logic():
    """Test if '1 Crore' is correctly converted to 10,000,000"""
    data = {
        'price': ['1 Crore', '50 Lakh'],
        'area': ['120 Sq. Yd.', '500 Sq. Yd.'],
        'beds': [3, 5],
        'baths': [2, 5],
        'location': ['DHA', 'Gulshan']
    }
    df = pd.DataFrame(data)
    
    df_clean = clean_pakistan_real_estate(df)
    
    assert df_clean['price'].iloc[0] == 10_000_000
    assert df_clean['price'].iloc[1] == 5_000_000

def test_dataframe_not_empty():
    """Ensure the cleaner doesn't delete all our rows"""
    data = {
        'price': ['1 Crore'], 'area': ['100 Sq. Yd.'], 
        'beds': [2], 'baths': [1], 'location': ['Nazimabad']
    }
    df = pd.DataFrame(data)
    df_clean = clean_pakistan_real_estate(df)
    assert len(df_clean) > 0