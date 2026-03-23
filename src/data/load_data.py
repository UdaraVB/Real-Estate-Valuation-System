import pandas as pd
from pathlib import Path

file_path = Path(r"E:\AI\Datasets\homes.csv")

if not file_path.exists():
    print("File not found →", file_path)
    print("Current folder →", Path.cwd())
else:
    df = pd.read_csv(file_path)
    print("Shape:", df.shape)
    print("\nMissing values:\n")
    print(df.isnull().sum())