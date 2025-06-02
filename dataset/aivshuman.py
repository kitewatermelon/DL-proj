import pandas as pd
import os

def load_aivshuman_dataset(csv_path='./data/AI_Human.csv'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if 'text' in df.columns and 'generated' in df.columns:
        df = df.rename(columns={'text': 'Generation'})
        df['label'] = df['generated'].astype(int)
        df = df.drop(columns=['generated'])
    else:
        raise KeyError("Expected columns 'text' and 'generated' not found.")

    df = df.dropna(subset=['Generation', 'label'])

    print(f"[✅ AIVSHuman] 데이터 로드 완료: {len(df):,} samples")
    return df
