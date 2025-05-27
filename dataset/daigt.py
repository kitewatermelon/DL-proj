import pandas as pd
import os

def load_daigt_dataset(csv_path='./data/daigt_external_dataset.csv'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    
    if not {'text', 'source_text'}.issubset(df.columns):
        raise ValueError("Dataset must contain 'text' and 'source_text' columns.")

    # 두 개의 데이터프레임 생성
    human_df = df[['text']].copy()
    human_df.rename(columns={'text': 'Generation'}, inplace=True)
    human_df['label'] = 'human'

    ai_df = df[['source_text']].copy()
    ai_df.rename(columns={'source_text': 'Generation'}, inplace=True)
    ai_df['label'] = 'AI'

    # 병합 및 셔플
    full_df = pd.concat([human_df, ai_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[✅ DAIGT] 데이터 로드 완료: {len(full_df):,} samples")
    return full_df
