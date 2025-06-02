import pandas as pd
import os

def load_gptwritingprompt_dataset(csv_path='./data/gpt-writing-prompts.csv'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[❌] 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)

    if 'story' not in df.columns or 'writer' not in df.columns:
        raise ValueError("필수 컬럼(story, writer)이 누락되었습니다.")

    full_df = pd.DataFrame({
        'Generation': df['story'],
        'label': df['writer']
    })

    full_df.dropna(inplace=True)
    full_df['label'] = full_df['label'].apply(lambda x: 0 if str(x).strip().lower() == 'human' else 1).astype(int)

    print(f"[✅ GPTWritingPrompt] 데이터 로드 완료: {len(full_df):,} samples")
    return full_df
