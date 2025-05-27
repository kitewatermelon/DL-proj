import pandas as pd
import os

def load_gpt2output_dataset(base_dir='data/GPT2OutputDataset'):
    all_data = []

    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)

        if filename.endswith('.jsonl'):
            try:
                df = pd.read_json(file_path, lines=True)
            except ValueError:
                print(f"[⚠️ 오류] {filename} 파일은 JSONL 형식이 아닙니다. 건너뜁니다.")
                continue

            if 'webtext' in filename:
                df['label'] = 'human'
            else:
                df['label'] = 'AI'

            df = df.rename(columns={'text': 'Generation'})  # 텍스트 컬럼명 정리
            all_data.append(df)

    if not all_data:
        raise RuntimeError("📛 데이터 파일이 하나도 로드되지 않았습니다.")

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df[['Generation', 'label']].dropna()

    print(f"[✅ GPT2OutputDataset] 데이터 로드 완료: {len(full_df):,} samples")

    return full_df
