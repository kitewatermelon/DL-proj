import pandas as pd
import os

def load_turingbench_dataset(root_path='./data/TuringBench'):
    data_list = []

    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Dataset root path not found: {root_path}")

    for folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder)
        if not os.path.isdir(folder_path):
            continue

        for split in ['train', 'valid', 'test']:
            csv_path = os.path.join(folder_path, f"{split}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['split'] = split  # 어디 데이터인지 표시
                data_list.append(df)

    if not data_list:
        print("[⚠️] 데이터가 없습니다.")
        return pd.DataFrame()

    full_df = pd.concat(data_list, ignore_index=True)

    # label 전처리
    if 'label' in full_df.columns:
        full_df['label'] = full_df['label'].apply(lambda x: 0 if str(x).strip().lower() == 'human' else 1)

    print(f"[✅ TuringBench] 전체 데이터 로드 완료: 총 {len(full_df):,}개 샘플")

    return full_df
