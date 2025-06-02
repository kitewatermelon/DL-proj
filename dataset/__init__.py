from sklearn.model_selection import train_test_split
import pandas as pd
from .turingbench import load_turingbench_dataset
from .aivshuman import load_aivshuman_dataset
from .daigt import load_daigt_dataset
from .gptwritingprompt import load_gptwritingprompt_dataset
from .gpt2output import load_gpt2output_dataset

def register_dataset(name):
    if name.lower() == 'turingbench':
        print('loading turingbench ...')
        return load_turingbench_dataset
    elif name.lower() == 'aivshuman':
        print('loading aivshuman ...')
        return load_aivshuman_dataset
    elif name.lower() == 'daigt':
        print('loading daigt ...')
        return load_daigt_dataset
    elif name.lower() == 'gptwriting':
        print('loading gptwriting ...')
        return load_gptwritingprompt_dataset
    elif name.lower() == 'gpt2output':
        print('loading gpt2output ...')
        return load_gpt2output_dataset
    raise ValueError(f"Dataset module for '{name}' not found.")

def load_all_datasets(dataset_names, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    all_data = []

    for name in dataset_names:
        loader_func = register_dataset(name)
        df = loader_func()
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df[['Generation', 'label']].dropna()

    # 비율 검증
    total_ratio = train_ratio + valid_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError(f"[❌] 비율의 합은 1이어야 합니다. 현재 합: {total_ratio}")

    # 분할
    test_size = test_ratio
    valid_size = valid_ratio / (train_ratio + valid_ratio)

    train_val_df, test_df = train_test_split(
        full_df, test_size=test_size, random_state=42, stratify=full_df['label']
    )
    train_df, valid_df = train_test_split(
        train_val_df, test_size=valid_size, random_state=42, stratify=train_val_df['label']
    )
    print(train_df['label'].unique())
    print(f"[✅] 데이터 분할 완료: Train={len(train_df):,} / Valid={len(valid_df):,} / Test={len(test_df):,}")
    return train_df, valid_df, test_df
