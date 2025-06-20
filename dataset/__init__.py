from sklearn.model_selection import train_test_split
import pandas as pd
from .turingbench import load_turingbench_dataset
from .aivshuman import load_aivshuman_dataset
from .daigt import load_daigt_dataset
from .gptwritingprompt import load_gptwritingprompt_dataset
from .gpt2output import load_gpt2output_dataset
import re
from nltk.corpus import stopwords
from nltk import download

# 처음 한 번만 다운로드 필요
download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # 특수문자 제거
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

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
    train_dfs, valid_dfs, test_dfs = [], [], []

    total_ratio = train_ratio + valid_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError(f"[❌] 비율의 합은 1이어야 합니다. 현재 합: {total_ratio}")

    for name in dataset_names:
        loader_func = register_dataset(name)
        df = loader_func()
        df = df[['Generation', 'label']].dropna()
        df['Generation'] = df['Generation'].astype(str).apply(preprocess_text)

        # 각 데이터셋에 대해 개별적으로 split
        test_size = test_ratio
        valid_size = valid_ratio / (train_ratio + valid_ratio)

        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['label']
        )
        train_df, valid_df = train_test_split(
            train_val_df, test_size=valid_size, random_state=42, stratify=train_val_df['label']
        )

        train_dfs.append(train_df)
        valid_dfs.append(valid_df)
        test_dfs.append(test_df)

    # 동일 분할끼리 합치기
    train_df = pd.concat(train_dfs, ignore_index=True)
    valid_df = pd.concat(valid_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    print(f"[✅] 데이터 분할 및 병합 완료: Train={len(train_df):,} / Valid={len(valid_df):,} / Test={len(test_df):,}")
    return train_df, valid_df, test_df
