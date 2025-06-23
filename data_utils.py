import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def remove_stopwords(text):
    return ' '.join([w for w in text.split() if w.lower() not in stop_words])

def load_balanced_dataset(version="plain"):
    df = pd.read_csv("dataset.csv").dropna(subset=["Generation", "label"])
    if version == "cleaned":
        df["text"] = df["Generation"].astype(str).apply(remove_stopwords)
    else:
        df["text"] = df["Generation"].astype(str)

    df = df[["text", "label"]]
    human = df[df.label == 0]
    ai = df[df.label == 1]
    min_len = min(len(human), len(ai))
    df_balanced = pd.concat([
        resample(human, replace=False, n_samples=min_len, random_state=42),
        resample(ai, replace=False, n_samples=min_len, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    train_df, temp_df = train_test_split(df_balanced, test_size=0.1, stratify=df_balanced["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    return train_df, val_df, test_df
