import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

sns.set(style="whitegrid")

# 디렉토리 생성
os.makedirs("eda_outputs", exist_ok=True)
os.makedirs("eda_outputs/eda_plots", exist_ok=True)

def run_text_eda(df, output_dir="eda_outputs"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud
    from collections import Counter
    from sklearn.feature_extraction.text import CountVectorizer

    sns.set(style="whitegrid")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/eda_plots", exist_ok=True)

    df = df[['Generation', 'label']].dropna()
    df['Generation'] = df['Generation'].astype(str)
    df['text_length'] = df['Generation'].apply(lambda x: len(x.split()))

    # 1. 라벨 분포
    label_counts = df['label'].value_counts()
    label_counts.to_csv(f"{output_dir}/label_distribution.csv")
    sns.countplot(x='label', data=df)
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/eda_plots/label_distribution.png")
    plt.close()

    # 2. 텍스트 길이 분포
    text_len_stats = df['text_length'].describe()
    text_len_stats.to_csv(f"{output_dir}/text_length_stats.csv")
    sns.histplot(df['text_length'], kde=True, bins=30)
    plt.title('Distribution of Text Lengths (in Words)')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.savefig(f"{output_dir}/eda_plots/text_length_distribution.png")
    plt.close()

    # 3. 라벨별 텍스트 길이 분포
    sns.boxplot(x='label', y='text_length', data=df)
    plt.title('Text Length by Label')
    plt.xlabel('Label')
    plt.ylabel('Text Length (Words)')
    plt.savefig(f"{output_dir}/eda_plots/text_length_by_label.png")
    plt.close()

    # 4. 전체 Top Words
    plot_top_words(df, output_dir=output_dir, save_name="all")

    # 5. 라벨별 Top Words
    for label in df['label'].unique():
        plot_top_words(df, label=label, output_dir=output_dir, save_name=f"label_{label}")

    # 6. 2-gram
    plot_ngrams(df, output_dir=output_dir, ngram_range=(2, 2), top_n=15)

    # 7. 샘플 출력
    samples = []
    for label in df['label'].unique():
        s = df[df['label'] == label]['Generation'].sample(3, random_state=42).to_frame()
        s['label'] = label
        samples.append(s)
    sample_df = pd.concat(samples)
    sample_df.to_csv(f"{output_dir}/sample_texts_by_label.csv", index=False)

def plot_top_words(df, label=None, output_dir="eda_outputs", save_name="default", n=20):
    from wordcloud import WordCloud
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import Counter

    if label is not None:
        text = ' '.join(df[df['label'] == label]['Generation'])
    else:
        text = ' '.join(df['Generation'])

    words = text.lower().split()
    common_words = Counter(words).most_common(n)

    pd.DataFrame(common_words, columns=['word', 'count']).to_csv(
        f"{output_dir}/top_words_{save_name}.csv", index=False)

    words_, counts_ = zip(*common_words)
    sns.barplot(x=list(counts_), y=list(words_))
    plt.title(f"Top {n} Words" + (f" for Label {label}" if label else ""))
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.savefig(f"{output_dir}/eda_plots/top_words_{save_name}.png")
    plt.close()

    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud" + (f" for Label {label}" if label else ""))
    plt.savefig(f"{output_dir}/eda_plots/wordcloud_{save_name}.png")
    plt.close()


def plot_ngrams(df, output_dir="eda_outputs", ngram_range=(2, 2), top_n=20):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(ngram_range=ngram_range).fit(df['Generation'])
    bag_of_words = vec.transform(df['Generation'])
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]

    pd.DataFrame(words_freq, columns=['ngram', 'count']).to_csv(
        f"{output_dir}/top_{ngram_range[0]}gram.csv", index=False)

    words, counts = zip(*words_freq)
    sns.barplot(x=list(counts), y=list(words))
    plt.title(f"Top {top_n} {ngram_range[0]}-grams")
    plt.xlabel('Frequency')
    plt.ylabel('N-gram')
    plt.savefig(f"{output_dir}/eda_plots/top_{ngram_range[0]}gram.png")
    plt.close()


from dataset import load_all_datasets
import pandas as pd
import os

DATASETS = ['daigt', 'turingbench', 'aivshuman', 'gptwriting', 'gpt2output']

for dataset_name in DATASETS:
    # 1. 데이터 로드
    train_df, valid_df, test_df = load_all_datasets([dataset_name])
    full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    # 2. 출력 경로 생성
    output_dir = f"eda_outputs/{dataset_name}"
    os.makedirs(f"{output_dir}/eda_plots", exist_ok=True)

    # 3. EDA 실행
    run_text_eda(full_df, output_dir=output_dir)


