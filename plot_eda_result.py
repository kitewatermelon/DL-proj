import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 비교할 데이터셋 이름
datasets = ["aivshuman", "daigt", "turingbench", "gptwriting", "gpt2output"]

# 비교할 시각화 종류
fig_names = [
    "label_distribution.png",
    "text_length_distribution.png",
    "text_length_by_label.png",
    "top_words_all.png",
    "top_words_label_0.png",
    "top_words_label_1.png",
    "top_2gram.png",
    "wordcloud_all.png",
    "wordcloud_label_0.png",
    "wordcloud_label_1.png"
]

base_dir = "/mnt/c/Users/Administrator/Desktop/DL-proj/eda_outputs"
output_dir = os.path.join(base_dir, "summary_figures")
os.makedirs(output_dir, exist_ok=True)

for fig_name in fig_names:
    fig, axs = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4))
    if len(datasets) == 1:
        axs = [axs]
    for i, dataset in enumerate(datasets):
        fig_path = os.path.join(base_dir, dataset, "eda_plots", fig_name)
        if os.path.exists(fig_path):
            img = mpimg.imread(fig_path)
            axs[i].imshow(img)
            axs[i].axis("off")
            axs[i].set_title(dataset, fontsize=12)
        else:
            axs[i].axis("off")
            axs[i].set_title(f"{dataset}\n(No Image)", fontsize=10, color="red")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"combined_{fig_name}")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 저장 완료: {save_path}")
