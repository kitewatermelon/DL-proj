# DL-proj
---
### Dataset source
박연수 - TuringBench: https://huggingface.co/datasets/turingbench/TuringBench/tree/main   
이찬이 - DAIGT: https://www.kaggle.com/datasets/alejopaullier/daigt-external-dataset   
이채은 - GPT-WritingPrompts: https://huggingface.co/datasets/vkpriya/GPT-WritingPrompts/viewer   
정치우 - AI Vs Human Text: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text?resource=download   
최민서 - GPT2 Output Dataset: https://github.com/openai/gpt-2-output-dataset/blob/master/download_dataset.py (small 계열만 download)


---
```
data 폴더 구조
data/
├── AI_Human.csv
├── GPT2OutputDataset
│   ├── small-117M-k40.test.jsonl
│   ├── small-117M-k40.train.jsonl
│   ├── small-117M-k40.valid.jsonl
│   ├── small-117M.test.jsonl
│   ├── small-117M.train.jsonl
│   ├── small-117M.valid.jsonl
│   ├── webtext.test.jsonl
│   ├── webtext.train.jsonl
│   └── webtext.valid.jsonl
├── TuringBench
│   ├── AA
│   ├── ...
│   └── TT_xlnet_large
├── daigt_external_dataset.csv
└── gpt-writing-prompts.csv
```