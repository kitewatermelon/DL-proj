import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from data_utils import load_balanced_dataset

# t-test 함수
def perform_ttest(scores1, scores2, model1_name, model2_name, results, metric='Accuracy'):
    t_stat, p_value = ttest_rel(scores1, scores2)
    results.append({
        'Model Pair': f"{model1_name} vs {model2_name}",
        't-statistic': t_stat,
        'p-value': p_value,
        'Significant': 'Yes' if p_value < 0.05 else 'No',
        'Metric': metric
    })

# 혼동 행렬 시각화 함수
def plot_confusion_matrix(y_true, y_pred, model_name, fold):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name} (Fold {fold + 1})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}_fold_{fold + 1}.png')
    plt.close()

if __name__ == "__main__":
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터셋 로드
    train_df, val_df, test_df = load_balanced_dataset("cleaned")
    texts = test_df['text'].tolist()
    labels = test_df['label'].tolist()

    # 10-fold 설정
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_indices = list(kf.split(texts))

    # 결과 저장용 리스트
    bert_acc_scores = []
    opt_acc_scores = []
    logistic_acc_scores = []
    svm_acc_scores = []
    ttest_results = []

    # TF-IDF 벡터라이저 로드 또는 생성
    try:
        tfidf_vectorizer = joblib.load('models/tfidf_vectorizer_cleaned.pkl')
    except FileNotFoundError:
        train_texts = train_df['text'].tolist()
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_vectorizer.fit(train_texts)
        joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer_cleaned.pkl')
    X_test = tfidf_vectorizer.transform(texts)

    # 사전 모델 로딩
    bert_model_name = 'bert-base-uncased'
    opt_model_name = 'bert-base-uncased'

    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    opt_tokenizer = AutoTokenizer.from_pretrained(opt_model_name)

    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=2)
    bert_model.load_state_dict(torch.load('models/bert_best_model_cleaned.pt'))
    bert_model.to(device)
    bert_model.eval()

    opt_model = AutoModelForSequenceClassification.from_pretrained(opt_model_name, num_labels=2)
    opt_model.load_state_dict(torch.load('models/best_model_cleaned_optimize.pt'))
    opt_model.to(device)
    opt_model.eval()

    logistic_model = joblib.load('models/logistic_cleaned.pkl')
    svm_model = joblib.load('models/svm_cleaned.pkl')

    # 각 폴드별 평가
    for fold, (train_idx, test_idx) in enumerate(fold_indices):
        fold_texts = [texts[i] for i in test_idx]
        fold_labels = [labels[i] for i in test_idx]
        fold_X_test = X_test[test_idx]

        # BERT 예측
        bert_encodings = bert_tokenizer(fold_texts, truncation=True, padding=True, return_tensors='pt')
        bert_preds = []
        with torch.no_grad():
            for i in range(0, len(fold_texts), 16):
                batch_inputs = bert_encodings['input_ids'][i:i+16].to(device)
                batch_mask = bert_encodings['attention_mask'][i:i+16].to(device)
                outputs = bert_model(batch_inputs, attention_mask=batch_mask)
                batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                bert_preds.extend(batch_preds)
        bert_acc = accuracy_score(fold_labels, bert_preds)
        bert_acc_scores.append(bert_acc)
        plot_confusion_matrix(fold_labels, bert_preds, "BERT", fold)

        # 최적화된 모델 예측
        opt_encodings = opt_tokenizer(fold_texts, truncation=True, padding=True, return_tensors='pt')
        opt_preds = []
        with torch.no_grad():
            for i in range(0, len(fold_texts), 16):
                batch_inputs = opt_encodings['input_ids'][i:i+16].to(device)
                batch_mask = opt_encodings['attention_mask'][i:i+16].to(device)
                outputs = opt_model(batch_inputs, attention_mask=batch_mask)
                batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                opt_preds.extend(batch_preds)
        opt_acc = accuracy_score(fold_labels, opt_preds)
        opt_acc_scores.append(opt_acc)
        plot_confusion_matrix(fold_labels, opt_preds, "Optimized Model", fold)

        # Logistic Regression
        logistic_preds = logistic_model.predict(fold_X_test)
        logistic_acc = accuracy_score(fold_labels, logistic_preds)
        logistic_acc_scores.append(logistic_acc)
        plot_confusion_matrix(fold_labels, logistic_preds, "Logistic Regression", fold)

        # SVM
        svm_preds = svm_model.predict(fold_X_test)
        svm_acc = accuracy_score(fold_labels, svm_preds)
        svm_acc_scores.append(svm_acc)
        plot_confusion_matrix(fold_labels, svm_preds, "SVM", fold)

        print(f"Fold {fold + 1} - BERT Acc: {bert_acc:.4f}, Opt Acc: {opt_acc:.4f}, Logistic Acc: {logistic_acc:.4f}, SVM Acc: {svm_acc:.4f}")

    # 평균 정확도 출력
    print("\nAverage Accuracy Scores:")
    print(f"BERT: {np.mean(bert_acc_scores):.4f} ± {np.std(bert_acc_scores):.4f}")
    print(f"Optimized Model: {np.mean(opt_acc_scores):.4f} ± {np.std(opt_acc_scores):.4f}")
    print(f"Logistic Regression: {np.mean(logistic_acc_scores):.4f} ± {np.std(logistic_acc_scores):.4f}")
    print(f"SVM: {np.mean(svm_acc_scores):.4f} ± {np.std(svm_acc_scores):.4f}")

    # t-test 수행
    perform_ttest(bert_acc_scores, opt_acc_scores, "BERT", "Optimized Model", ttest_results)
    perform_ttest(bert_acc_scores, logistic_acc_scores, "BERT", "Logistic Regression", ttest_results)
    perform_ttest(bert_acc_scores, svm_acc_scores, "BERT", "SVM", ttest_results)
    perform_ttest(opt_acc_scores, logistic_acc_scores, "Optimized Model", "Logistic Regression", ttest_results)
    perform_ttest(opt_acc_scores, svm_acc_scores, "Optimized Model", "SVM", ttest_results)
    perform_ttest(logistic_acc_scores, svm_acc_scores, "Logistic Regression", "SVM", ttest_results)

    # t-test 결과 저장
    ttest_df = pd.DataFrame(ttest_results)
    ttest_df.to_csv('ttest_results_accuracy.csv', index=False)
    print("\nt-test results saved to 'ttest_results_accuracy.csv':")
    print(ttest_df)

    # 폴드별 정확도 박스플롯
    acc_data = pd.DataFrame({
        'BERT': bert_acc_scores,
        'Optimized Model': opt_acc_scores,
        'Logistic Regression': logistic_acc_scores,
        'SVM': svm_acc_scores
    })
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=acc_data)
    plt.title('Accuracy Distribution Across 10 Folds')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_distribution.png')
    plt.close()

    # 평균 정확도 바 차트
    accuracies = {
        'BERT': np.mean(bert_acc_scores),
        'Optimized Model': np.mean(opt_acc_scores),
        'Logistic Regression': np.mean(logistic_acc_scores),
        'SVM': np.mean(svm_acc_scores)
    }
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.title('Average Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('accuracy_comparison.png')
    plt.close()
