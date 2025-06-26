import os
import time
import joblib
from data_utils import load_balanced_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from visualize import plot_confusion_matrix, plot_roc_curve, plot_pr_curve

# 모델 정의
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "svm": CalibratedClassifierCV(LinearSVC()),
 }

versions = ["cleaned"]

os.makedirs("plot", exist_ok=True)
os.makedirs("models", exist_ok=True)

for version in versions:
    print(f"\n\n===== 🔄 버전: {version} =====")
    train_df, val_df, test_df = load_balanced_dataset(version=version)

    # 벡터화
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]

    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"]

    for name, model in models.items():
        print(f"\n🚀 [{version} / {name}] 모델 학습 시작...")
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        print(f"⏱️ 학습 시간: {elapsed:.2f}초")

        y_pred = model.predict(X_test)

        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_prob = model.decision_function(X_test)

        # 성능 평가
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")
        print(f"✅ F1-score: {f1:.4f}")

        # 모델 저장
        model_path = f"models/{name}_{version}.pkl"
        joblib.dump(model, model_path)
        print(f"💾 모델 저장 완료: {model_path}")


        # 시각화 저장
        base_path = f"plot/{name}_{version}"
        plot_confusion_matrix(y_test, y_pred, f"{base_path}_confusion_matrix.png", f"Confusion Matrix - {version} / {name}")
        plot_roc_curve(y_test, y_prob, f"{base_path}_roc_curve.png", f"ROC Curve - {version} / {name}")
        plot_pr_curve(y_test, y_prob, f"{base_path}_pr_curve.png", f"PR Curve - {version} / {name}")
        print(f"📊 시각화 저장 완료: {base_path}_*.png")

    # 벡터라이저 저장
    vectorizer_path = f"models/tfidf_vectorizer_{version}.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    print(f"💾 벡터라이저 저장 완료: {vectorizer_path}")
