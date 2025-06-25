import os
from data_utils import load_balanced_dataset  # 함수명 맞게 변경해주세요
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from visualize import plot_confusion_matrix, plot_roc_curve, plot_pr_curve

models = {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(),
    "naive_bayes": MultinomialNB(),
    "svm": CalibratedClassifierCV(LinearSVC())
}

versions = ["cleaned", "plain"]  # 두 버전

os.makedirs("plot", exist_ok=True)

for version in versions:
    print(f"\n\n===== 버전: {version} =====")
    train_df, val_df, test_df = load_balanced_dataset(version=version)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]

    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"]

    for name, model in models.items():
        print(f"\n🚀 [{version} / {name}] 모델 학습 중...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")
        print(f"✅ F1-score: {f1:.4f}")

        # 시각화 저장
        base_path = f"plot/{name}"
        plot_confusion_matrix(y_test, y_pred, f"{base_path}_confusion_matrix_{version}.png", f"Confusion Matrix - {version} / {name}")
        plot_roc_curve(y_test, y_prob, f"{base_path}_roc_curve_{version}.png", f"ROC Curve - {version} / {name}")
        plot_pr_curve(y_test, y_prob, f"{base_path}_pr_curve_{version}.png", f"PR Curve - {version} / {name}")
        print(f"📊 [{version} / {name}] 시각화 저장 완료")
