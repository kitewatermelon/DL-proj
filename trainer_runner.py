import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score, f1_score
from visualize import plot_confusion_matrix, plot_roc_curve, plot_pr_curve, plot_loss_curve

def train_and_evaluate(version, train_df, val_df, test_df, device):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

    train_ds = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize_fn, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(tokenize_fn, batched=True)

    # 재할당 필수! (그래야 실제로 반영됨)
    train_ds = train_ds.remove_columns(["text"]).rename_column("label", "labels")
    val_ds = val_ds.remove_columns(["text"]).rename_column("label", "labels")
    test_ds = test_ds.remove_columns(["text"]).rename_column("label", "labels")

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = torch.from_numpy(logits).argmax(dim=-1).numpy()
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds)
        }

    training_args = TrainingArguments(
        output_dir=f"./results_{version}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        logging_dir=f"./logs_{version}",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    train_result = trainer.train()

    # train/val loss 기록 가져오기 (로그에서 확인 가능)
    train_losses = train_result.training_loss_history if hasattr(train_result, 'training_loss_history') else None
    # Trainer 기본 리턴에는 없어서 직접 콜백 구현 필요 (여기서는 생략 가능)

    results = trainer.predict(test_ds)
    logits = torch.tensor(results.predictions)
    probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
    preds = logits.argmax(dim=-1).numpy()
    labels = results.label_ids

    plot_confusion_matrix(labels, preds, f"plot/confusion_matrix_{version}.png", f"Confusion Matrix - {version}")
    plot_roc_curve(labels, probs, f"plot/roc_curve_{version}.png", f"ROC Curve - {version}")
    plot_pr_curve(labels, probs, f"plot/pr_curve_{version}.png", f"PR Curve - {version}")

    # train/val loss가 있으면 플롯, 없으면 무시
    if train_losses is not None:
        plot_loss_curve(train_losses, train_losses, f"plot/loss_curve_{version}.png", f"Loss Curve - {version}")

    torch.save(model.state_dict(), f"best_model_{version}.pt")
    print(f"✅ [{version}] 모델 저장 완료")
