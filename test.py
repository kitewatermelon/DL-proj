import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, roc_auc_score
from dataset_class import TextDataset
from models.bert_classifier import BertClassifier

def test_model(test_df, model_path="best_model.pt", model_name='bert-base-uncased', batch_size=16):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertClassifier(model_name).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_dataset = TextDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    preds, truths = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['label'].cuda()

            outputs = model(input_ids, attention_mask)
            preds.extend(torch.sigmoid(outputs).cpu().numpy())
            truths.extend(labels.cpu().numpy())

    preds_bin = [1 if p > 0.5 else 0 for p in preds]
    acc = accuracy_score(truths, preds_bin)
    auc = roc_auc_score(truths, preds)
    print(f"[Test] Accuracy: {acc:.4f}, AUC: {auc:.4f}")
