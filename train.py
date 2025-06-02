import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from models.bert_classifier import BertClassifier
from dataset_class import TextDataset
from visualize import plot_loss_accuracy

def train_model(train_df, valid_df, model_name='bert-base-uncased', epochs=3, batch_size=16, lr=2e-5, save_path="best_model.pt"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertClassifier(model_name).cuda()

    train_dataset = TextDataset(train_df, tokenizer)
    valid_dataset = TextDataset(valid_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*epochs)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_auc = 0.0

    train_losses, valid_losses, valid_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"[Train Epoch {epoch+1}]"):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['label'].cuda()

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                labels = batch['label'].cuda()
                outputs = model(input_ids, attention_mask)
                preds.extend(torch.sigmoid(outputs).cpu().numpy())
                truths.extend(labels.cpu().numpy())

        preds_bin = [1 if p > 0.5 else 0 for p in preds]
        acc = accuracy_score(truths, preds_bin)
        auc = roc_auc_score(truths, preds)
        valid_losses.append(loss_fn(torch.tensor(preds), torch.tensor(truths)).item())
        valid_accuracies.append(acc)

        print(f"[Valid] Acc: {acc:.4f}, AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved at epoch {epoch+1} with AUC: {auc:.4f}")

    plot_loss_accuracy(train_losses, valid_losses, valid_accuracies)
