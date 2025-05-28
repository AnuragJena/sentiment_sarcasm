from google.colab import drive
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the directory on Google Drive where the file is stored
folder_path = '/content/drive/My Drive/Colab Files/Output/'

# List all files in the folder
files = os.listdir(folder_path)

# Filter files that start with the prefix and end with ".csv"
csv_files = [file for file in files if file.startswith('sentiment_sarcasm_reviews_') and file.endswith('.csv')]
print("CSV Files -->", csv_files)

# Sort files by modification time to get the latest one
csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)

# The latest file is the first in the sorted list
latest_file = csv_files[0]

# Define the full path to the latest file
latest_file_path = os.path.join(folder_path, latest_file)

print("Folder Path -->", folder_path)
print("Files in the folder -->", files)
print(f"The latest file is: {latest_file}")
print(f"Full Address -----> {latest_file_path}")

# Load the DataFrame from the latest CSV file
df = pd.read_csv(latest_file_path)

# Drop missing and map labels
df = df.dropna(subset=["Review", "Sentiment", "Sarcasm"])
df["Sentiment"] = df["Sentiment"].map({"Positive": 1, "Negative": 0})
df["Sarcasm"] = df["Sarcasm"].map({"Yes": 1, "No": 0})

# Tokenizer and model base
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder_model = AutoModel.from_pretrained(model_name)

# Dataset
class MultiTaskDataset(Dataset):
    def __init__(self, texts, sentiments, sarcasms, tokenizer, max_len=128):
        self.texts = texts
        self.sentiments = sentiments
        self.sarcasms = sarcasms
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sentiment_label": torch.tensor(self.sentiments[idx], dtype=torch.float),
            "sarcasm_label": torch.tensor(self.sarcasms[idx], dtype=torch.float)
        }

# Model
class MultiTaskModel(nn.Module):
    def __init__(self, encoder, hidden_dim=128):
        super(MultiTaskModel, self).__init__()
        self.encoder = encoder
        self.lstm = nn.LSTM(input_size=encoder.config.hidden_size, hidden_size=hidden_dim,
                            batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(hidden_dim * 2, 64)
        self.sentiment_out = nn.Linear(64, 1)
        self.sarcasm_out = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask):
        enc_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x, _ = self.lstm(enc_output.last_hidden_state)
        attn_weights = torch.softmax(self.attn(x), dim=1)
        x = torch.sum(attn_weights * x, dim=1)
        x = self.dropout(x)
        x = torch.relu(self.dense(x))
        return self.sentiment_out(x), self.sarcasm_out(x)

# Prepare data
X_train, X_test, y_train_sent, y_test_sent, y_train_sarc, y_test_sarc = train_test_split(
    df["Review"].tolist(), df["Sentiment"].tolist(), df["Sarcasm"].tolist(),
    test_size=0.2, random_state=42
)

train_dataset = MultiTaskDataset(X_train, y_train_sent, y_train_sarc, tokenizer)
test_dataset = MultiTaskDataset(X_test, y_test_sent, y_test_sarc, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskModel(encoder_model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()


train_losses, train_accuracies = [], []
for epoch in range(10):
    model.train()
    total_loss = 0
    correct_preds = 0
    total_preds = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sentiment_labels = batch["sentiment_label"].to(device).unsqueeze(1)
        sarcasm_labels = batch["sarcasm_label"].to(device).unsqueeze(1)

        optimizer.zero_grad()
        sent_out, sarc_out = model(input_ids, attention_mask)
        sent_loss = criterion(sent_out, sentiment_labels)
        sarc_loss = criterion(sarc_out, sarcasm_labels)
        loss = sent_loss + sarc_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(sent_out) > 0.5).float()
        correct_preds += (preds == sentiment_labels).sum().item()
        total_preds += sentiment_labels.size(0)

    avg_loss = total_loss / len(train_loader)
    acc = correct_preds / total_preds
    train_losses.append(avg_loss)
    train_accuracies.append(acc)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
# Save model
model_path = "/content/drive/MyDrive/Colab Notebooks/multitask_sentiment_sarcasm_review.pt"
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to {model_path}")
# Evaluation & Visualization
from sklearn.metrics import (classification_report, precision_score,
recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve)
import seaborn as sns

# Evaluate on test set
model.eval()
all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["sentiment_label"].to(device).unsqueeze(1)
        out, _ = model(input_ids, attention_mask)
        probs = torch.sigmoid(out).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds.flatten())
        all_probs.extend(probs.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

print("Accuracy:", accuracy_score(all_labels, all_preds)) # Use all_preds which contains class labels (0 or 1)
print(classification_report(all_labels, all_preds))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
cm = confusion_matrix(all_labels, all_preds)
fpr, tpr, _ = roc_curve(all_labels, all_probs)
auc_score = roc_auc_score(all_labels, all_probs)

# Visuals
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0, 0])
axs[0, 0].set_title("Confusion Matrix")
axs[0, 0].set_xlabel("Predicted")
axs[0, 0].set_ylabel("Actual")

axs[0, 1].plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
axs[0, 1].plot([0, 1], [0, 1], linestyle='--')
axs[0, 1].set_title("ROC Curve")
axs[0, 1].set_xlabel("False Positive Rate")
axs[0, 1].set_ylabel("True Positive Rate")
axs[0, 1].legend()

axs[1, 0].plot(train_losses, marker='o')
axs[1, 0].set_title("Training Loss per Epoch")
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Loss")

axs[1, 1].plot(train_accuracies, marker='o', color='green')
axs[1, 1].set_title("Training Accuracy per Epoch")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()
