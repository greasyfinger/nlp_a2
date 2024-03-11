import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torchcrf import CRF

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Set up the training and evaluation pipeline
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# Pre defined variables
num_labels = 14
idx_to_label = [
    "O",
    "CASE_NUMBER",
    "COURT",
    "DATE",
    "GPE",
    "JUDGE",
    "ORG",
    "OTHER_PERSON",
    "PETITIONER",
    "PRECEDENT",
    "PROVISION",
    "RESPONDENT",
    "STATUTE",
    "WITNESS",
]

bert_name = "nlpaueb/legal-bert-base-uncased"

bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
bert_model = AutoModel.from_pretrained(bert_name)


# Create a PyTorch dataset and data loader
class SequenceTaggingDataset(Dataset):
    def __init__(self, tokens, labels, tokenizer):
        self.tokens = tokens
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_to_idx = {label: idx for idx, label in enumerate(idx_to_label)}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        labels = self.labels[idx]

        encoded = self.tokenizer.encode_plus(
            tokens,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=100,
            return_tensors="pt",
            is_split_into_words=True,
        )

        # Retrieve word IDs for each token
        word_ids = encoded.word_ids()

        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)  # -100
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
                try:
                    label_ids.append(self.label_to_idx[labels[word_idx][2:]])
                except:
                    label_ids.append(self.label_to_idx[labels[word_idx]])
            else:
                label_ids.append(-100)  # -100
            previous_word_idx = word_idx

        encoded["labels"] = torch.tensor(label_ids, dtype=torch.long)

        return encoded["input_ids"][0], encoded["labels"]


# Load and preprocess the data
def load_data(file_path, auxil):
    with open(file_path, "r") as f:
        data = json.load(f)

    aux = []
    for entry in data.values():
        aux.append(entry[auxil])

    return aux


train_tokens = load_data("task_1/token_dataset/train_set.json", "tokens")
train_labels = load_data("task_1/label_dataset/train_set.json", "labels")

val_tokens = load_data("task_1/token_dataset/val_set.json", "tokens")
val_labels = load_data("task_1/label_dataset/val_set.json", "labels")

train_dataset = SequenceTaggingDataset(train_tokens, train_labels, bert_tokenizer)
val_dataset = SequenceTaggingDataset(val_tokens, val_labels, bert_tokenizer)

# Define your data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


class BiLSTMCRFSequenceTaggingModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMCRFSequenceTaggingModel, self).__init__()
        self.embedding = bert_model.embeddings
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, input_ids, labels):
        embeddings = self.embedding(input_ids)
        out, _ = self.rnn(embeddings)
        out = self.fc(self.dropout(out))
        if labels is not None:
            loss = -self.crf(out, labels)
            return loss
        else:
            predictions = self.crf.decode(out)
            return predictions


model = BiLSTMCRFSequenceTaggingModel(
    embedding_dim=768, hidden_dim=256, num_layers=1, num_classes=num_labels
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


def run_epoch(run_name, model):
    # Initialize W&B
    wandb.login(key="cbecd600ce14e66bbbed0c7b4bb7fb317f48a47a", relogin=True)
    wandb.init(project="nlp_a2", name=run_name)
    wandb.watch(model)

    # Train and evaluate the model
    num_epochs = 50
    best_val_f1 = 0
    best_val_loss = 100
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss = 0
        train_f1 = 0
        train_acc = 0
        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate F1-score
            predicted = outputs.argmax(dim=2).cpu().numpy()
            true = labels.cpu().numpy()
            train_f1 += f1_score(true.flatten(), predicted.flatten(), average="macro")
            train_acc += accuracy_score(true.flatten(), predicted.flatten())

        model.eval()
        val_loss = 0
        val_f1 = 0
        val_acc = 0
        with torch.no_grad():
            for inputs, labels in val_loader:

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = model(inputs, labels)
                val_loss += loss.item()

                # Calculate F1-score
                predicted = outputs.argmax(dim=2).cpu().numpy()
                true = labels.cpu().numpy()
                val_f1 += f1_score(true.flatten(), predicted.flatten(), average="macro")
                val_acc += accuracy_score(true.flatten(), predicted.flatten())

        # Log metrics to W&B
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_f1 /= len(train_loader)
        val_f1 /= len(val_loader)
        train_acc /= len(train_loader)
        val_acc /= len(val_loader)

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_f1": train_f1,
                "val_f1": val_f1,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            }
        )

        # Save the best model checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "final" + run_name + ".pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Check if training should be stopped
        if epochs_without_improvement >= 5:
            print(f"Stopping early at epoch {epoch+1} due to no improvement.")
            break

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Train F1: {train_f1}, Val F1: {val_f1}"
        )

    wandb.finish()


run_epoch("BiLSTM_CRF_Legal_bert", model)
