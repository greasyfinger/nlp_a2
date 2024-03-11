from gensim.models import KeyedVectors
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from torchcrf import CRF
import numpy as np

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Set up the training and evaluation pipeline
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

START_TAG = "<START>"
STOP_TAG = "<STOP>"

# Pre defined variables
num_labels = 14
tag_to_ix = {
    "O": 0,
    "CASE_NUMBER": 1,
    "COURT": 2,
    "DATE": 3,
    "GPE": 4,
    "JUDGE": 5,
    "ORG": 6,
    "OTHER_PERSON": 7,
    "PETITIONER": 8,
    "PRECEDENT": 9,
    "PROVISION": 10,
    "RESPONDENT": 11,
    "STATUTE": 12,
    "WITNESS": 13,
}

idx_to_label = {
    0: "O",
    1: "CASE_NUMBER",
    2: "COURT",
    3: "DATE",
    4: "GPE",
    5: "JUDGE",
    6: "ORG",
    7: "OTHER_PERSON",
    8: "PETITIONER",
    9: "PRECEDENT",
    10: "PROVISION",
    11: "RESPONDENT",
    12: "STATUTE",
    13: "WITNESS",
}

word2vec_model = word2vec_model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True
)


def get_word_embeddings(tokens):
    embeddings = []
    for token in tokens:
        try:
            embedding = torch.tensor(word2vec_model[token])
        except KeyError:
            embedding = torch.zeros(word2vec_model.vector_size)
        embeddings.append(embedding)
    return torch.stack(embeddings)


# Create a PyTorch dataset and data loader
class SequenceTaggingDataset(Dataset):
    def __init__(self, tokens, labels, tokenizer):
        self.tokens = tokens
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_to_idx = tag_to_ix

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        labels = self.labels[idx]

        encoded_tokens = self.tokenizer(tokens)

        for i in range(80 - len(encoded_tokens)):
            encoded_tokens.append(self.tokenizer("O")[0])

        if(type(labels[0]) != int):
            for i, label_hx in enumerate(labels):
                if(type(label_hx) == int):
                    print("fuck you ", label_hx)
                    print(labels)
                    print(tokens)

                if label_hx.startswith("B_") or label_hx.startswith("I_"):
                    labels[i] = tag_to_ix[label_hx[2:]]
                else:
                    labels[i] = tag_to_ix[label_hx]

            for i in range(80 - len(labels)):
                labels.append(0)

        labels_out = torch.tensor(labels)
        encoded_tokens = torch.tensor(encoded_tokens)

        return encoded_tokens, labels_out


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


word_to_ix = {"<UNK>": 0}
for sentence, tags in zip(train_tokens, train_labels):
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)


def tokenizer(tokens):
    token_ids = []
    for token in tokens:
        try:
            token_ids.append(word_to_ix[token])
        except:
            token_ids.append(word_to_ix["<UNK>"])
    return token_ids


train_dataset = SequenceTaggingDataset(train_tokens, train_labels, tokenizer)
val_dataset = SequenceTaggingDataset(val_tokens, val_labels, tokenizer)

# Define your data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


class BiLSTMCRFSequenceTaggingModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMCRFSequenceTaggingModel, self).__init__()
        self.embedding = nn.Embedding(len(word_to_ix), embedding_dim)

        for word, idx in word_to_ix.items():
            if word in word2vec_model:
                self.embedding.weight.data[idx] = torch.tensor(word2vec_model[word])

        self.rnn = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, input_ids, labels = None):
        embeddings = self.embedding(input_ids)
        out, _ = self.rnn(embeddings)
        out = self.fc(self.dropout(out))
        if labels is not None:
            loss = -self.crf(out, labels)
            return loss
        else:
            predictions = self.crf.decode(out)
            return predictions

model = BiLSTMCRFSequenceTaggingModel(embedding_dim=300,
                                      hidden_dim=256,
                                      num_layers=1,
                                      num_classes=num_labels).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def run_epoch(run_name, model):
    # Initialize W&B
    wandb.login(key="cbecd600ce14e66bbbed0c7b4bb7fb317f48a47a", relogin=True)
    wandb.init(project="nlp_a2", name=run_name)
    wandb.watch(model)

    # Train and evaluate the model
    num_epochs = 50
    best_val_f1 = 0
    best_val_loss = 100
    epochs_without_improvement = 0
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
            predicted = np.array(outputs)
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
                predicted = np.array(outputs)
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
            torch.save(model.state_dict(), "final"+run_name + ".pt")

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


run_epoch("BiLSTM_CRF_word2vec", model)