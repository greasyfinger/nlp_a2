import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Set up the training and evaluation pipeline
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# change current directory
os.chdir("task_2_torch")


# Load and preprocess the data
def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    texts = []
    labels = []
    for entry in data.values():
        texts.append(entry["text"])
        labels.append(entry["labels"])

    return texts, labels


# Create a PyTorch dataset and data loader
class SequenceTaggingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        label_ids = []
        for label in labels:
            if label.startswith("B"):
                label_ids.append(1)
            elif label.startswith("I"):
                label_ids.append(2)
            else:
                label_ids.append(0)
        for _ in range(512 - len(labels)):
            label_ids.append(0)

        return encoded["input_ids"].squeeze(), torch.tensor(label_ids)


def get_data(tokenizer):

    train_texts, train_labels = load_data("tag_output/train_bio.json")
    val_texts, val_labels = load_data("tag_output/val_bio.json")

    train_dataset = SequenceTaggingDataset(train_texts, train_labels, tokenizer)
    val_dataset = SequenceTaggingDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


def get_model(model):

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    return model, optimizer, criterion


def run_epochs(model, tokenizer, run_name):

    train_loader, val_loader = get_data(tokenizer)
    model, optimizer, criterion = get_model(model)
    # Initialize W&B
    wandb.login(key="cbecd600ce14e66bbbed0c7b4bb7fb317f48a47a", relogin=True)
    wandb.init(project="nlp_a2", name=run_name)
    wandb.watch(model)

    # Train and evaluate the model
    num_epochs = 10
    best_val_f1 = 0
    best_val_loss = 100
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss = 0
        train_f1 = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 3), labels.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate F1-score
            predicted = outputs.argmax(dim=2).cpu().numpy()
            true = labels.cpu().numpy()
            train_f1 += f1_score(true.flatten(), predicted.flatten(), average="macro")

        model.eval()
        val_loss = 0
        val_f1 = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, 3), labels.view(-1))
                val_loss += loss.item()

                # Calculate F1-score
                predicted = outputs.argmax(dim=2).cpu().numpy()
                true = labels.cpu().numpy()
                val_f1 += f1_score(true.flatten(), predicted.flatten(), average="macro")

        # Log metrics to W&B
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_f1 /= len(train_loader)
        val_f1 /= len(val_loader)

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_f1": train_f1,
                "val_f1": val_f1,
            }
        )

        # Save the best model checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), run_name + ".pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Check if training should be stopped
        if epochs_without_improvement >= 3:
            print(f"Stopping early at epoch {epoch+1} due to no improvement.")
            break

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Train F1: {train_f1}, Val F1: {val_f1}"
        )

    wandb.finish()
