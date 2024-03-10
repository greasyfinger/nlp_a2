import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from blueprint import run_epochs

num_labels = 27

# BERT tokenizer and embedding layer
print("RUNNING ON LEGAL BERT EMBEDDINGS")
model_name = "nlpaueb/legal-bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
embeddings = AutoModel.from_pretrained(model_name).embeddings

# RNN model
print("RUNNING RNN")
class RNNSequenceTaggingModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes):
        super(RNNSequenceTaggingModel, self).__init__()
        self.embedding = embeddings
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        out, _ = self.rnn(embeddings)
        out = self.fc(self.dropout(out))
        return out


model = RNNSequenceTaggingModel(768, 256, 1, num_labels)
run_epochs(model, tokenizer, "RNN-Legal_Bert_1")


# LSTM model
print("RUNNING LSTM")
class LSTMSequenceTaggingModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes):
        super(LSTMSequenceTaggingModel, self).__init__()
        self.embedding = embeddings
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        out, _ = self.lstm(embeddings)
        out = self.fc(self.dropout(out))
        return out


model = LSTMSequenceTaggingModel(768, 256, 1, num_labels)
run_epochs(model, tokenizer, "LSTM-Legal_Bert_1")


# GRU Model
print("RUNNING GRU")
class GRUSequenceTaggingModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes):
        super(GRUSequenceTaggingModel, self).__init__()
        self.embedding = embeddings
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        out, _ = self.gru(embeddings)
        out = self.fc(self.dropout(out))
        return out


model = GRUSequenceTaggingModel(768, 256, 1, num_labels)
run_epochs(model, tokenizer, "GRU-Legal_Bert_1")
