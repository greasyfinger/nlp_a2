import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from blueprint import run_epochs

num_labels = 27

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
embeddings = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased").embeddings

class RNNSequenceTaggingModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes):
        super(RNNSequenceTaggingModel, self).__init__()
        self.embedding = embeddings
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        out, _ = self.rnn(embeddings)
        out = self.fc(self.dropout(out))
        return out


model = RNNSequenceTaggingModel(768, 256, 1, num_labels)

run_epochs(model, tokenizer, "RNN-Legal_Bert_1")
