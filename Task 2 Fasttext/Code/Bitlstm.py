import json
import string
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from gensim.models import KeyedVectors

import tensorflow_addons as tfa
# Label mapping
label_mapping = {
    'O': 0, 
    'B': 1,  
    'I': 2   
}

# Clean text function
def clean_text(text):
    return text

# Preprocess data function
def preprocess_data(data):
    texts = []
    labels = []
    for case_id, case_data in data.items():
        text = case_data['text'].lower()
        labels_for_case = case_data['labels']  
        words = nltk.word_tokenize(text)
        for i, word in enumerate(words):
            if i < len(labels_for_case):
                label = labels_for_case[i]
                texts.append(word)
                labels.append(label_mapping[label])
    return texts, labels

# Load and preprocess train data
with open('/home/nalin21478/nlp_a2/task_2/dataset/train_bio.json', 'r') as f:
    train_data = json.load(f)
train_texts, train_labels = preprocess_data(train_data)

# Load and preprocess test data
with open('/home/nalin21478/nlp_a2/task_2/dataset/test_bio.json', 'r') as f:
    test_data = json.load(f)
test_texts, test_labels = preprocess_data(test_data)

# Load and preprocess validation data
with open('/home/nalin21478/nlp_a2/task_2/dataset/val_bio.json', 'r') as f:
    val_data = json.load(f)
val_texts, val_labels = preprocess_data(val_data)

# Tokenize texts and convert them to sequences of word indices
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)

# Pad sequences to 100
max_sequence_length = 100  
train_sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_sequence_length)
test_sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_sequence_length)
val_sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(val_sequences, maxlen=max_sequence_length)

# Load FastText word embeddings
import fasttext.util
import numpy as np
ft = fasttext.load_model('/home/nalin21478/nlp_a2/Task 2 Fasttext/Fasttext/cc.en.300.bin')

# %%
vocab_size = len(tokenizer.word_index) + 1




embedding_dim = 300  
weight_matrix = np.zeros((vocab_size, embedding_dim))


for word, i in tokenizer.word_index.items():

    embedding_vector = ft.get_word_vector(word)
    weight_matrix[i] = embedding_vector


# Convert integer labels to one-hot encoded format
y_train = tf.keras.utils.to_categorical(train_labels)
y_test = tf.keras.utils.to_categorical(test_labels)
y_val = tf.keras.utils.to_categorical(val_labels)

class BiLSTM_CRF(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[weight_matrix], trainable=False)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hidden_dim, return_sequences=True))
        self.dense = Dense(tagset_size)
        self.dropout = Dropout(0.5)
        self.crf = tfa.layers.CRF(tagset_size)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.bi_lstm(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.crf(x)
        return x

# Create BiLSTM_CRF model
hidden_dim = 128
tagset_size = 3  # Assuming 3 tags: 'O', 'B', 'I'
model = BiLSTM_CRF(vocab_size, embedding_dim, hidden_dim, tagset_size)
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow_addons.text import crf_log_likelihood, crf_decode

def custom_loss(y_true, y_pred):
    crf = y_pred[0]
    logits = y_pred[1]
    return -crf_log_likelihood(crf, y_true, logits)[0]

# Define your model architecture and layers here

# Compile the model using the custom loss function
model.compile(optimizer='adam', loss=custom_loss)



# Train the model
history = model.fit(train_sequences_padded, y_train, epochs=10, batch_size=32, validation_data=(val_sequences_padded, y_val))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_sequences_padded, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# %%
