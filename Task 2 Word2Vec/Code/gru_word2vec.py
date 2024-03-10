# %%
import json
import string
import nltk

label_mapping = {
    'O': 0, 
    'B': 1,  
    'I': 2   
}

def clean_text(text):
    return text


def preprocess_data(data):
    texts = []
    labels = []
    for case_id, case_data in data.items():
        text = case_data['text'].lower()
        labels_for_case = case_data['labels']  
        
        # Tokenize the text into words
        words = nltk.word_tokenize(text)
        
        for i, word in enumerate(words):
            # Ensure that the index is within the range of labels_for_case
            if i < len(labels_for_case):
                # Find the label for this word
                label = labels_for_case[i]
                
                # Append the word and its corresponding label
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



# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize texts and convert them to sequences of word indices
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)

# Convert texts to sequences of word indices
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)

# Pad sequences to ensure uniform length
max_sequence_length = 100  # choose a suitable maximum sequence length
train_sequences_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_sequences_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)
val_sequences_padded = pad_sequences(val_sequences, maxlen=max_sequence_length)


# %%
len(train_sequences_padded), len(test_sequences_padded), len(val_sequences_padded)

# %%
len(train_labels), len(test_labels), len(val_labels)

# %%
from gensim.models import KeyedVectors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense

import numpy as np

# Load Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format('/home/nalin21478/nlp_a2/GoogleNews-vectors-negative300.bin', binary=True)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 300  # Assuming you want 300-dimensional embeddings

# Initialize embedding matrix with zeros
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# Fill embedding matrix with Word2Vec embeddings for words in tokenizer's word index
for word, i in tokenizer.word_index.items():
    if word in word2vec_model:
        embedding_matrix[i] = word2vec_model[word]


# %%
from keras.utils import to_categorical

# Convert integer labels to one-hot encoded format
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)
y_val = to_categorical(val_labels)


# %%
y_train.shape, y_test.shape, y_val.shape

# %%

import tensorflow_addons as tfa
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense, Dropout
from keras.callbacks import EarlyStopping

# Define your model
from keras.callbacks import EarlyStopping

# Define your model
model = Sequential()

model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False))
model.add(Dropout(0.2)) 
model.add(GRU(units=64))
model.add(Dropout(0.2))  

# Add Dense output layer
model.add(Dense(units=3, activation='softmax'))

# Instantiate F1 Score metric
f1_score = tfa.metrics.F1Score(num_classes=3, average='macro')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_score])

# Print model summary
model.summary()

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# Fit the model with early stopping
history = model.fit(train_sequences_padded, y_train, epochs=20, batch_size=64, 
                    validation_data=(val_sequences_padded, y_val), callbacks=[early_stopping])

import matplotlib.pyplot as plt



# %%


# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/home/nalin21478/nlp_a2/Task 2 Word2Vec/Plots/loss_plot_GRU_glove.png')  # Save the plot as an image
plt.show()

# Plot training and validation F1 score
plt.figure(figsize=(12, 6))
plt.plot(history.history['f1_score'], label='Training F1 Score')
plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
plt.title('Training and Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.savefig('/home/nalin21478/nlp_a2/Task 2 Word2Vec/Plots/f1_score_plot_GRU_glove.png')  # Save the plot as an image
plt.show()

# Evaluate the model on the test data
_, test_accuracy, test_f1_score = model.evaluate(test_sequences_padded, y_test)

print("Test Accuracy:", test_accuracy)
print("Test F1 Score:", test_f1_score)

# Save the history object as a JSON file
with open('/home/nalin21478/nlp_a2/Task 2 Word2Vec/History/history_GRU_glove.json', 'w') as f:
    json.dump(history.history, f)

import pickle
with open("/home/nalin21478/nlp_a2/Task 2 Word2Vec/Models/t2_GRU_Glove.pkl", "wb") as file:
    pickle.dump(model, file)


# %%


# %%



