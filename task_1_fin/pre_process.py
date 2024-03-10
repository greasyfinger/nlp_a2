import json
import random
import os
import shutil
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the training data
with open("raw_dataset/train_data.json", "r") as f:
    data = json.load(f)

# Split the data into training and validation sets with stratified sampling
train_data, val_data = train_test_split(
    data, test_size=0.15, random_state=929
)

# Create or recreate the dataset folder
dataset_folder = "dataset"
if os.path.exists(dataset_folder):
    shutil.rmtree(dataset_folder)
os.makedirs(dataset_folder)

# Save the training set as train_set.json
with open(os.path.join(dataset_folder, "train_set.json"), "w") as f:
    json.dump(train_data, f, indent=4)

# Save the validation set as val_set.json
with open(os.path.join(dataset_folder, "val_set.json"), "w") as f:
    json.dump(val_data, f, indent=4)

# Copy the test_data.json as test_set.json
shutil.copy("raw_dataset/test_data.json", os.path.join(dataset_folder, "test_set.json"))



# Get all labels from the data
all_labels = [label for entry in data for annotation in entry["annotations"][0]["result"] for label in annotation["value"]["labels"]]

# Get all labels from the training data
train_labels = [label for entry in train_data for annotation in entry["annotations"][0]["result"] for label in annotation["value"]["labels"]]

# Get all labels from the validation data
val_labels = [label for entry in val_data for annotation in entry["annotations"][0]["result"] for label in annotation["value"]["labels"]]

# Count the occurrences of each label in the training and validation data separately
label_counts_all = Counter(all_labels)
label_counts_train = Counter(train_labels)
label_counts_val = Counter(val_labels)

# Sort labels by frequency
sorted_labels_all = sorted(label_counts_all.items(), key=lambda x: x[0])
sorted_labels_train = sorted(label_counts_train.items(), key=lambda x: x[0])
sorted_labels_val = sorted(label_counts_val.items(), key=lambda x: x[0])

# Plot the distribution of labels in the entire dataset
plt.figure(figsize=(12, 6))
plt.bar([label[0] for label in sorted_labels_all], [count[1] for count in sorted_labels_all], color='green')
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Distribution of Labels in Entire Dataset')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('entire_dataset_label_distribution.png')
plt.close()

# Plot the distribution of labels in the training set
plt.figure(figsize=(12, 6))
plt.bar([label[0] for label in sorted_labels_train], [count[1] for count in sorted_labels_train], color='skyblue')
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Distribution of Labels in Training Set')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('train_label_distribution.png')
plt.close()

# Plot the distribution of labels in the validation set
plt.figure(figsize=(12, 6))
plt.bar([label[0] for label in sorted_labels_val], [count[1] for count in sorted_labels_val], color='orange')
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Distribution of Labels in Validation Set')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('val_label_distribution.png')
plt.close()

print("distribution saved as entire_dataset_label_distribution.png, train_label_distribution.png, and val_label_distribution.png")