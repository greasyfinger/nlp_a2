import json
import random
import os
import shutil
from collections import Counter
from sklearn.model_selection import train_test_split

# Load the training data
with open("train_data.json", "r") as f:
    data = json.load(f)

# Get all labels from the data
all_labels = [
    label
    for entry in data
    for annotation in entry["annotations"][0]["result"]
    for label in annotation["value"]["labels"]
]

# Count the occurrences of each label
label_counts = Counter(all_labels)

# Print the unique labels
print("Unique labels in the training data:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

# Shuffle the data randomly
random.shuffle(data)

# Split the data into training and validation sets with stratified sampling
train_data, val_data = train_test_split(
    data, test_size=0.15, random_state=42
)
# train_data, val_data = train_test_split(
#     data, test_size=0.15, stratify=all_labels, random_state=42
# )

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
shutil.copy("test_data.json", os.path.join(dataset_folder, "test_set.json"))
