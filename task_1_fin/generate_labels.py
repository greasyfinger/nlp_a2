import re
import json
import os
import shutil
import re

def bio_chunking(text, annotations):
    # Initialize tokens and labels
    tokens = []
    labels = []

    # Process each annotation
    last_end = 0
    for annotation in annotations:
        start = annotation["value"]["start"]
        end = annotation["value"]["end"]
        labels_list = annotation["value"]["labels"]

        # Assign "O" labels to tokens between last_end and current start
        remaining_text = text[last_end:start]
        remaining_words = re.findall(r'\S+', remaining_text)
        for word in remaining_words:
            tokens.append(word)
            labels.append("O")

        # Extract the word associated with the annotation
        word = text[start:end]

        # Tokenize the word based on non-word characters (excluding spaces)
        word_tokens = re.findall(r'\S+', word)

        # Assign "B" label to the first token and "I" label to subsequent tokens
        if word_tokens:
            tokens.append(word_tokens[0])
            labels.append("B_" + labels_list[0])
            for i in range(1, len(word_tokens)):
                tokens.append(word_tokens[i])
                labels.append("I_" + labels_list[0])

        last_end = end

    # Assign "O" labels to remaining tokens after the last annotation
    remaining_text = text[last_end:]
    remaining_words = re.findall(r'\S+', remaining_text)
    for word in remaining_words:
        tokens.append(word)
        labels.append("O")

    return tokens, labels


def tag_json(input_file, output_file, token_file):
    output = {}
    token_output = {}

    # Load and process the training data
    with open(input_file, "r") as f:
        data = json.load(f)

    for entry in data:
        case_id = entry["id"]
        text = entry["data"]["text"]
        annotations = entry["annotations"][0]["result"]

        tokens, labels = bio_chunking(text, annotations)

        output[case_id] = {"text": text, "labels": labels}
        token_output[case_id] = {"text": text, "tokens": tokens}

    with open(token_file, "w") as f:
        json.dump(token_output, f, indent=4)

    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    # Create or recreate the output folders
    bio_output_folder = "label_dataset"
    token_output_folder = "token_dataset"
    if os.path.exists(bio_output_folder):
        shutil.rmtree(bio_output_folder)
    if os.path.exists(token_output_folder):
        shutil.rmtree(token_output_folder)
    os.makedirs(bio_output_folder)
    os.makedirs(token_output_folder)

    tag_json(
        "dataset/train_set.json", 
        os.path.join(bio_output_folder, "train_set.json"),
        os.path.join(token_output_folder, "train_set.json")
    )
    tag_json(
        "dataset/val_set.json", 
        os.path.join(bio_output_folder, "val_set.json"),
        os.path.join(token_output_folder, "val_set.json")
    )
    tag_json(
        "dataset/test_set.json", 
        os.path.join(bio_output_folder, "test_set.json"),
        os.path.join(token_output_folder, "test_set.json")
    )

