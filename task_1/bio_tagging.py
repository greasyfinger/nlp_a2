import re
import json
import os
import shutil
from typing import Dict, List

# changing current working directory
os.chdir("task_1")


def bio_chunking(text: str, annotations: List[Dict]) -> List[str]:

    clean_text = re.sub(
        r"^[\s,.;()\[\]]+|[\s,.;()\[\]]+$", "", re.sub(r"\s+", " ", text)
    ).strip()
    
    tokens = clean_text.split()
    labels = ["O"] * len(tokens)

    for annotation in annotations:
        start = annotation["value"]["start"]
        end = annotation["value"]["end"]
        labels_list = annotation["value"]["labels"]

        start_token_idx = clean_text[:start].count(" ")
        end_token_idx = clean_text[:end].count(" ")

        for label in labels_list:
            labels[start_token_idx] = f"B_{label}"
            for i in range(start_token_idx + 1, end_token_idx):
                labels[i] = f"I_{label}"

    return labels


def tag_json(input_file, output_file):
    output = {}

    # Load and process the training data
    with open(input_file, "r") as f:
        data = json.load(f)

    for entry in data:
        case_id = entry["id"]
        text = entry["data"]["text"]
        annotations = entry["annotations"][0]["result"]

        labels = bio_chunking(text, annotations)

        output[case_id] = {"text": text, "labels": labels}

    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    # Create or recreate the output folder
    bio_output_folder = "tag_output"
    if os.path.exists(bio_output_folder):
        shutil.rmtree(bio_output_folder)
    os.makedirs(bio_output_folder)

    tag_json(
        "dataset/train_set.json", os.path.join(bio_output_folder, "train_bio.json")
    )
    tag_json("dataset/val_set.json", os.path.join(bio_output_folder, "val_bio.json"))
    tag_json("dataset/test_set.json", os.path.join(bio_output_folder, "test_bio.json"))
