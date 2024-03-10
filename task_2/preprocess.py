import json
import os
import shutil

# changing current working directory
os.chdir("task_2")


def bio_chunking(text, aspects):

    tokens = text.split()
    labels = ["O"] * len(tokens)

    for aspect in aspects:
        start = aspect["from"]
        end = aspect["to"]
        labels[start] = "B"
        for i in range(start + 1, end):
            labels[i] = "I"

    return labels


def tag_json(input_file, output_file):
    output = {}

    with open(input_file, "r") as f:
        data = json.load(f)

    for i, entry in enumerate(data, start=1):
        text = entry["raw_words"]
        aspects = entry["aspects"]

        labels = bio_chunking(text, aspects)

        output[str(i)] = {"text": text, "labels": labels}

    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    # Create or recreate output folder
    bio_output_folder = "dataset"
    if os.path.exists(bio_output_folder):
        shutil.rmtree(bio_output_folder)
    os.makedirs(bio_output_folder)

    tag_json(
        "raw_dataset/train_set.json", os.path.join(bio_output_folder, "train_set.json")
    )
    tag_json(
        "raw_dataset/val_set.json", os.path.join(bio_output_folder, "val_set.json")
    )
    tag_json(
        "raw_dataset/test_set.json", os.path.join(bio_output_folder, "test_set.json")
    )
