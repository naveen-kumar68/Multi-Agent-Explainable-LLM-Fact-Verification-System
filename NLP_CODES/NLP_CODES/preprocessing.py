

import os
import json
import re
from tqdm import tqdm



TRAIN_PATH = r"C:\Users\rishi\Downloads\NLP DATASET\train.jsonl"

DEV_PATH = r"C:\Users\rishi\Downloads\NLP DATASET\shared_task_dev.jsonl"

# OUTPUT FOLDER
OUTPUT_DIR = r"C:\Users\rishi\Downloads\NLP DATASET\processed_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


LABEL_MAP = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2
}

REVERSE_LABEL_MAP = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT ENOUGH INFO"
}

def clean_text(text):

    # lowercase
    text = text.lower()

    # remove special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def extract_evidence(evidence_list):

    extracted = []

    for evidence_group in evidence_list:

        for item in evidence_group:

            # item format:
            # [annotation_id, evidence_id, wiki_page, sentence_id]

            if len(item) >= 4:

                wiki_page = item[2]
                sentence_id = item[3]

                if wiki_page is not None:

                    extracted.append({
                        "wiki_page": wiki_page,
                        "sentence_id": sentence_id
                    })

    return extracted



def load_fever_dataset(file_path, limit=None):

    claims = []
    labels = []
    evidence_refs = []

    print(f"\nLoading Dataset: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:

        for i, line in enumerate(tqdm(f)):

            data = json.loads(line)

            claim = clean_text(data["claim"])

            label = LABEL_MAP[data["label"]]

            evidence = extract_evidence(data["evidence"])

            claims.append(claim)
            labels.append(label)
            evidence_refs.append(evidence)

            # limit for testing
            if limit and i >= limit:
                break

    return claims, labels, evidence_refs

def save_processed_data(claims, labels, evidence_refs, output_file):

    processed_data = []

    for claim, label, evidence in zip(claims, labels, evidence_refs):

        processed_data.append({

            "claim": claim,

            "label": label,

            "label_name": REVERSE_LABEL_MAP[label],

            "evidence": evidence

        })

    with open(output_file, "w", encoding="utf-8") as f:

        json.dump(processed_data, f, indent=2)

    print(f"\nSaved File: {output_file}")


if __name__ == "__main__":


    train_claims, train_labels, train_evidence = load_fever_dataset(
        TRAIN_PATH,
        limit=5000
    )


    TRAIN_OUTPUT = os.path.join(
        OUTPUT_DIR,
        "train_processed.json"
    )

    save_processed_data(
        train_claims,
        train_labels,
        train_evidence,
        TRAIN_OUTPUT
    )



    dev_claims, dev_labels, dev_evidence = load_fever_dataset(
        DEV_PATH,
        limit=1000
    )


    DEV_OUTPUT = os.path.join(
        OUTPUT_DIR,
        "dev_processed.json"
    )

    save_processed_data(
        dev_claims,
        dev_labels,
        dev_evidence,
        DEV_OUTPUT
    )


    print("SAMPLE OUTPUT")


    for i in range(5):

        print(f"CLAIM:\n{train_claims[i]}\n")

        print(f"LABEL:\n{REVERSE_LABEL_MAP[train_labels[i]]}\n")

        print(f"EVIDENCE REFERENCES:\n{train_evidence[i]}\n")

        print("=" * 70)