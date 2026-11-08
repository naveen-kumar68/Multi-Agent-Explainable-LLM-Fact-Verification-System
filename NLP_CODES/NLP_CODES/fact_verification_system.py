
# BERT FACT VERIFICATION WITH RETRIEVED EVIDENCE


# PURPOSE:
# 1. Load processed FEVER train/dev data
# 2. Load FAISS retrieval system
# 3. Retrieve top evidence for each claim
# 4. Combine claim + evidence
# 5. Train transformer verifier
# 6. Predict SUPPORTS / REFUTES / NEI


import json
import pickle
import numpy as np
import faiss
import torch

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset

from sklearn.metrics import accuracy_score, f1_score




TRAIN_FILE = r"C:\Users\rishi\Downloads\NLP DATASET\processed_data\train_processed.json"

DEV_FILE = r"C:\Users\rishi\Downloads\NLP DATASET\processed_data\dev_processed.json"

FAISS_INDEX_PATH = r"C:\Users\rishi\Downloads\NLP DATASET\mini_faiss.index"

METADATA_PATH = r"C:\Users\rishi\Downloads\NLP DATASET\mini_wiki_metadata.pkl"




print("\nLoading SBERT retrieval model...")

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')



print("\nLoading FAISS index...")

index = faiss.read_index(FAISS_INDEX_PATH)

print("FAISS loaded successfully.")




print("\nLoading wiki metadata...")

with open(METADATA_PATH, "rb") as f:
    wiki_metadata = pickle.load(f)

print("Metadata loaded successfully.")




def load_data(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


print("\nLoading FEVER data...")

train_data = load_data(TRAIN_FILE)

dev_data = load_data(DEV_FILE)

print("Train samples:", len(train_data))

print("Dev samples:", len(dev_data))



def retrieve_evidence(claim, top_k=1):

    query_embedding = sbert_model.encode([claim])

    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    evidence_sentences = []

    for idx in indices[0]:

        evidence_sentences.append(
            wiki_metadata[idx]["text"]
        )

    return " ".join(evidence_sentences)



def prepare_dataset(data, limit=None):

    texts = []
    labels = []

    print("\nPreparing claim-evidence pairs...")

    for i, item in enumerate(tqdm(data)):

        claim = item["claim"]

        label = item["label"]

        # retrieve evidence
        evidence = retrieve_evidence(claim)

        # combine claim + evidence
        combined_text = claim + " [SEP] " + evidence

        texts.append(combined_text)

        labels.append(label)

        if limit and i >= limit:
            break

    return texts, labels



train_texts, train_labels = prepare_dataset(
    train_data,
    limit=3000
)

dev_texts, dev_labels = prepare_dataset(
    dev_data,
    limit=500
)


print("\nLoading BERT tokenizer...")

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)




def tokenize(batch):

    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )




train_dataset = Dataset.from_dict({

    "text": train_texts,

    "label": train_labels
})

dev_dataset = Dataset.from_dict({

    "text": dev_texts,

    "label": dev_labels
})


# tokenize
train_dataset = train_dataset.map(tokenize, batched=True)

dev_dataset = dev_dataset.map(tokenize, batched=True)



train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

dev_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)



print("\nLoading BERT verifier...")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

print("BERT loaded successfully.")



def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    f1 = f1_score(labels, predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "f1": f1
    }




training_args = TrainingArguments(

    output_dir="./bert_verifier",

    logging_dir="./logs",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=2,

    weight_decay=0.01,

    logging_steps=50
)


trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=dev_dataset,

    compute_metrics=compute_metrics
)



print("\nStarting BERT training...")

trainer.train()

print("\nTraining completed.")



print("\nEvaluating model...")

results = trainer.evaluate()

print("\nEvaluation Results:\n")

print(results)



MODEL_SAVE_PATH = r"C:\Users\rishi\Downloads\NLP DATASET\bert_fact_verifier"

trainer.save_model(MODEL_SAVE_PATH)

tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("\nModel saved successfully.")



label_map_reverse = {

    0: "SUPPORTS",

    1: "REFUTES",

    2: "NOT ENOUGH INFO"
}

print("CUSTOM CLAIM TEST")


while True:

    claim = input("\nEnter claim (or exit): ")

    if claim.lower() == "exit":
        break

    # retrieve evidence
    evidence = retrieve_evidence(claim)

    print("\nRetrieved Evidence:\n")

    print(evidence)

    # combine
    combined = claim + " [SEP] " + evidence

    # tokenize
    inputs = tokenizer(
        combined,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    # predict
    with torch.no_grad():

        outputs = model(**inputs)

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    print("\nPrediction:")

    print(label_map_reverse[prediction])