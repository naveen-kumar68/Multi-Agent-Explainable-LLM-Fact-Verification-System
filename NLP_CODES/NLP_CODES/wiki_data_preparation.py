

import json
import os
import glob

from tqdm import tqdm

TRAIN_FILE = r"C:\Users\rishi\Downloads\NLP DATASET\processed_data\train_10000.json"

DEV_FILE = r"C:\Users\rishi\Downloads\NLP DATASET\processed_data\dev_processed.json"

WIKI_FOLDER = r"C:\Users\rishi\Downloads\NLP DATASET\wiki-pages (1)\wiki-pages"

OUTPUT_WIKI = r"C:\Users\rishi\Downloads\NLP DATASET\mini_wiki_train_dev.jsonl"



with open(TRAIN_FILE, "r", encoding="utf-8") as f:

    train_data = json.load(f)

print("Train Samples:",
      len(train_data))



with open(DEV_FILE, "r", encoding="utf-8") as f:

    dev_data = json.load(f)

print("Dev Samples:",
      len(dev_data))



required_pages = set()


def collect_pages(dataset):

    for item in dataset:

        if "evidence" not in item:
            continue

        for evidence in item["evidence"]:

            page = evidence["wiki_page"]

            required_pages.add(page)


collect_pages(train_data)

collect_pages(dev_data)

print("Total Required Pages:",
      len(required_pages))

wiki_files = glob.glob(

    os.path.join(
        WIKI_FOLDER,
        "*.jsonl"
    )
)

print("\nTotal Wiki Files Found:",
      len(wiki_files))


saved_pages = 0


with open(OUTPUT_WIKI, "w", encoding="utf-8") as outfile:

    for wiki_file in wiki_files:

        print(f"\nProcessing: {wiki_file}")

        with open(wiki_file, "r", encoding="utf-8") as infile:

            for line in tqdm(infile):

                data = json.loads(line)

                page_id = data["id"]

                if page_id in required_pages:

                    outfile.write(line)

                    saved_pages += 1

print("Required Pages:",
      len(required_pages))

print("Saved Wiki Pages:",
      saved_pages)

print("\nMini Wiki File Created:")

print(OUTPUT_WIKI)