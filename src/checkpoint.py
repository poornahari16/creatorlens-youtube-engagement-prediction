import os
import json

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "progress.json")


def save_checkpoint(category, keyword, total_records):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint = {
        "category": category,
        "keyword": keyword,
        "total_records": total_records
    }

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=4)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None

    with open(CHECKPOINT_FILE, "r") as f:
        return json.load(f)


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)