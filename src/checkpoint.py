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

    print(f"Saving checkpoint to: {CHECKPOINT_FILE}")
    print(checkpoint)

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=4)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None

    # File exists but is empty
    if os.path.getsize(CHECKPOINT_FILE) == 0:
        return None

    try:
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("Checkpoint file is corrupted. Starting fresh.")
        return None


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)