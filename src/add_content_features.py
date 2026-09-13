import os
import re
import numpy as np
import pandas as pd


INPUT_FILE = "data/splits/train.csv"
OUTPUT_DIR = "data/splits"


def tokenize(text):
    """Convert text into simple lowercase word tokens."""
    if pd.isna(text):
        return []

    text = str(text).lower()
    return re.findall(r"\b[a-z0-9]+\b", text)


def safe_ratio(a, b):
    """Calculate a ratio safely."""
    if b == 0:
        return 0.0
    return a / b


def add_content_features(df):
    df = df.copy()

    # ---------------------------------------------------------
    # Tokenize title, description and tags
    # ---------------------------------------------------------
    title_words = df["title"].apply(tokenize)
    description_words = df["description"].apply(tokenize)
    tag_words = df["tags"].apply(tokenize)

    # Convert lists to sets for overlap calculations
    title_sets = title_words.apply(set)
    description_sets = description_words.apply(set)
    tag_sets = tag_words.apply(set)

    # ---------------------------------------------------------
    # Title <-> Description relationship
    # ---------------------------------------------------------
    df["title_description_overlap_count"] = [
        len(t & d)
        for t, d in zip(title_sets, description_sets)
    ]

    df["title_description_overlap_ratio"] = [
        safe_ratio(len(t & d), len(t))
        for t, d in zip(title_sets, description_sets)
    ]

    # ---------------------------------------------------------
    # Title <-> Tags relationship
    # ---------------------------------------------------------
    df["title_tags_overlap_count"] = [
        len(t & g)
        for t, g in zip(title_sets, tag_sets)
    ]

    df["title_tags_overlap_ratio"] = [
        safe_ratio(len(t & g), len(t))
        for t, g in zip(title_sets, tag_sets)
    ]

    # ---------------------------------------------------------
    # Description <-> Tags relationship
    # ---------------------------------------------------------
    df["description_tags_overlap_count"] = [
        len(d & g)
        for d, g in zip(description_sets, tag_sets)
    ]

    df["description_tags_overlap_ratio"] = [
        safe_ratio(len(d & g), len(d))
        for d, g in zip(description_sets, tag_sets)
    ]

    # ---------------------------------------------------------
    # Title quality / composition
    # ---------------------------------------------------------
    df["title_unique_word_ratio"] = [
        safe_ratio(len(set(words)), len(words))
        for words in title_words
    ]

    df["title_avg_word_length"] = [
        np.mean([len(word) for word in words]) if words else 0.0
        for words in title_words
    ]

    df["title_long_word_count"] = [
        sum(len(word) >= 8 for word in words)
        for words in title_words
    ]

    # ---------------------------------------------------------
    # Description quality / composition
    # ---------------------------------------------------------
    df["description_unique_word_ratio"] = [
        safe_ratio(len(set(words)), len(words))
        for words in description_words
    ]

    df["description_avg_word_length"] = [
        np.mean([len(word) for word in words]) if words else 0.0
        for words in description_words
    ]

    df["description_long_word_count"] = [
        sum(len(word) >= 8 for word in words)
        for words in description_words
    ]

    # ---------------------------------------------------------
    # Relative content length
    # ---------------------------------------------------------
    df["description_to_title_length_ratio"] = [
        safe_ratio(len(d), len(t))
        for t, d in zip(title_words, description_words)
    ]

    df["description_to_title_word_ratio"] = [
        safe_ratio(len(d), len(t))
        for t, d in zip(title_words, description_words)
    ]

    # ---------------------------------------------------------
    # General content indicators
    # ---------------------------------------------------------
    df["total_content_word_count"] = [
        len(t) + len(d)
        for t, d in zip(title_words, description_words)
    ]

    df["total_unique_content_word_count"] = [
        len(set(t + d))
        for t, d in zip(title_words, description_words)
    ]

    # ---------------------------------------------------------
    # Remove temporary objects
    # ---------------------------------------------------------
    return df


def process_file(input_file, output_file):
    print(f"\nReading: {input_file}")

    df = pd.read_csv(input_file)

    print(f"Rows before: {len(df):,}")
    print(f"Columns before: {len(df.columns):,}")

    df = add_content_features(df)

    print(f"Rows after: {len(df):,}")
    print(f"Columns after: {len(df.columns):,}")

    new_features = [
        "title_description_overlap_count",
        "title_description_overlap_ratio",
        "title_tags_overlap_count",
        "title_tags_overlap_ratio",
        "description_tags_overlap_count",
        "description_tags_overlap_ratio",
        "title_unique_word_ratio",
        "title_avg_word_length",
        "title_long_word_count",
        "description_unique_word_ratio",
        "description_avg_word_length",
        "description_long_word_count",
        "description_to_title_length_ratio",
        "description_to_title_word_ratio",
        "total_content_word_count",
        "total_unique_content_word_count",
    ]

    print("\nNew features:")
    for feature in new_features:
        print(f"  - {feature}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":

    files = {
        "train": "data/splits/train.csv",
        "validation": "data/splits/validation.csv",
        "test": "data/splits/test.csv",
    }

    for split_name, input_file in files.items():

        output_file = f"data/splits/{split_name}_v2.csv"

        process_file(
            input_file=input_file,
            output_file=output_file
        )

    print("\n" + "=" * 60)
    print("CONTENT FEATURE ENGINEERING COMPLETE")
    print("=" * 60)