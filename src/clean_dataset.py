import pandas as pd
import sys


def clean_dataset(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["Text", "Label"])

    initial_count = len(df)
    df_cleaned = df.drop_duplicates(subset=["Text"], keep="first")
    final_count = len(df_cleaned)
    removed_count = initial_count - final_count

    df_cleaned.to_csv(file_path, sep="\t", index=False, header=False)
    print(f"Removed {removed_count} duplicate rows.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_dataset.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    clean_dataset(file_path)
