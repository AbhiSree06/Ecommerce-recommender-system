import json
import csv
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "Electronics.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "interactions_basic.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

required_fields = [
    "user_id",
    "parent_asin",
    "rating",
    "timestamp",
    "verified_purchase"
]

total_records = 0
valid_records = 0
invalid_records = 0

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

    writer = csv.DictWriter(
        outfile,
        fieldnames=[
            "user_id",
            "product_id",
            "rating",
            "timestamp",
            "verified_purchase"
        ]
    )

    writer.writeheader()

    for line in tqdm(infile, desc="Processing interactions"):

        total_records += 1

        try:
            record = json.loads(line)

            # Check essential attributes
            if (
                not record.get("user_id")
                or not record.get("parent_asin")
                or record.get("timestamp") is None
                or record.get("rating") is None
            ):
                invalid_records += 1
                continue

            writer.writerow({
                "user_id": record["user_id"],
                "product_id": record["parent_asin"],
                "rating": record["rating"],
                "timestamp": record["timestamp"],
                "verified_purchase": record.get(
                    "verified_purchase", False
                )
            })

            valid_records += 1

        except (json.JSONDecodeError, TypeError):
            invalid_records += 1


print("\n" + "=" * 60)
print("INTERACTION PREPROCESSING COMPLETE")
print("=" * 60)

print(f"Total records read : {total_records:,}")
print(f"Valid records      : {valid_records:,}")
print(f"Invalid records    : {invalid_records:,}")
print(f"\nSaved to:\n{OUTPUT_FILE}")