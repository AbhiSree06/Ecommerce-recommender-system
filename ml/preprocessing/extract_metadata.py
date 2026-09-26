import json
import csv
from pathlib import Path
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERACTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "interactions.csv"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "meta_Electronics.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "products_basic.csv"
)


# --------------------------------------------------
# STEP 1: Get product IDs from final interactions
# --------------------------------------------------

print("=" * 65)
print("EXTRACTING PRODUCT METADATA")
print("=" * 65)

print("\nLoading required product IDs...")

interactions = pd.read_csv(
    INTERACTIONS_FILE,
    usecols=["product_id"]
)

required_products = set(
    interactions["product_id"]
    .dropna()
    .astype(str)
    .unique()
)

print(
    f"Products required from metadata: "
    f"{len(required_products):,}"
)


# --------------------------------------------------
# STEP 2: Scan metadata
# --------------------------------------------------

matched_records = []
matched_ids = set()

total_metadata_records = 0

print("\nScanning metadata file...")

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as infile:

    for line in tqdm(
        infile,
        desc="Reading metadata"
    ):

        total_metadata_records += 1

        try:
            record = json.loads(line)

        except json.JSONDecodeError:
            continue

        product_id = record.get("parent_asin")

        if product_id not in required_products:
            continue

        # Avoid duplicate metadata rows
        if product_id in matched_ids:
            continue

        matched_records.append({
            "product_id": product_id,

            "title": record.get("title"),

            "main_category":
                record.get("main_category"),

            "categories":
                json.dumps(
                    record.get("categories", []),
                    ensure_ascii=False
                ),

            "features":
                json.dumps(
                    record.get("features", []),
                    ensure_ascii=False
                ),

            "description":
                json.dumps(
                    record.get("description", []),
                    ensure_ascii=False
                ),

            "price":
                record.get("price"),

            "average_rating":
                record.get("average_rating"),

            "rating_count":
                record.get("rating_number"),

            "store":
                record.get("store"),

            "images":
                json.dumps(
                    record.get("images", []),
                    ensure_ascii=False
                ),

            "details":
                json.dumps(
                    record.get("details", {}),
                    ensure_ascii=False
                )
        })

        matched_ids.add(product_id)

        # Stop early if all required products were found
        if len(matched_ids) == len(required_products):
            break


# --------------------------------------------------
# STEP 3: Save extracted metadata
# --------------------------------------------------

products_df = pd.DataFrame(matched_records)

products_df.to_csv(
    OUTPUT_FILE,
    index=False,
    quoting=csv.QUOTE_MINIMAL
)


# --------------------------------------------------
# STEP 4: Check missing products
# --------------------------------------------------

missing_products = (
    required_products - matched_ids
)


print("\n" + "=" * 65)
print("METADATA EXTRACTION COMPLETE")
print("=" * 65)

print(
    f"Metadata records scanned : "
    f"{total_metadata_records:,}"
)

print(
    f"Products required        : "
    f"{len(required_products):,}"
)

print(
    f"Products matched         : "
    f"{len(matched_ids):,}"
)

print(
    f"Products missing         : "
    f"{len(missing_products):,}"
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)


if missing_products:

    missing_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "missing_product_ids.txt"
    )

    with open(
        missing_file,
        "w",
        encoding="utf-8"
    ) as f:

        for product_id in sorted(missing_products):
            f.write(product_id + "\n")

    print(
        f"\nMissing product IDs saved to:\n"
        f"{missing_file}"
    )