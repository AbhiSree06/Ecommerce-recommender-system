import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "interactions_basic.csv"
)

user_counts = Counter()
product_counts = Counter()

total_rows = 0

print("Analyzing interaction frequencies...\n")

# Read in chunks instead of loading all 43.9M rows at once
for chunk in tqdm(
    pd.read_csv(
        INPUT_FILE,
        usecols=["user_id", "product_id"],
        chunksize=500_000
    ),
    desc="Reading chunks"
):

    user_counts.update(chunk["user_id"])
    product_counts.update(chunk["product_id"])

    total_rows += len(chunk)


print("\n" + "=" * 65)
print("INTERACTION FREQUENCY ANALYSIS")
print("=" * 65)

print(f"Total interactions : {total_rows:,}")
print(f"Unique users       : {len(user_counts):,}")
print(f"Unique products    : {len(product_counts):,}")

print("\nUSER ACTIVITY")
print("-" * 65)

for threshold in [2, 3, 5, 10, 20, 50]:
    count = sum(
        1 for value in user_counts.values()
        if value >= threshold
    )

    print(
        f"Users with >= {threshold:2} interactions : "
        f"{count:,}"
    )

print("\nPRODUCT ACTIVITY")
print("-" * 65)

for threshold in [2, 3, 5, 10, 20, 50]:
    count = sum(
        1 for value in product_counts.values()
        if value >= threshold
    )

    print(
        f"Products with >= {threshold:2} interactions : "
        f"{count:,}"
    )

print("\nMOST ACTIVE USERS")
print("-" * 65)

for user_id, count in user_counts.most_common(10):
    print(user_id, ":", count)

print("\nMOST INTERACTED PRODUCTS")
print("-" * 65)

for product_id, count in product_counts.most_common(10):
    print(product_id, ":", count)