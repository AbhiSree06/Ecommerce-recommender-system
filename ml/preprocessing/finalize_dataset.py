import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERACTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "interactions.csv"
)

PRODUCTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "products.csv"
)

USERS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "users.csv"
)


print("=" * 65)
print("FINAL DATASET VALIDATION")
print("=" * 65)


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

interactions = pd.read_csv(INTERACTIONS_FILE)
products = pd.read_csv(PRODUCTS_FILE)


# --------------------------------------------------
# Basic counts
# --------------------------------------------------

print("\nINTERACTIONS")
print("-" * 65)

print(f"Rows            : {len(interactions):,}")
print(f"Unique users    : {interactions['user_id'].nunique():,}")
print(f"Unique products : {interactions['product_id'].nunique():,}")


print("\nPRODUCTS")
print("-" * 65)

print(f"Rows               : {len(products):,}")
print(f"Unique product IDs : {products['product_id'].nunique():,}")


# --------------------------------------------------
# Product alignment
# --------------------------------------------------

interaction_products = set(
    interactions["product_id"].astype(str)
)

metadata_products = set(
    products["product_id"].astype(str)
)

missing_metadata = (
    interaction_products - metadata_products
)

unused_metadata = (
    metadata_products - interaction_products
)


print("\nPRODUCT ALIGNMENT")
print("-" * 65)

print(
    f"Interaction products without metadata : "
    f"{len(missing_metadata):,}"
)

print(
    f"Metadata products without interactions : "
    f"{len(unused_metadata):,}"
)


# --------------------------------------------------
# Duplicate checks
# --------------------------------------------------

print("\nDUPLICATE CHECK")
print("-" * 65)

print(
    f"Duplicate interaction rows : "
    f"{interactions.duplicated().sum():,}"
)

print(
    f"Duplicate product IDs      : "
    f"{products['product_id'].duplicated().sum():,}"
)


# --------------------------------------------------
# Missing essential fields
# --------------------------------------------------

print("\nMISSING ESSENTIAL VALUES")
print("-" * 65)

for column in [
    "user_id",
    "product_id",
    "timestamp",
    "rating"
]:
    print(
        f"{column:15}: "
        f"{interactions[column].isna().sum():,}"
    )

print(
    f"{'product title':15}: "
    f"{products['title'].isna().sum():,}"
)

print(
    f"{'semantic_text':15}: "
    f"{products['semantic_text'].isna().sum():,}"
)


# --------------------------------------------------
# Chronological validation
# --------------------------------------------------

chronological = True

for _, group in interactions.groupby(
    "user_id",
    sort=False
):

    if not group["timestamp"].is_monotonic_increasing:
        chronological = False
        break


print("\nSEQUENCE VALIDATION")
print("-" * 65)

print(
    f"Interactions chronological per user : "
    f"{chronological}"
)


# --------------------------------------------------
# Create users.csv
# --------------------------------------------------

user_summary = (
    interactions
    .groupby("user_id")
    .agg(
        interaction_count=(
            "product_id",
            "size"
        ),
        unique_products=(
            "product_id",
            "nunique"
        ),
        average_rating=(
            "rating",
            "mean"
        ),
        first_interaction=(
            "timestamp",
            "min"
        ),
        last_interaction=(
            "timestamp",
            "max"
        )
    )
    .reset_index()
)


# Convert Amazon millisecond timestamps
# into readable datetime values

user_summary["first_interaction_date"] = (
    pd.to_datetime(
        user_summary["first_interaction"],
        unit="ms",
        errors="coerce"
    )
)

user_summary["last_interaction_date"] = (
    pd.to_datetime(
        user_summary["last_interaction"],
        unit="ms",
        errors="coerce"
    )
)


user_summary.to_csv(
    USERS_FILE,
    index=False
)


# --------------------------------------------------
# Assertions
# --------------------------------------------------

assert len(missing_metadata) == 0
assert len(unused_metadata) == 0

assert interactions.duplicated().sum() == 0
assert products["product_id"].duplicated().sum() == 0

assert interactions[
    [
        "user_id",
        "product_id",
        "timestamp",
        "rating"
    ]
].isna().sum().sum() == 0

assert products[
    [
        "product_id",
        "title",
        "semantic_text"
    ]
].isna().sum().sum() == 0

assert chronological is True


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\nUSERS DATASET")
print("-" * 65)

print(
    f"Users created : "
    f"{len(user_summary):,}"
)

print(
    f"Min interactions/user : "
    f"{user_summary['interaction_count'].min()}"
)

print(
    f"Max interactions/user : "
    f"{user_summary['interaction_count'].max()}"
)

print(
    f"Average interactions/user : "
    f"{user_summary['interaction_count'].mean():.2f}"
)


print("\n" + "=" * 65)
print("DATASET VALIDATION PASSED")
print("=" * 65)

print("\nFinal files:")

print(f"Interactions : {INTERACTIONS_FILE}")
print(f"Products     : {PRODUCTS_FILE}")
print(f"Users        : {USERS_FILE}")