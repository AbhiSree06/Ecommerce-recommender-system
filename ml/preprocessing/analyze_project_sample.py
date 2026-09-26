import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical_interactions.csv"
)

df = pd.read_csv(INPUT_FILE)

print("=" * 65)
print("PROJECT SAMPLE ANALYSIS")
print("=" * 65)

print(f"\nUsers        : {df['user_id'].nunique():,}")
print(f"Products     : {df['product_id'].nunique():,}")
print(f"Interactions : {len(df):,}")

# --------------------------------------------------
# Product interaction counts
# --------------------------------------------------

product_counts = df["product_id"].value_counts()

print("\nPRODUCT FREQUENCY")
print("-" * 65)

for threshold in [1, 2, 3, 5, 10, 20, 50]:
    count = (product_counts >= threshold).sum()

    print(
        f"Products with >= {threshold:2} interactions : "
        f"{count:,}"
    )

# --------------------------------------------------
# How many interactions survive each threshold?
# --------------------------------------------------

print("\nINTERACTIONS RETAINED")
print("-" * 65)

for threshold in [2, 3, 5, 10]:

    valid_products = product_counts[
        product_counts >= threshold
    ].index

    filtered = df[
        df["product_id"].isin(valid_products)
    ]

    print(
        f"Product threshold >= {threshold:2} : "
        f"{len(filtered):,} interactions | "
        f"{filtered['product_id'].nunique():,} products | "
        f"{filtered['user_id'].nunique():,} users"
    )

# --------------------------------------------------
# User survival after product filtering
# --------------------------------------------------

print("\nUSERS RETAINING ENOUGH INTERACTIONS")
print("-" * 65)

for threshold in [2, 3, 5, 10]:

    valid_products = product_counts[
        product_counts >= threshold
    ].index

    filtered = df[
        df["product_id"].isin(valid_products)
    ]

    user_counts = filtered["user_id"].value_counts()

    users_2 = (user_counts >= 2).sum()
    users_3 = (user_counts >= 3).sum()
    users_5 = (user_counts >= 5).sum()

    print(f"\nProduct threshold >= {threshold}")

    print(f"Users with >= 2 interactions : {users_2:,}")
    print(f"Users with >= 3 interactions : {users_3:,}")
    print(f"Users with >= 5 interactions : {users_5:,}")

# --------------------------------------------------
# Most popular products
# --------------------------------------------------

print("\nTOP 20 PRODUCTS")
print("-" * 65)

print(product_counts.head(20))