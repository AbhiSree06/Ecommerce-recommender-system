import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical_interactions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "interactions.csv"
)

MIN_PRODUCT_INTERACTIONS = 3
MIN_USER_INTERACTIONS = 3

df = pd.read_csv(INPUT_FILE)

print("=" * 65)
print("ITERATIVE INTERACTION FILTERING")
print("=" * 65)

print("\nBefore filtering:")
print(f"Users        : {df['user_id'].nunique():,}")
print(f"Products     : {df['product_id'].nunique():,}")
print(f"Interactions : {len(df):,}")

iteration = 0

while True:

    iteration += 1

    old_users = df["user_id"].nunique()
    old_products = df["product_id"].nunique()
    old_interactions = len(df)

    # --------------------------------------------
    # Remove sparse products
    # --------------------------------------------

    product_counts = df["product_id"].value_counts()

    valid_products = product_counts[
        product_counts >= MIN_PRODUCT_INTERACTIONS
    ].index

    df = df[
        df["product_id"].isin(valid_products)
    ]

    # --------------------------------------------
    # Remove users with insufficient history
    # --------------------------------------------

    user_counts = df["user_id"].value_counts()

    valid_users = user_counts[
        user_counts >= MIN_USER_INTERACTIONS
    ].index

    df = df[
        df["user_id"].isin(valid_users)
    ]

    new_users = df["user_id"].nunique()
    new_products = df["product_id"].nunique()
    new_interactions = len(df)

    print(f"\nIteration {iteration}")
    print("-" * 40)
    print(f"Users        : {new_users:,}")
    print(f"Products     : {new_products:,}")
    print(f"Interactions : {new_interactions:,}")

    # Stop when nothing changes
    if (
        old_users == new_users
        and old_products == new_products
        and old_interactions == new_interactions
    ):
        break


# --------------------------------------------
# Sort chronologically
# --------------------------------------------

df = df.sort_values(
    ["user_id", "timestamp"]
).reset_index(drop=True)


# --------------------------------------------
# Final validation
# --------------------------------------------

user_counts = df["user_id"].value_counts()
product_counts = df["product_id"].value_counts()

assert user_counts.min() >= MIN_USER_INTERACTIONS
assert product_counts.min() >= MIN_PRODUCT_INTERACTIONS


# --------------------------------------------
# Save
# --------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 65)
print("FINAL INTERACTION DATASET")
print("=" * 65)

print(f"Users                    : {df['user_id'].nunique():,}")
print(f"Products                 : {df['product_id'].nunique():,}")
print(f"Interactions             : {len(df):,}")
print(f"Minimum interactions/user: {user_counts.min()}")
print(f"Maximum interactions/user: {user_counts.max()}")
print(f"Average interactions/user: {len(df) / df['user_id'].nunique():.2f}")

print(
    f"Minimum interactions/product: "
    f"{product_counts.min()}"
)

print(
    f"Maximum interactions/product: "
    f"{product_counts.max()}"
)

print(f"\nSaved to:\n{OUTPUT_FILE}")