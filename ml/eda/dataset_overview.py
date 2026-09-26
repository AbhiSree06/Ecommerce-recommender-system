import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "eda" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTERACTIONS_FILE = DATA_DIR / "interactions.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"
USERS_FILE = DATA_DIR / "users.csv"


# --------------------------------------------------
# Load final Stage 3 datasets
# --------------------------------------------------

print("=" * 70)
print("STAGE 4 — EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nLoading datasets...")

interactions = pd.read_csv(INTERACTIONS_FILE)
products = pd.read_csv(PRODUCTS_FILE)
users = pd.read_csv(USERS_FILE)

print("Datasets loaded successfully.")


# --------------------------------------------------
# Overall dataset size
# --------------------------------------------------

print("\n" + "=" * 70)
print("1. DATASET OVERVIEW")
print("=" * 70)

print(f"\nUsers        : {interactions['user_id'].nunique():,}")
print(f"Products     : {interactions['product_id'].nunique():,}")
print(f"Interactions : {len(interactions):,}")


# --------------------------------------------------
# Interaction statistics
# --------------------------------------------------

user_activity = interactions.groupby("user_id").size()
product_activity = interactions.groupby("product_id").size()

print("\nUSER ACTIVITY")
print("-" * 70)

print(f"Minimum interactions/user : {user_activity.min()}")
print(f"Maximum interactions/user : {user_activity.max()}")
print(f"Mean interactions/user    : {user_activity.mean():.2f}")
print(f"Median interactions/user  : {user_activity.median():.2f}")


print("\nPRODUCT ACTIVITY")
print("-" * 70)

print(f"Minimum interactions/product : {product_activity.min()}")
print(f"Maximum interactions/product : {product_activity.max()}")
print(f"Mean interactions/product    : {product_activity.mean():.2f}")
print(f"Median interactions/product  : {product_activity.median():.2f}")


# --------------------------------------------------
# Ratings
# --------------------------------------------------

print("\nRATING STATISTICS")
print("-" * 70)

print(f"Minimum rating : {interactions['rating'].min():.1f}")
print(f"Maximum rating : {interactions['rating'].max():.1f}")
print(f"Mean rating    : {interactions['rating'].mean():.2f}")
print(f"Median rating  : {interactions['rating'].median():.2f}")


# --------------------------------------------------
# Product information
# --------------------------------------------------

print("\nPRODUCT METADATA")
print("-" * 70)

print(
    f"Products with category : "
    f"{products['categories'].notna().sum():,}"
)

print(
    f"Products with brand    : "
    f"{products['brand'].notna().sum():,}"
)

print(
    f"Products with price    : "
    f"{products['price'].notna().sum():,}"
)

print(
    f"Products with image    : "
    f"{products['image_url'].notna().sum():,}"
)

print(
    f"Products with features : "
    f"{products['features'].notna().sum():,}"
)

print(
    f"Products with description : "
    f"{products['description'].notna().sum():,}"
)


# --------------------------------------------------
# Semantic text
# --------------------------------------------------

semantic_lengths = (
    products["semantic_text"]
    .fillna("")
    .str.len()
)

print("\nSEMANTIC TEXT")
print("-" * 70)

print(
    f"Minimum length : "
    f"{semantic_lengths.min():,} characters"
)

print(
    f"Maximum length : "
    f"{semantic_lengths.max():,} characters"
)

print(
    f"Mean length    : "
    f"{semantic_lengths.mean():,.2f} characters"
)

print(
    f"Median length  : "
    f"{semantic_lengths.median():,.2f} characters"
)


# --------------------------------------------------
# User-product matrix sparsity
# --------------------------------------------------

number_users = interactions["user_id"].nunique()
number_products = interactions["product_id"].nunique()
number_interactions = len(interactions)

possible_interactions = (
    number_users * number_products
)

density = (
    number_interactions /
    possible_interactions
)

sparsity = 1 - density


print("\nUSER-PRODUCT MATRIX")
print("-" * 70)

print(
    f"Possible user-product pairs : "
    f"{possible_interactions:,}"
)

print(
    f"Observed interactions       : "
    f"{number_interactions:,}"
)

print(
    f"Matrix density              : "
    f"{density * 100:.6f}%"
)

print(
    f"Matrix sparsity             : "
    f"{sparsity * 100:.6f}%"
)


# --------------------------------------------------
# Dataset time range
# --------------------------------------------------

timestamps = pd.to_datetime(
    interactions["timestamp"],
    unit="ms",
    errors="coerce"
)

print("\nTIME RANGE")
print("-" * 70)

print(f"First interaction : {timestamps.min()}")
print(f"Last interaction  : {timestamps.max()}")


# --------------------------------------------------
# Save numerical overview
# --------------------------------------------------

summary = pd.DataFrame({
    "metric": [
        "users",
        "products",
        "interactions",
        "avg_interactions_per_user",
        "avg_interactions_per_product",
        "mean_rating",
        "matrix_density_percent",
        "matrix_sparsity_percent",
        "mean_semantic_text_length"
    ],

    "value": [
        number_users,
        number_products,
        number_interactions,
        user_activity.mean(),
        product_activity.mean(),
        interactions["rating"].mean(),
        density * 100,
        sparsity * 100,
        semantic_lengths.mean()
    ]
})

summary_file = OUTPUT_DIR / "dataset_summary.csv"

summary.to_csv(
    summary_file,
    index=False
)


print("\n" + "=" * 70)
print("OVERVIEW COMPLETE")
print("=" * 70)

print(
    f"\nSummary saved to:\n"
    f"{summary_file}"
)