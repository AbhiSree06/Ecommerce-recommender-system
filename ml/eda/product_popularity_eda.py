import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "eda"
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("=" * 70)
print("PRODUCT POPULARITY ANALYSIS")
print("=" * 70)

interactions = pd.read_csv(INTERACTIONS_FILE)
products = pd.read_csv(PRODUCTS_FILE)

print(
    f"\nInteractions loaded : "
    f"{len(interactions):,}"
)

print(
    f"Products loaded     : "
    f"{len(products):,}"
)


# ==================================================
# 1. PRODUCT INTERACTION COUNTS
# ==================================================

product_activity = (
    interactions
    .groupby("product_id")
    .size()
    .sort_values(ascending=False)
)


print("\n" + "=" * 70)
print("1. PRODUCT ACTIVITY")
print("=" * 70)

print(
    f"\nProducts : "
    f"{len(product_activity):,}"
)

print(
    f"Minimum interactions/product : "
    f"{product_activity.min()}"
)

print(
    f"Maximum interactions/product : "
    f"{product_activity.max()}"
)

print(
    f"Mean interactions/product    : "
    f"{product_activity.mean():.2f}"
)

print(
    f"Median interactions/product  : "
    f"{product_activity.median():.2f}"
)


print("\nPercentiles:")

for percentile in [
    25,
    50,
    75,
    90,
    95,
    99
]:

    value = product_activity.quantile(
        percentile / 100
    )

    print(
        f"{percentile:>2}th percentile : "
        f"{value:.0f}"
    )


# ==================================================
# 2. PRODUCT ACTIVITY THRESHOLDS
# ==================================================

print("\n" + "=" * 70)
print("2. PRODUCT ACTIVITY THRESHOLDS")
print("=" * 70)


thresholds = [
    3,
    4,
    5,
    10,
    20,
    50,
    100
]


threshold_rows = []

for threshold in thresholds:

    count = (
        product_activity >= threshold
    ).sum()

    percentage = (
        count /
        len(product_activity) *
        100
    )

    threshold_rows.append({
        "minimum_interactions": threshold,
        "products": count,
        "percentage": percentage
    })

    print(
        f"Products with >= {threshold:>3} interactions : "
        f"{count:>6,} "
        f"({percentage:.2f}%)"
    )


pd.DataFrame(
    threshold_rows
).to_csv(
    OUTPUT_DIR
    / "product_activity_thresholds.csv",
    index=False
)


# ==================================================
# 3. TOP PRODUCTS
# ==================================================

print("\n" + "=" * 70)
print("3. TOP 20 MOST INTERACTED PRODUCTS")
print("=" * 70)


product_info = products[
    [
        "product_id",
        "title",
        "brand",
        "main_category",
        "average_rating",
        "rating_count"
    ]
].copy()


popularity_df = (
    product_activity
    .rename("interaction_count")
    .reset_index()
    .merge(
        product_info,
        on="product_id",
        how="left"
    )
)


top_20 = popularity_df.head(20).copy()


for rank, row in enumerate(
    top_20.itertuples(),
    start=1
):

    title = str(row.title)

    if len(title) > 60:
        title = title[:57] + "..."

    print(
        f"{rank:>2}. "
        f"{row.product_id} | "
        f"{row.interaction_count:>4} interactions | "
        f"{title}"
    )


popularity_df.to_csv(
    OUTPUT_DIR
    / "product_popularity.csv",
    index=False
)


# ==================================================
# 4. POPULARITY CONCENTRATION
# ==================================================

print("\n" + "=" * 70)
print("4. POPULARITY CONCENTRATION")
print("=" * 70)


total_interactions = (
    product_activity.sum()
)


for top_percentage in [
    1,
    5,
    10,
    20
]:

    number_products = max(
        1,
        round(
            len(product_activity)
            * top_percentage
            / 100
        )
    )

    interactions_from_top = (
        product_activity
        .iloc[:number_products]
        .sum()
    )

    interaction_share = (
        interactions_from_top
        / total_interactions
        * 100
    )

    print(
        f"Top {top_percentage:>2}% products "
        f"({number_products:,}) account for "
        f"{interaction_share:.2f}% "
        f"of interactions"
    )


# ==================================================
# 5. PRODUCT POPULARITY DISTRIBUTION GRAPH
# ==================================================

frequency_distribution = (
    product_activity
    .value_counts()
    .sort_index()
)


plt.figure(
    figsize=(10, 5)
)

plt.bar(
    frequency_distribution.index,
    frequency_distribution.values
)

plt.title(
    "Distribution of Product Interaction Counts"
)

plt.xlabel(
    "Number of Interactions per Product"
)

plt.ylabel(
    "Number of Products"
)

plt.xlim(
    0,
    min(
        50,
        int(product_activity.max())
    )
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "product_popularity_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 6. TOP 20 PRODUCTS GRAPH
# ==================================================

top_graph = top_20.sort_values(
    "interaction_count",
    ascending=True
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_graph["product_id"],
    top_graph["interaction_count"]
)

plt.title(
    "Top 20 Products by Historical Interaction Count"
)

plt.xlabel(
    "Number of Interactions"
)

plt.ylabel(
    "Product ID"
)

plt.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "top_20_products.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# Complete
# ==================================================

print("\n" + "=" * 70)
print("PRODUCT POPULARITY ANALYSIS COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(
    OUTPUT_DIR
    / "product_popularity_distribution.png"
)

print(
    OUTPUT_DIR
    / "top_20_products.png"
)

print(
    OUTPUT_DIR
    / "product_popularity.csv"
)