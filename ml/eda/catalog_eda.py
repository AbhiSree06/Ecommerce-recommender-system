import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
print("CATALOG ANALYSIS")
print("=" * 70)

products = pd.read_csv(PRODUCTS_FILE)

print(
    f"\nProducts loaded: "
    f"{len(products):,}"
)


# ==================================================
# 1. MAIN CATEGORY ANALYSIS
# ==================================================

print("\n" + "=" * 70)
print("1. MAIN CATEGORY ANALYSIS")
print("=" * 70)


category_series = (
    products["main_category"]
    .fillna("")
    .astype(str)
    .str.strip()
)

valid_categories = category_series[
    category_series != ""
]


category_counts = (
    valid_categories
    .value_counts()
)


print(
    f"\nProducts with main category : "
    f"{len(valid_categories):,}"
)

print(
    f"Missing main category       : "
    f"{len(products) - len(valid_categories):,}"
)

print(
    f"Unique main categories      : "
    f"{category_counts.size:,}"
)


print("\nTop main categories:\n")

for rank, (category, count) in enumerate(
    category_counts.head(15).items(),
    start=1
):

    percentage = (
        count /
        len(products) *
        100
    )

    print(
        f"{rank:>2}. "
        f"{category[:55]:<55} "
        f"{count:>5,} "
        f"({percentage:.2f}%)"
    )


category_counts.rename_axis(
    "main_category"
).reset_index(
    name="product_count"
).to_csv(
    OUTPUT_DIR / "category_distribution.csv",
    index=False
)


# --------------------------------------------------
# Top categories graph
# --------------------------------------------------

top_categories = (
    category_counts
    .head(15)
    .sort_values()
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_categories.index,
    top_categories.values
)

plt.title(
    "Top 15 Product Categories"
)

plt.xlabel(
    "Number of Products"
)

plt.ylabel(
    "Main Category"
)

plt.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "top_categories.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 2. BRAND ANALYSIS
# ==================================================

print("\n" + "=" * 70)
print("2. BRAND ANALYSIS")
print("=" * 70)


brand_series = (
    products["brand"]
    .fillna("")
    .astype(str)
    .str.strip()
)

valid_brands = brand_series[
    brand_series != ""
]


brand_counts = (
    valid_brands
    .value_counts()
)


print(
    f"\nProducts with brand : "
    f"{len(valid_brands):,}"
)

print(
    f"Products without brand : "
    f"{len(products) - len(valid_brands):,}"
)

print(
    f"Unique brands : "
    f"{brand_counts.size:,}"
)


print("\nTop brands:\n")

for rank, (brand, count) in enumerate(
    brand_counts.head(20).items(),
    start=1
):

    percentage = (
        count /
        len(products) *
        100
    )

    print(
        f"{rank:>2}. "
        f"{brand[:45]:<45} "
        f"{count:>5,} "
        f"({percentage:.2f}%)"
    )


brand_counts.rename_axis(
    "brand"
).reset_index(
    name="product_count"
).to_csv(
    OUTPUT_DIR / "brand_distribution.csv",
    index=False
)


# --------------------------------------------------
# Top brands graph
# --------------------------------------------------

top_brands = (
    brand_counts
    .head(15)
    .sort_values()
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_brands.index,
    top_brands.values
)

plt.title(
    "Top 15 Brands in Product Catalog"
)

plt.xlabel(
    "Number of Products"
)

plt.ylabel(
    "Brand"
)

plt.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "top_brands.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 3. PRICE ANALYSIS
# ==================================================

print("\n" + "=" * 70)
print("3. PRICE ANALYSIS")
print("=" * 70)


prices = pd.to_numeric(
    products["price"],
    errors="coerce"
)

valid_prices = prices[
    prices.notna()
    & (prices > 0)
]


print(
    f"\nProducts with valid positive price : "
    f"{len(valid_prices):,}"
)

print(
    f"Products without valid price       : "
    f"{len(products) - len(valid_prices):,}"
)


if len(valid_prices) > 0:

    print(
        f"\nMinimum price : "
        f"${valid_prices.min():,.2f}"
    )

    print(
        f"Maximum price : "
        f"${valid_prices.max():,.2f}"
    )

    print(
        f"Mean price    : "
        f"${valid_prices.mean():,.2f}"
    )

    print(
        f"Median price  : "
        f"${valid_prices.median():,.2f}"
    )


    print("\nPrice percentiles:")

    for percentile in [
        25,
        50,
        75,
        90,
        95,
        99
    ]:

        value = valid_prices.quantile(
            percentile / 100
        )

        print(
            f"{percentile:>2}th percentile : "
            f"${value:,.2f}"
        )


# ==================================================
# 4. PRICE RANGES
# ==================================================

print("\n" + "=" * 70)
print("4. PRICE RANGE DISTRIBUTION")
print("=" * 70)


price_bins = [
    0,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    float("inf")
]

price_labels = [
    "$0-$10",
    "$10-$25",
    "$25-$50",
    "$50-$100",
    "$100-$250",
    "$250-$500",
    "$500-$1000",
    "$1000+"
]


price_ranges = pd.cut(
    valid_prices,
    bins=price_bins,
    labels=price_labels,
    right=False
)


price_range_counts = (
    price_ranges
    .value_counts()
    .reindex(price_labels)
    .fillna(0)
    .astype(int)
)


print()

for price_range, count in (
    price_range_counts.items()
):

    percentage = (
        count /
        len(valid_prices) *
        100
    )

    print(
        f"{price_range:<12}: "
        f"{count:>5,} "
        f"({percentage:.2f}%)"
    )


price_range_counts.rename_axis(
    "price_range"
).reset_index(
    name="product_count"
).to_csv(
    OUTPUT_DIR / "price_range_distribution.csv",
    index=False
)


# --------------------------------------------------
# Price range graph
# --------------------------------------------------

plt.figure(figsize=(10, 5))

bars = plt.bar(
    price_range_counts.index,
    price_range_counts.values
)

plt.title(
    "Product Price Range Distribution"
)

plt.xlabel(
    "Price Range"
)

plt.ylabel(
    "Number of Products"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.25
)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        height,

        f"{int(height):,}",

        ha="center",
        va="bottom",
        fontsize=8
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "price_range_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 5. OUTLIER-AWARE PRICE HISTOGRAM
# ==================================================

if len(valid_prices) > 0:

    upper_limit = (
        valid_prices.quantile(0.99)
    )

    plot_prices = valid_prices[
        valid_prices <= upper_limit
    ]

    plt.figure(figsize=(10, 5))

    plt.hist(
        plot_prices,
        bins=40
    )

    plt.title(
        "Product Price Distribution "
        "(Up to 99th Percentile)"
    )

    plt.xlabel(
        "Price (USD)"
    )

    plt.ylabel(
        "Number of Products"
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "price_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==================================================
# Complete
# ==================================================

print("\n" + "=" * 70)
print("CATALOG ANALYSIS COMPLETE")
print("=" * 70)

print("\nGenerated graphs:")

print(
    OUTPUT_DIR / "top_categories.png"
)

print(
    OUTPUT_DIR / "top_brands.png"
)

print(
    OUTPUT_DIR / "price_range_distribution.png"
)

print(
    OUTPUT_DIR / "price_distribution.png"
)