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
print("TEMPORAL AND METADATA ANALYSIS")
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
# 1. CONVERT TIMESTAMPS
# ==================================================

interactions["datetime"] = pd.to_datetime(
    interactions["timestamp"],
    unit="ms",
    errors="coerce"
)

invalid_timestamps = (
    interactions["datetime"]
    .isna()
    .sum()
)


print("\n" + "=" * 70)
print("1. TEMPORAL COVERAGE")
print("=" * 70)

print(
    f"\nInvalid timestamps : "
    f"{invalid_timestamps:,}"
)

print(
    f"First interaction  : "
    f"{interactions['datetime'].min()}"
)

print(
    f"Last interaction   : "
    f"{interactions['datetime'].max()}"
)


# ==================================================
# 2. INTERACTIONS BY YEAR
# ==================================================

interactions["year"] = (
    interactions["datetime"]
    .dt.year
)


year_counts = (
    interactions["year"]
    .value_counts()
    .sort_index()
)


print("\nInteractions by year:\n")

for year, count in year_counts.items():

    percentage = (
        count /
        len(interactions) *
        100
    )

    print(
        f"{int(year)} : "
        f"{count:>6,} "
        f"({percentage:.2f}%)"
    )


year_counts.rename_axis(
    "year"
).reset_index(
    name="interaction_count"
).to_csv(
    OUTPUT_DIR
    / "interactions_by_year.csv",
    index=False
)


# --------------------------------------------------
# Yearly interaction graph
# --------------------------------------------------

plt.figure(figsize=(11, 5))

plt.plot(
    year_counts.index,
    year_counts.values,
    marker="o"
)

plt.title(
    "Historical Interactions by Year"
)

plt.xlabel("Year")
plt.ylabel("Number of Interactions")

plt.xticks(
    year_counts.index,
    rotation=45
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "interactions_by_year.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 3. RECENT-DATA CONCENTRATION
# ==================================================

print("\n" + "=" * 70)
print("2. RECENT DATA CONCENTRATION")
print("=" * 70)


for start_year in [
    2010,
    2015,
    2018,
    2020
]:

    count = (
        interactions["year"]
        >= start_year
    ).sum()

    percentage = (
        count /
        len(interactions)
        * 100
    )

    print(
        f"Interactions from {start_year}+ : "
        f"{count:>6,} "
        f"({percentage:.2f}%)"
    )


# ==================================================
# 4. METADATA COMPLETENESS
# ==================================================

print("\n" + "=" * 70)
print("3. PRODUCT METADATA COMPLETENESS")
print("=" * 70)


metadata_columns = [
    "title",
    "main_category",
    "categories",
    "brand",
    "price",
    "average_rating",
    "rating_count",
    "image_url",
    "features",
    "description",
    "useful_details",
    "semantic_text"
]


metadata_rows = []


for column in metadata_columns:

    if column not in products.columns:
        continue

    series = products[column]

    # Numeric columns
    if pd.api.types.is_numeric_dtype(series):

        available = (
            series
            .notna()
            .sum()
        )

    else:

        cleaned = (
            series
            .fillna("")
            .astype(str)
            .str.strip()
        )

        available = (
            cleaned != ""
        ).sum()

    missing = (
        len(products)
        - available
    )

    completeness = (
        available /
        len(products)
        * 100
    )

    metadata_rows.append({
        "attribute": column,
        "available": available,
        "missing": missing,
        "completeness_percent":
            completeness
    })


metadata_summary = pd.DataFrame(
    metadata_rows
)


for row in metadata_summary.itertuples():

    print(
        f"{row.attribute:<18} : "
        f"{row.available:>6,} available | "
        f"{row.missing:>5,} missing | "
        f"{row.completeness_percent:>6.2f}%"
    )


metadata_summary.to_csv(
    OUTPUT_DIR
    / "metadata_completeness.csv",
    index=False
)


# --------------------------------------------------
# Metadata completeness graph
# --------------------------------------------------

plot_data = (
    metadata_summary
    .sort_values(
        "completeness_percent"
    )
)


plt.figure(figsize=(10, 7))

bars = plt.barh(
    plot_data["attribute"],
    plot_data["completeness_percent"]
)

plt.title(
    "Product Metadata Completeness"
)

plt.xlabel(
    "Completeness (%)"
)

plt.ylabel(
    "Product Attribute"
)

plt.xlim(
    0,
    105
)

plt.grid(
    axis="x",
    alpha=0.25
)


for bar in bars:

    width = bar.get_width()

    plt.text(
        width + 0.5,
        bar.get_y()
        + bar.get_height() / 2,

        f"{width:.1f}%",

        va="center",
        fontsize=8
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "metadata_completeness.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 5. PRODUCTS WITH MULTIPLE MISSING FIELDS
# ==================================================

important_optional_fields = [
    "main_category",
    "categories",
    "brand",
    "price",
    "image_url",
    "features",
    "description"
]


missing_counts = pd.Series(
    0,
    index=products.index
)


for column in important_optional_fields:

    series = products[column]

    if pd.api.types.is_numeric_dtype(series):

        missing = (
            series.isna()
        )

    else:

        cleaned = (
            series
            .fillna("")
            .astype(str)
            .str.strip()
        )

        missing = (
            cleaned == ""
        )

    missing_counts += (
        missing.astype(int)
    )


print("\n" + "=" * 70)
print("4. OPTIONAL METADATA GAPS PER PRODUCT")
print("=" * 70)


gap_distribution = (
    missing_counts
    .value_counts()
    .sort_index()
)


for gaps, count in gap_distribution.items():

    percentage = (
        count /
        len(products)
        * 100
    )

    print(
        f"Products missing {gaps} optional fields : "
        f"{count:>6,} "
        f"({percentage:.2f}%)"
    )


# ==================================================
# Complete
# ==================================================

print("\n" + "=" * 70)
print("TEMPORAL AND METADATA ANALYSIS COMPLETE")
print("=" * 70)

print("\nGenerated graphs:")

print(
    OUTPUT_DIR
    / "interactions_by_year.png"
)

print(
    OUTPUT_DIR
    / "metadata_completeness.png"
)