import ast
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "products_basic.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "products.csv"
)


# --------------------------------------------------
# Helper: safely parse JSON-like columns
# --------------------------------------------------

def parse_nested(value, default):

    if pd.isna(value):
        return default

    if isinstance(value, (list, dict)):
        return value

    try:
        return json.loads(value)

    except (json.JSONDecodeError, TypeError):

        try:
            return ast.literal_eval(value)

        except (ValueError, SyntaxError, TypeError):
            return default


# --------------------------------------------------
# Helper: normalize text
# --------------------------------------------------

def clean_text(value):

    if value is None or pd.isna(value):
        return ""

    value = str(value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


# --------------------------------------------------
# Extract brand
# --------------------------------------------------

def extract_brand(store, details):

    details = parse_nested(details, {})

    # Prefer explicit Brand field
    brand = details.get("Brand")

    if brand:
        return clean_text(brand)

    # Manufacturer is second preference
    manufacturer = details.get("Manufacturer")

    if manufacturer:
        return clean_text(manufacturer)

    # Store is final fallback
    if pd.notna(store):
        return clean_text(store)

    return ""


# --------------------------------------------------
# Extract MAIN product image
# --------------------------------------------------

def extract_main_image(images):

    images = parse_nested(images, [])

    if not images:
        return ""

    # First preference: MAIN image
    for image in images:

        if not isinstance(image, dict):
            continue

        if image.get("variant") == "MAIN":

            return (
                image.get("hi_res")
                or image.get("large")
                or image.get("thumb")
                or ""
            )

    # Fallback to first available image
    for image in images:

        if not isinstance(image, dict):
            continue

        url = (
            image.get("hi_res")
            or image.get("large")
            or image.get("thumb")
        )

        if url:
            return url

    return ""


# --------------------------------------------------
# Convert list fields to readable text
# --------------------------------------------------

def list_to_text(value):

    items = parse_nested(value, [])

    if not isinstance(items, list):
        return clean_text(items)

    cleaned_items = []

    for item in items:

        text = clean_text(item)

        if text:
            cleaned_items.append(text)

    return " | ".join(cleaned_items)


# --------------------------------------------------
# Select useful details for semantic understanding
# --------------------------------------------------

USEFUL_DETAIL_KEYS = [
    "Brand",
    "Manufacturer",
    "Model Name",
    "Item model number",
    "Color",
    "Material",
    "Compatible Devices",
    "Connectivity Technology",
    "Special Feature",
    "Special Features",
    "Hardware Platform",
    "Operating System",
    "Memory Storage Capacity",
    "Screen Size",
    "Form Factor",
    "Connector Type"
]


def extract_useful_details(details):

    details = parse_nested(details, {})

    if not isinstance(details, dict):
        return ""

    selected = []

    for key in USEFUL_DETAIL_KEYS:

        value = details.get(key)

        if value is None:
            continue

        # Skip nested dictionaries/lists here
        if isinstance(value, (dict, list)):
            continue

        value = clean_text(value)

        if value:
            selected.append(
                f"{key}: {value}"
            )

    return " | ".join(selected)


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("=" * 65)
print("PREPARING FINAL PRODUCT DATASET")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

print(f"\nProducts loaded: {len(df):,}")


# --------------------------------------------------
# Clean core fields
# --------------------------------------------------

df["title"] = (
    df["title"]
    .fillna("")
    .apply(clean_text)
)

df["main_category"] = (
    df["main_category"]
    .fillna("")
    .apply(clean_text)
)


# --------------------------------------------------
# Nested fields
# --------------------------------------------------

df["categories_text"] = (
    df["categories"]
    .apply(list_to_text)
)

df["features_text"] = (
    df["features"]
    .apply(list_to_text)
)

df["description_text"] = (
    df["description"]
    .apply(list_to_text)
)


# --------------------------------------------------
# Brand
# --------------------------------------------------

df["brand"] = df.apply(
    lambda row: extract_brand(
        row["store"],
        row["details"]
    ),
    axis=1
)


# --------------------------------------------------
# Main product image
# --------------------------------------------------

df["image_url"] = (
    df["images"]
    .apply(extract_main_image)
)


# --------------------------------------------------
# Useful product details
# --------------------------------------------------

df["useful_details"] = (
    df["details"]
    .apply(extract_useful_details)
)


# --------------------------------------------------
# Numeric fields
# --------------------------------------------------

df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

df["average_rating"] = pd.to_numeric(
    df["average_rating"],
    errors="coerce"
)

df["rating_count"] = pd.to_numeric(
    df["rating_count"],
    errors="coerce"
)


# --------------------------------------------------
# Construct semantic text
# --------------------------------------------------

def build_semantic_text(row):

    components = [
        row["title"],
        row["main_category"],
        row["categories_text"],
        row["brand"],
        row["features_text"],
        row["description_text"],
        row["useful_details"]
    ]

    components = [
        clean_text(x)
        for x in components
        if clean_text(x)
    ]

    return " ".join(components)


df["semantic_text"] = df.apply(
    build_semantic_text,
    axis=1
)


# --------------------------------------------------
# Final columns
# --------------------------------------------------

final_df = df[
    [
        "product_id",
        "title",
        "main_category",
        "categories_text",
        "brand",
        "price",
        "average_rating",
        "rating_count",
        "image_url",
        "features_text",
        "description_text",
        "useful_details",
        "semantic_text"
    ]
].copy()


# Rename for cleaner final table
final_df = final_df.rename(
    columns={
        "categories_text": "categories",
        "features_text": "features",
        "description_text": "description"
    }
)


# --------------------------------------------------
# Remove products without usable semantic information
# --------------------------------------------------

before = len(final_df)

final_df = final_df[
    final_df["semantic_text"].str.len() > 0
].copy()

removed = before - len(final_df)


# --------------------------------------------------
# Save
# --------------------------------------------------

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINAL PRODUCT DATASET")
print("=" * 65)

print(f"Products                 : {len(final_df):,}")
print(f"Empty semantic text      : {removed:,}")

print(
    f"Products with title      : "
    f"{(final_df['title'] != '').sum():,}"
)

print(
    f"Products with category   : "
    f"{(final_df['categories'] != '').sum():,}"
)

print(
    f"Products with brand      : "
    f"{(final_df['brand'] != '').sum():,}"
)

print(
    f"Products with price      : "
    f"{final_df['price'].notna().sum():,}"
)

print(
    f"Products with image      : "
    f"{(final_df['image_url'] != '').sum():,}"
)

print(
    f"Products with features   : "
    f"{(final_df['features'] != '').sum():,}"
)

print(
    f"Products with description: "
    f"{(final_df['description'] != '').sum():,}"
)

print(
    f"Products with details    : "
    f"{(final_df['useful_details'] != '').sum():,}"
)

print(
    f"\nAverage semantic text length: "
    f"{final_df['semantic_text'].str.len().mean():.0f} characters"
)

print(f"\nSaved to:\n{OUTPUT_FILE}")