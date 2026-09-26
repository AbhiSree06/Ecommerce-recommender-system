import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sentence_transformers import SentenceTransformer


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
# Model
# --------------------------------------------------

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


print("=" * 70)
print("SEMANTIC TEXT AND MINILM TOKEN ANALYSIS")
print("=" * 70)


# --------------------------------------------------
# Load products
# --------------------------------------------------

products = pd.read_csv(
    PRODUCTS_FILE
)

texts = (
    products["semantic_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

print(
    f"\nProducts loaded : "
    f"{len(products):,}"
)


# --------------------------------------------------
# Load model/tokenizer
# --------------------------------------------------

print(
    "\nLoading MiniLM model..."
)

model = SentenceTransformer(
    MODEL_NAME
)

tokenizer = model.tokenizer

max_seq_length = (
    model.max_seq_length
)

print(
    f"Model maximum sequence length : "
    f"{max_seq_length} tokens"
)


# ==================================================
# 1. CHARACTER LENGTHS
# ==================================================

character_lengths = (
    products["semantic_text"]
    .fillna("")
    .astype(str)
    .str.len()
)


print("\n" + "=" * 70)
print("1. CHARACTER LENGTH ANALYSIS")
print("=" * 70)

print(
    f"\nMinimum : "
    f"{character_lengths.min():,}"
)

print(
    f"Maximum : "
    f"{character_lengths.max():,}"
)

print(
    f"Mean    : "
    f"{character_lengths.mean():,.2f}"
)

print(
    f"Median  : "
    f"{character_lengths.median():,.2f}"
)


# ==================================================
# 2. TOKEN LENGTHS
# ==================================================

print("\n" + "=" * 70)
print("2. TOKEN LENGTH ANALYSIS")
print("=" * 70)

print(
    "\nTokenizing semantic text..."
)


token_lengths = []


for index, text in enumerate(
    texts,
    start=1
):

    # No truncation here because we want
    # the REAL token length of each text.
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )

    token_lengths.append(
        len(encoded["input_ids"])
    )

    if index % 1000 == 0:

        print(
            f"Processed "
            f"{index:,}/{len(texts):,}"
        )


token_lengths = pd.Series(
    token_lengths,
    index=products.index
)


print(
    f"\nMinimum token length : "
    f"{token_lengths.min():,}"
)

print(
    f"Maximum token length : "
    f"{token_lengths.max():,}"
)

print(
    f"Mean token length    : "
    f"{token_lengths.mean():,.2f}"
)

print(
    f"Median token length  : "
    f"{token_lengths.median():,.2f}"
)


print("\nToken percentiles:")

for percentile in [
    25,
    50,
    75,
    90,
    95,
    99
]:

    value = token_lengths.quantile(
        percentile / 100
    )

    print(
        f"{percentile:>2}th percentile : "
        f"{value:.0f}"
    )


# ==================================================
# 3. MINILM TRUNCATION ANALYSIS
# ==================================================

print("\n" + "=" * 70)
print("3. MINILM TRUNCATION ANALYSIS")
print("=" * 70)


within_limit = (
    token_lengths
    <= max_seq_length
).sum()

over_limit = (
    token_lengths
    > max_seq_length
).sum()


within_percentage = (
    within_limit
    / len(products)
    * 100
)

over_percentage = (
    over_limit
    / len(products)
    * 100
)


print(
    f"\nProducts within "
    f"{max_seq_length} tokens : "
    f"{within_limit:,} "
    f"({within_percentage:.2f}%)"
)

print(
    f"Products exceeding "
    f"{max_seq_length} tokens : "
    f"{over_limit:,} "
    f"({over_percentage:.2f}%)"
)


# ==================================================
# 4. TOKEN THRESHOLDS
# ==================================================

print("\n" + "=" * 70)
print("4. TOKEN LENGTH THRESHOLDS")
print("=" * 70)


thresholds = [
    128,
    256,
    384,
    512,
    768,
    1024
]


threshold_rows = []


for threshold in thresholds:

    count = (
        token_lengths
        > threshold
    ).sum()

    percentage = (
        count
        / len(products)
        * 100
    )

    threshold_rows.append({
        "threshold":
            threshold,

        "products_exceeding":
            count,

        "percentage":
            percentage
    })

    print(
        f"Products > {threshold:>4} tokens : "
        f"{count:>6,} "
        f"({percentage:.2f}%)"
    )


pd.DataFrame(
    threshold_rows
).to_csv(
    OUTPUT_DIR
    / "semantic_token_thresholds.csv",
    index=False
)


# ==================================================
# 5. SAVE PRODUCT TOKEN STATISTICS
# ==================================================

token_data = pd.DataFrame({
    "product_id":
        products["product_id"],

    "character_length":
        character_lengths,

    "token_length":
        token_lengths,

    "exceeds_model_limit":
        token_lengths
        > max_seq_length
})


token_data.to_csv(
    OUTPUT_DIR
    / "semantic_text_lengths.csv",
    index=False
)


# ==================================================
# 6. TOKEN LENGTH GRAPH
# ==================================================

plot_limit = (
    token_lengths
    .quantile(0.99)
)


plot_lengths = (
    token_lengths[
        token_lengths
        <= plot_limit
    ]
)


plt.figure(
    figsize=(10, 5)
)

plt.hist(
    plot_lengths,
    bins=50
)

plt.axvline(
    max_seq_length,
    linestyle="--",
    linewidth=2,
    label=(
        f"MiniLM limit "
        f"({max_seq_length} tokens)"
    )
)

plt.title(
    "Semantic Text Token-Length Distribution"
)

plt.xlabel(
    "Number of Tokens"
)

plt.ylabel(
    "Number of Products"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "semantic_token_length_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 7. LONGEST PRODUCT TEXTS
# ==================================================

print("\n" + "=" * 70)
print("5. PRODUCTS WITH LONGEST SEMANTIC TEXT")
print("=" * 70)


longest_indices = (
    token_lengths
    .nlargest(10)
    .index
)


for rank, index in enumerate(
    longest_indices,
    start=1
):

    product_id = (
        products.loc[
            index,
            "product_id"
        ]
    )

    title = str(
        products.loc[
            index,
            "title"
        ]
    )

    if len(title) > 55:
        title = (
            title[:52]
            + "..."
        )

    print(
        f"{rank:>2}. "
        f"{product_id} | "
        f"{token_lengths[index]:>5,} tokens | "
        f"{title}"
    )


# ==================================================
# Complete
# ==================================================

print("\n" + "=" * 70)
print("SEMANTIC TOKEN ANALYSIS COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(
    OUTPUT_DIR
    / "semantic_text_lengths.csv"
)

print(
    OUTPUT_DIR
    / "semantic_token_thresholds.csv"
)

print(
    OUTPUT_DIR
    / "semantic_token_length_distribution.png"
)