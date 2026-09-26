import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRODUCTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "products.csv"
)

products = pd.read_csv(PRODUCTS_FILE)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("=" * 70)
print("SEMANTIC FIELD CONTRIBUTION ANALYSIS")
print("=" * 70)

print("\nLoading MiniLM tokenizer...")

model = SentenceTransformer(MODEL_NAME)
tokenizer = model.tokenizer

print(
    f"Maximum model sequence length : "
    f"{model.max_seq_length}"
)


# Fields currently contributing to semantic representation
fields = [
    "title",
    "main_category",
    "categories",
    "brand",
    "features",
    "description",
    "useful_details"
]


results = []


for field in fields:

    print("\n" + "=" * 70)
    print(field.upper())
    print("=" * 70)

    texts = (
        products[field]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    non_empty = texts != ""

    character_lengths = (
        texts.str.len()
    )

    token_lengths = []

    for text in texts:

        if not text:
            token_lengths.append(0)
            continue

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False
        )

        token_lengths.append(
            len(encoded["input_ids"])
        )


    token_lengths = pd.Series(
        token_lengths
    )


    print(
        f"\nAvailable products : "
        f"{non_empty.sum():,}"
    )

    print(
        f"Availability       : "
        f"{non_empty.mean() * 100:.2f}%"
    )

    print(
        f"Mean characters    : "
        f"{character_lengths.mean():,.2f}"
    )

    print(
        f"Median characters  : "
        f"{character_lengths.median():,.2f}"
    )

    print(
        f"Mean tokens        : "
        f"{token_lengths.mean():,.2f}"
    )

    print(
        f"Median tokens      : "
        f"{token_lengths.median():,.2f}"
    )

    print(
        f"95th percentile    : "
        f"{token_lengths.quantile(0.95):,.0f} tokens"
    )

    print(
        f"Maximum tokens     : "
        f"{token_lengths.max():,}"
    )


    results.append({
        "field": field,

        "available_products":
            int(non_empty.sum()),

        "availability_percent":
            non_empty.mean() * 100,

        "mean_characters":
            character_lengths.mean(),

        "median_characters":
            character_lengths.median(),

        "mean_tokens":
            token_lengths.mean(),

        "median_tokens":
            token_lengths.median(),

        "p95_tokens":
            token_lengths.quantile(0.95),

        "max_tokens":
            token_lengths.max()
    })


# --------------------------------------------------
# Save summary
# --------------------------------------------------

results_df = pd.DataFrame(results)

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

output_file = (
    OUTPUT_DIR
    / "semantic_field_statistics.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# Overall contribution
# --------------------------------------------------

print("\n" + "=" * 70)
print("AVERAGE TOKEN CONTRIBUTION")
print("=" * 70)

total_average = (
    results_df["mean_tokens"].sum()
)


for row in results_df.itertuples():

    contribution = (
        row.mean_tokens
        / total_average
        * 100
    )

    print(
        f"{row.field:<18} : "
        f"{row.mean_tokens:>7.2f} avg tokens | "
        f"{contribution:>6.2f}%"
    )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nSummary saved to:\n"
    f"{output_file}"
)