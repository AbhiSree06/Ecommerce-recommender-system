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
# Load interactions
# --------------------------------------------------

print("=" * 70)
print("RATING AND USER ACTIVITY ANALYSIS")
print("=" * 70)

interactions = pd.read_csv(
    INTERACTIONS_FILE
)

print(
    f"\nInteractions loaded: "
    f"{len(interactions):,}"
)


# ==================================================
# 1. RATING DISTRIBUTION
# ==================================================

print("\n" + "=" * 70)
print("1. RATING DISTRIBUTION")
print("=" * 70)


rating_counts = (
    interactions["rating"]
    .value_counts()
    .sort_index()
)

rating_percentages = (
    rating_counts /
    len(interactions) *
    100
)


rating_table = pd.DataFrame({
    "rating": rating_counts.index,
    "count": rating_counts.values,
    "percentage": rating_percentages.values
})


print("\nRating counts:\n")

for _, row in rating_table.iterrows():

    print(
        f"{row['rating']:.0f} stars : "
        f"{int(row['count']):,} "
        f"({row['percentage']:.2f}%)"
    )


rating_table.to_csv(
    OUTPUT_DIR / "rating_distribution.csv",
    index=False
)


# --------------------------------------------------
# Rating graph
# --------------------------------------------------

plt.figure(figsize=(8, 5))

bars = plt.bar(
    rating_counts.index.astype(str),
    rating_counts.values
)

plt.title(
    "Distribution of User Ratings"
)

plt.xlabel("Rating")
plt.ylabel("Number of Interactions")

plt.grid(
    axis="y",
    alpha=0.25
)


# Add values above bars
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height):,}",
        ha="center",
        va="bottom",
        fontsize=9
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "rating_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 2. USER ACTIVITY
# ==================================================

print("\n" + "=" * 70)
print("2. USER ACTIVITY")
print("=" * 70)


user_activity = (
    interactions
    .groupby("user_id")
    .size()
)


print(
    f"\nUsers : "
    f"{len(user_activity):,}"
)

print(
    f"Minimum interactions : "
    f"{user_activity.min()}"
)

print(
    f"Maximum interactions : "
    f"{user_activity.max()}"
)

print(
    f"Mean interactions    : "
    f"{user_activity.mean():.2f}"
)

print(
    f"Median interactions  : "
    f"{user_activity.median():.2f}"
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

    value = user_activity.quantile(
        percentile / 100
    )

    print(
        f"{percentile:>2}th percentile : "
        f"{value:.0f}"
    )


# --------------------------------------------------
# Exact sequence-length distribution
# --------------------------------------------------

sequence_distribution = (
    user_activity
    .value_counts()
    .sort_index()
)

sequence_table = pd.DataFrame({
    "interaction_count":
        sequence_distribution.index,

    "number_of_users":
        sequence_distribution.values
})

sequence_table[
    "percentage_of_users"
] = (
    sequence_table["number_of_users"]
    / len(user_activity)
    * 100
)


sequence_table.to_csv(
    OUTPUT_DIR
    / "user_sequence_length_distribution.csv",
    index=False
)


print("\nUsers by sequence length:\n")

for _, row in sequence_table.iterrows():

    print(
        f"{int(row['interaction_count']):>2} interactions : "
        f"{int(row['number_of_users']):>5,} users "
        f"({row['percentage_of_users']:.2f}%)"
    )


# --------------------------------------------------
# User activity graph
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.bar(
    sequence_distribution.index,
    sequence_distribution.values
)

plt.title(
    "Distribution of User Interaction Sequence Lengths"
)

plt.xlabel(
    "Number of Interactions per User"
)

plt.ylabel(
    "Number of Users"
)

plt.xticks(
    range(
        int(user_activity.min()),
        int(user_activity.max()) + 1,
        2
    )
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "user_sequence_length_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 3. CUMULATIVE USER ACTIVITY
# ==================================================

print("\n" + "=" * 70)
print("3. USER ACTIVITY THRESHOLDS")
print("=" * 70)


thresholds = [
    3,
    4,
    5,
    6,
    10,
    15,
    20
]


threshold_data = []

for threshold in thresholds:

    count = (
        user_activity >= threshold
    ).sum()

    percentage = (
        count /
        len(user_activity) *
        100
    )

    threshold_data.append({
        "minimum_interactions":
            threshold,

        "users":
            count,

        "percentage":
            percentage
    })

    print(
        f"Users with >= {threshold:>2} interactions : "
        f"{count:>6,} "
        f"({percentage:.2f}%)"
    )


pd.DataFrame(
    threshold_data
).to_csv(
    OUTPUT_DIR
    / "user_activity_thresholds.csv",
    index=False
)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(
    OUTPUT_DIR
    / "rating_distribution.png"
)

print(
    OUTPUT_DIR
    / "user_sequence_length_distribution.png"
)