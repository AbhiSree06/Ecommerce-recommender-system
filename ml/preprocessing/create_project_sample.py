import pandas as pd
from pathlib import Path
from collections import Counter
import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "interactions_basic.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical_interactions.csv"
)

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Project sampling settings
MIN_USER_INTERACTIONS = 5
MAX_USER_INTERACTIONS = 50
TARGET_USERS = 20_000

print("=" * 65)
print("CREATING PROJECT INTERACTION SAMPLE")
print("=" * 65)

# --------------------------------------------------
# STEP 1: Count interactions per user
# --------------------------------------------------

print("\nStep 1: Counting user interactions...")

user_counts = Counter()

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=["user_id"],
    chunksize=500_000
):
    user_counts.update(chunk["user_id"])

# --------------------------------------------------
# STEP 2: Find eligible users
# --------------------------------------------------

eligible_users = [
    user_id
    for user_id, count in user_counts.items()
    if MIN_USER_INTERACTIONS <= count <= MAX_USER_INTERACTIONS
]

print(f"Eligible users: {len(eligible_users):,}")

# --------------------------------------------------
# STEP 3: Randomly select users
# --------------------------------------------------

if len(eligible_users) < TARGET_USERS:
    raise ValueError(
        f"Only {len(eligible_users):,} eligible users found."
    )

selected_users = set(
    random.sample(eligible_users, TARGET_USERS)
)

print(f"Selected users: {len(selected_users):,}")

# --------------------------------------------------
# STEP 4: Retrieve ALL interactions of selected users
# --------------------------------------------------

print("\nStep 2: Extracting selected user histories...")

selected_chunks = []

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=500_000
):

    filtered = chunk[
        chunk["user_id"].isin(selected_users)
    ]

    if not filtered.empty:
        selected_chunks.append(filtered)

project_data = pd.concat(
    selected_chunks,
    ignore_index=True
)

# --------------------------------------------------
# STEP 5: Remove exact duplicate interactions
# --------------------------------------------------

before_duplicates = len(project_data)

project_data = project_data.drop_duplicates()

removed_duplicates = before_duplicates - len(project_data)

# --------------------------------------------------
# STEP 6: Sort chronologically
# --------------------------------------------------

project_data = project_data.sort_values(
    by=["user_id", "timestamp"]
).reset_index(drop=True)

# --------------------------------------------------
# STEP 7: Save
# --------------------------------------------------

project_data.to_csv(
    OUTPUT_FILE,
    index=False
)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("PROJECT DATASET CREATED")
print("=" * 65)

print(f"Users                 : {project_data['user_id'].nunique():,}")
print(f"Products              : {project_data['product_id'].nunique():,}")
print(f"Interactions          : {len(project_data):,}")
print(f"Duplicates removed    : {removed_duplicates:,}")

avg = len(project_data) / project_data["user_id"].nunique()

print(f"Average/user          : {avg:.2f}")

print("\nRating distribution:")
print(
    project_data["rating"]
    .value_counts()
    .sort_index()
)

print(f"\nSaved to:\n{OUTPUT_FILE}")