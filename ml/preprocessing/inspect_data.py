import json
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

print("=" * 70)
print("RAW DATASET INSPECTION")
print("=" * 70)

# Find all JSONL files
jsonl_files = list(RAW_DIR.glob("*.jsonl"))

if not jsonl_files:
    print("\nNo .jsonl files found inside data/raw/")
    raise SystemExit

print(f"\nNumber of JSONL files found: {len(jsonl_files)}")

for file_path in jsonl_files:

    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    # File size
    size_mb = file_path.stat().st_size / (1024 * 1024)
    size_gb = size_mb / 1024

    if size_gb >= 1:
        print(f"Size: {size_gb:.2f} GB")
    else:
        print(f"Size: {size_mb:.2f} MB")

    # Read only first 3 records
    print("\nFirst 3 records:")

    with open(file_path, "r", encoding="utf-8") as f:
        for i in range(3):
            line = f.readline()

            if not line:
                break

            record = json.loads(line)

            print(f"\n--- Record {i + 1} ---")
            print(json.dumps(record, indent=2, ensure_ascii=False))

            if i == 0:
                print("\nAttributes found:")
                print(list(record.keys()))