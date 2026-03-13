import sys
from pathlib import Path


def fix_ceiling_file(dataset_root: str) -> None:
    root = Path(dataset_root)
    path = root / "Area_5" / "office_19" / "Annotations" / "ceiling_1.txt"
    value_to_fix = "103.00000"
    fixed_value = "103.000000"

    if not path.exists():
        print(f"File not found: {path}. Skipping fix.")
        return

    with open(path, "r") as f:
        lines = f.readlines()

    with open(path, "w") as f:
        for line in lines:
            if value_to_fix in line:
                line = line.replace(value_to_fix, fixed_value)
            f.write(line)

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python 0-fix_s3dis_ceiling_bug.py <dataset_root>")
        sys.exit(1)

    dataset_root = sys.argv[1]
    fix_ceiling_file(dataset_root)


if __name__ == "__main__":
    main()
