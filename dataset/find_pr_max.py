import numpy as np
from pathlib import Path
from tqdm import tqdm


def find_pr_max(pr_dir: Path) -> int:
    """
    finds the maximum value in the piano roll files
    in a given directory
    """
    max_value = 0
    for fp in tqdm(pr_dir.iterdir()):
        if "piano" in str(fp):
            file_max = np.load(fp).max()
            if file_max > max_value:
                max_value = file_max
    return max_value


def main():
    pr_max = find_pr_max(pr_dir=Path())
    print(pr_max)


if __name__ == "__main__":
    main()
