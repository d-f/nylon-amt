import csv
import random
from pathlib import Path
from typing import List, Tuple


def convert_path(sg_filepath: Path) -> Path:
    """
    converts the name of a spectrogram filename to the
    corresponding piano roll filename
    """
    return sg_filepath.with_name(sg_filepath.stem.replace("-spec", "-piano") + ".npy")


def partition_datasets(sg_files: List[Path], val_prop: float) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    partitions the files into train, validation and test datasets
    """
    random.shuffle(sg_files)
    total = len(sg_files)
    val_amt = int(total * val_prop)
    
    val_sg = sg_files[:val_amt]
    test_sg = sg_files[val_amt:2*val_amt]
    train_sg = sg_files[2*val_amt:]
    
    return val_sg, test_sg, train_sg


def save_csv(csv_path: Path, csv_list: List[Path]):
    """
    saves a csv file
    """
    with open(csv_path, mode="w", newline="") as opened_csv:
        writer = csv.writer(opened_csv)
        writer.writerows([[str(row)] for row in csv_list])


def main():
    file_dir = Path("C:\\personal_ML\\nylon_gpt\\processed_data\\")
    sg_files = [x for x in file_dir.iterdir() if "spec" in x.name]
    
    val_sg, test_sg, train_sg = partition_datasets(sg_files, val_prop=0.1)
    
    dataset_types = ['val', 'test', 'train']
    sg_datasets = [val_sg, test_sg, train_sg]
    
    for dtype, sg_dataset in zip(dataset_types, sg_datasets):
        pr_dataset = [convert_path(x) for x in sg_dataset]
        
        save_csv(Path(f"C:\\personal_ML\\nylon_gpt\\dataset_csv\\{dtype}_sg.csv"), sg_dataset)
        save_csv(Path(f"C:\\personal_ML\\nylon_gpt\\dataset_csv\\{dtype}_pr.csv"), pr_dataset)


if __name__ == "__main__":
    main()
