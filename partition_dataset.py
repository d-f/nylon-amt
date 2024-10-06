import csv
import random
from pathlib import Path


def convert_path(sg_filepath: Path) -> Path:
    """
    converts the name of an spectrogram filename to the
    corresponding piano roll filename
    """
    pr_filename = sg_filepath.name.rpartition("-spec.npy")[0] + "-piano.npy"
    pr_path = sg_filepath.parent.joinpath(pr_filename)
    return pr_path


def partition_datasets(sg_files, val_prop):
    val_amt = int(len(sg_files) * val_prop)
    val_sg = random.sample(sg_files, val_amt)

    sg_files = [x for x in sg_files if x not in val_sg]

    test_sg = random.sample(sg_files, val_amt)
    sg_files = [x for x in sg_files if x not in test_sg]

    train_sg = sg_files

    val_pr = [convert_path(x) for x in val_sg]
    test_pr = [convert_path(x) for x in test_sg]
    train_pr = [convert_path(x) for x in train_sg]

    return val_sg, val_pr, test_sg, test_pr, train_sg, train_pr


def save_csv(csv_path, csv_list):
    with open(csv_path, mode="w", newline="") as opened_csv:
        writer = csv.writer(opened_csv)
        for row in csv_list:
            writer.writerow((row,))


def main():
    file_dir=Path("C:\\personal_ML\\music-transcription\\save\\")
    sg_files = [x for x in file_dir.iterdir() if "spec" in str(x)]
    val_sg, val_pr, test_sg, test_pr, train_sg, train_pr = partition_datasets(sg_files=sg_files, val_prop=0.1)
    save_csv(csv_list=val_sg, csv_path=Path("C:\\personal_ML\\music-transcription\\val_sg.csv"))
    save_csv(csv_list=val_pr, csv_path=Path("C:\\personal_ML\\music-transcription\\val_pr.csv"))
    save_csv(csv_list=test_sg, csv_path=Path("C:\\personal_ML\\music-transcription\\test_sg.csv"))
    save_csv(csv_list=test_pr, csv_path=Path("C:\\personal_ML\\music-transcription\\test_pr.csv"))
    save_csv(csv_list=train_sg, csv_path=Path("C:\\personal_ML\\music-transcription\\train_sg.csv"))
    save_csv(csv_list=train_pr, csv_path=Path("C:\\personal_ML\\music-transcription\\train_pr.csv"))



if __name__ == "__main__":
    main()
