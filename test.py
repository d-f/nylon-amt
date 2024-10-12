from dataset.data_utils import open_csv, PianoRollDataset
from train import define_model
from pathlib import Path
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from typing import Type
from model.nylon_gpt import NylonGPT
from dataset.data_utils import PianoRollDataset, collate_fn


def test_model(model: type[NylonGPT], test_ds: Type[PianoRollDataset], device: Type[torch.device], batch_size: int) -> torch.tensor:
    """
    defines accuracy as the proportion of correctly predicted piano roll notes
    predictions are considered incorrect if they extend beyond the ground truth
    """
    test_dl = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn)
    with torch.no_grad():
        acc = 0
        for i, output_dict in tqdm(enumerate(test_dl)):
            x = output_dict['x'].to(device)
            labels = output_dict["labels"].to(device).squeeze(1)
            
            predictions = model(x).transpose(1, 2)
            predictions = (torch.sigmoid(predictions) > 0.5).float()

            if predictions.shape[2] > labels.shape[2]:
                overlap = predictions[:, :, :labels.shape[2]]
                incorrect = predictions[:, :, labels.shape[2]:].numel()
                
                correct_mask = (overlap == labels)
                
                correct = correct_mask.sum()
                incorrect += (correct_mask == 0).sum()
                
                acc += correct / (incorrect + correct)

            elif predictions.shape[2] == labels.shape[2]:
                correct_mask = (predictions == labels)
                
                correct = correct_mask.sum()
                incorrect = (correct_mask == 0).sum()
                
                acc += correct / (incorrect + correct)

            else:
                overlap = labels[:, :, :predictions.shape[2]]
                incorrect = labels[:, :, predictions.shape[2]:].numel()

                correct_mask = (overlap == predictions)

                correct = correct_mask.sum()
                incorrect += (correct_mask == 0).sum()
                
                acc += correct / (incorrect + correct)

        acc /= i
    return acc


def load_model(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)


def parse_cla() -> Type[argparse.Namespace]:
    """
    parses command line arguments
    """
    parser = argparse.ArgumentParser()
    # maximum value found in the piano roll dataset
    parser.add_argument("-pr_max", type=int, default=204) 
    # length of the audio segments
    parser.add_argument("-spec_len", type=int, default=100+1) 
    # number of features for piano roll (+1 for eos)
    parser.add_argument("-pr_dim", type=int, default=129)
    # number of spectrogram features 
    parser.add_argument("-sg_dim", type=int, default=40) 
    # size of the embedding dimension
    parser.add_argument("-embed_dim", type=int, default=500) 
    # max number of tokens generated
    parser.add_argument("-max_gen", type=int, default=256) 
    # batch size
    parser.add_argument("-bs", type=int, default=25) 
    # folder to save model checkpoint to
    parser.add_argument("-output_dir", type=str, default="./model_2_results") 
    # dimensionality of feed forward layers in transformer
    parser.add_argument("-n_inner", type=int, default=1024) 
    # number of hidden layers in the transformer
    parser.add_argument("-n_layer", type=int, default=5)
    # number of attention heads 
    parser.add_argument("-n_head", type=int, default=5) 
    # maximum sequence length
    parser.add_argument("-n_positions", type=int, default=256) 
    # folder that contains dataset csvs
    parser.add_argument("-csv_dir", type=Path, default=Path("C:\\personal_ML\\nylon_gpt\\dataset_csv\\")) 
    # folder with model checkpoint
    parser.add_argument(
        "-model_dir", 
        type=Path, 
        default=Path("C:\\personal_ML\\nylon_gpt\\training_results\\model_2_results\\checkpoint-3171\\")
        )
    return parser.parse_args()


def main():
    args = parse_cla()
    device = torch.device("cuda")
    test_sg = open_csv(args.csv_dir.joinpath("test_sg.csv"))
    test_pr = open_csv(args.csv_dir.joinpath("test_pr.csv"))

    test_ds = PianoRollDataset(device=device, pr_files=test_pr, sg_files=test_sg, pr_max=args.pr_max)
    model = define_model(
        spec_len=args.spec_len, 
        pr_dim=args.pr_dim, 
        embed_dim=args.embed_dim, 
        sg_dim=args.sg_dim, 
        device=device, 
        max_gen=args.max_gen,
        n_inner=args.n_inner,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_positions=args.n_positions
        ).to(device)

    load_model(model=model, checkpoint_path=args.model_dir.joinpath("pytorch_model.bin"))

    acc = test_model(model=model, test_ds=test_ds, device=device, batch_size=args.bs)
    print("accuracy:", acc)


if __name__ == "__main__":
    main()
