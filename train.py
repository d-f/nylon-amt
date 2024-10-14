from pathlib import Path
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Config
from typing import Type
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
from model.nylon_gpt import NylonGPT
from model.model_utils import EarlyStoppingCallback
from dataset.data_utils import PianoRollDataset, collate_fn, open_csv


def define_model(
        spec_len: int, 
        pr_dim: int, 
        embed_dim, 
        sg_dim, 
        device, 
        max_gen,
        n_inner: int,
        n_layer: int,
        n_head: int,
        n_positions: int,
        ) -> Type[GPT2LMHeadModel]:
    """
    returns GPT2 model
    """
    config = GPT2Config.from_pretrained('gpt2')
    config.n_inner = n_inner
    config.n_layer = n_layer
    config.n_head = n_head
    config.n_positions = n_positions
    config.n_embd = embed_dim
    gpt_model = GPT2LMHeadModel(config)
    gpt_model.resize_token_embeddings(new_num_tokens=pr_dim)
    gpt_model.gradient_checkpointing_enable()
    nylon_gpt = NylonGPT(
        input_dim=spec_len, 
        pr_dim=pr_dim, 
        gpt=gpt_model, 
        embed_dim=embed_dim, 
        sg_dim=sg_dim, 
        device=device, 
        max_token_gen=max_gen
        )
    
    return nylon_gpt


def train(
        num_epochs: int,
        patience: int,
        patience_thresh: float,
        batch_size: int,
        output_dir: str,
        lr: float,
        model: Type[NylonGPT],
        train_ds: Type[PianoRollDataset],
        val_ds: Type[PianoRollDataset],
        resume_from_checkpoint=None
        ) -> None:
    
    early_stopping_callback = EarlyStoppingCallback(
        patience=patience,
        threshold=patience_thresh
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_strategy="epoch",
        logging_first_step=False,
        learning_rate=lr,
        optim="adamw_hf",
        max_grad_norm=1.0,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1, 
        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss",
        save_safetensors=False,      
        push_to_hub=False,        
    )
    trainer = Trainer(
        model=model,
        args=training_args, 
        train_dataset=train_ds,
        data_collator=collate_fn,
        eval_dataset=val_ds,
        callbacks=[early_stopping_callback]
    )
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()


def parse_cla() -> Type[argparse.Namespace]:
    """
    parses command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-pr_max", type=int, default=204) # maximum value found in the piano roll dataset
    parser.add_argument("-spec_len", type=int, default=256) # length of the audio segments
    parser.add_argument("-pr_dim", type=int, default=129) # number of features for piano roll (+1 for eos)
    parser.add_argument("-sg_dim", type=int, default=40) # number of spectrogram features
    parser.add_argument("-embed_dim", type=int, default=128) # size of the embedding dimension
    parser.add_argument("-lr", type=float, default=1e-4) # learning rate
    parser.add_argument("-bs", type=int, default=40) # batch size
    parser.add_argument("-patience", type=int, default=5) # number of times to allow for stopping criteria to be met consecutively
    parser.add_argument("-patience_thresh", type=float, default=0.01) # threshold to consider loss having improved or not
    parser.add_argument("-output_dir", type=str, default="C:\\personal_ML\\nylon_gpt\\training_results\\model_3_results") # folder to save model checkpoint to
    parser.add_argument("-num_epochs", type=int, default=2) # number of training iterations
    parser.add_argument("-n_inner", type=int, default=128) # dimensionality of feed forward layers in transformer
    parser.add_argument("-n_layer", type=int, default=1) # number of hidden layers in the transformer
    parser.add_argument("-n_head", type=int, default=2) # number of attention heads
    parser.add_argument("-n_positions", type=int, default=350) # maximum generated sequence length
    parser.add_argument("-chkpt_num", type=int, default=None) # number on the checkpoint folder name e.g. checkpoint-100
    parser.add_argument("-csv_dir", type=Path, default=Path("C:\\personal_ML\\nylon_gpt\\dataset_csv\\")) # folder that contains dataset csvs
    return parser.parse_args()


def main():
    args = parse_cla()
    device = torch.device("cuda")
    train_sg = open_csv(args.csv_dir.joinpath("train_sg.csv"))
    train_pr = open_csv(args.csv_dir.joinpath("train_pr.csv"))
    val_sg = open_csv(args.csv_dir.joinpath("val_sg.csv"))
    val_pr = open_csv(args.csv_dir.joinpath("val_pr.csv"))

    train_ds = PianoRollDataset(device=device, pr_max=args.pr_max, sg_files=train_sg, pr_files=train_pr)
    val_ds = PianoRollDataset(device=device, pr_files=val_pr, sg_files=val_sg, pr_max=args.pr_max)
    model = define_model(
        spec_len=args.spec_len, 
        pr_dim=args.pr_dim, 
        embed_dim=args.embed_dim, 
        sg_dim=args.sg_dim, 
        device=device, 
        max_gen=args.n_positions,
        n_inner=args.n_inner,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_positions=args.n_positions
        )

    if args.chkpt_num:
        train(
            num_epochs=args.num_epochs,
            patience=args.patience,
            patience_thresh=args.patience_thresh,
            batch_size=args.bs,
            output_dir=args.output_dir,
            lr=args.lr,
            model=model,
            train_ds=train_ds, 
            val_ds=val_ds,
            resume_from_checkpoint=Path(args.output_dir).joinpath(f"checkpoint-{args.chkpt_num}")
            )
    else:
        train(
            num_epochs=args.num_epochs,
            patience=args.patience,
            patience_thresh=args.patience_thresh,
            batch_size=args.bs,
            output_dir=args.output_dir,
            lr=args.lr,
            model=model,
            train_ds=train_ds, 
            val_ds=val_ds,
            )


if __name__ == "__main__":
    main()
