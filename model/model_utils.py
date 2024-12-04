
from typing import Type, Dict
from transformers import TrainingArguments, TrainerState, TrainerControl, TrainerCallback, GPT2LMHeadModel


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int, threshold: float):
        self.patience = patience
        self.threshold = threshold
        self.patience_counter = 0
        self.best_metric = None
        
    def on_evaluate(
            self, 
            args: TrainingArguments, 
            state: Type[TrainerState], 
            control: Type[TrainerControl], 
            metrics: Dict, 
            **kwargs: Dict
            ) -> None:
        eval_metric = metrics.get("eval_loss")
        if eval_metric is None:
            return
        
        if self.best_metric is None:
            self.best_metric = eval_metric
            return
        
        if eval_metric >= (self.best_metric - self.threshold):
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                control.should_training_stop = True
        else:
            self.patience_counter = 0
            self.best_metric = eval_metric


def model_summary(model: Type[GPT2LMHeadModel]) -> None:
    trainable = 0
    frozen = 0
    for param in model.parameters():
        num_params = 1
        for param_dim in param.shape:
            num_params *= param_dim
        if param.requires_grad:
            trainable += num_params
        else:
            frozen += num_params

    print(f"Number of trainable parameters: {trainable}")
    print(f"Number of non-trainable parameters: {frozen}")


def enable_all_parameters(model: Type[GPT2LMHeadModel]) -> None:
    """
    enables all parameters in a model to be adjusted during training
    """
    for param in model.parameters():
        param.requires_grad = True
    
