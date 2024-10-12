from typing import Type, Dict
from transformers import TrainingArguments, TrainerState, TrainerControl, TrainerCallback


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
