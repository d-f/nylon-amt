from typing import Type
import torch
from transformers import GPT2LMHeadModel
import torch.nn as nn


class NylonGPT(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            pr_dim: int, 
            gpt: Type[GPT2LMHeadModel], 
            embed_dim: int, 
            sg_dim: int, 
            device: Type[torch.device], 
            max_token_gen: int
            ) -> None:
        
        super(NylonGPT, self).__init__()
        self.input_dim = input_dim
        self.gpt = gpt
        self.sg_embed = nn.Sequential(
            nn.Linear(sg_dim, embed_dim),
            nn.LayerNorm(embed_dim), 
            nn.Dropout(0.2)
        )
        self.pr_embed = nn.Sequential(
            nn.Linear(pr_dim, embed_dim),
            nn.LayerNorm(embed_dim),  
            nn.Dropout(0.2)
        )
        self.embed_dim = embed_dim
        self.device = device
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        self.gpt.to(device)
        self.pr_embed.to(device)
        self.max_token_gen = max_token_gen

        self.eos_token = torch.zeros(pr_dim).to(device)
        self.eos_token[-1] = 1
        
    def create_padding_mask(self, labels: torch.tensor) -> torch.tensor:
        # creates a mask where -1 values are 0 (ignored) and others are 1
        return (labels != -1).float()
    
    def embed_pr(self, x: torch.tensor) -> torch.tensor:
        """
        embed the piano roll [batch, 129] -> [batch, embed_dim]
        """
        return self.pr_embed(x)
    
    def embed_sg(self, x: torch.tensor) -> torch.tensor:
        """
        embed spectrogram [batch, 1, 40, n] -> [batch, embed_dim, n]
        """
        x = x.squeeze(1) 
        # transpose so that feature dimenions match along piano roll and spectrogram projections
        x = x.transpose(1, 2) 
        embedded = self.sg_embed(x)
        return embedded
    
    def forward(self, x: torch.tensor, labels=None):
        """
        model forward function, takes in spectrogram and outputs piano roll
        """
        batch_size = x.shape[0]
        # embed spectrogram
        sg_embedded = self.embed_sg(x)
        
        outputs = []
        total_loss = 0.0
        num_valid_tokens = 0
        # keep track of which sequences have finished in the batch
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool).to(self.device)

        if labels is not None:
            # pad labels to ignore -1
            seq_length = labels.size(3)
            padding_mask = self.create_padding_mask(labels)
        else:
            seq_length = self.max_token_gen

        for t in range(seq_length):
            if sg_embedded.size(1) > self.gpt.config.n_positions:
                sg_embedded = sg_embedded[:, -self.gpt.config.n_positions:, :]

            # create padding mask for attention to ignore -1
            attention_mask = None
            if labels is not None:
                attention_mask = (sg_embedded.sum(dim=-1) != 0).float()

            gpt_output = self.gpt(
                inputs_embeds=sg_embedded,
                attention_mask=attention_mask
            ).logits
            
            next_token_logits = gpt_output[:, -1, :]
            outputs.append(next_token_logits)

            if labels is None:
                # inference mode
                next_token_preds = (torch.sigmoid(next_token_logits) > 0.5).float()
                eos_detected = torch.all(torch.abs(next_token_preds - self.eos_token) < 1e-6, dim=1)
                finished_sequences = finished_sequences | eos_detected
                if torch.all(finished_sequences):
                    break
                next_token = next_token_preds
            else:
                # teacher forcing
                next_token = labels.squeeze(1)[:, :, t]
                
                # calculate loss only for non-padded tokens
                step_loss = self.loss_fn(next_token_logits, next_token)
                mask = padding_mask.squeeze(1)[:, :, t]
                masked_loss = step_loss * mask
                
                # sum the masked losses and count valid tokens
                total_loss += masked_loss.sum()
                num_valid_tokens += mask.sum()

            next_token_embed = self.embed_pr(next_token).unsqueeze(1)
            sg_embedded = torch.cat([sg_embedded, next_token_embed], dim=1)

        outputs = torch.stack(outputs, dim=1)

        if labels is not None:
            return (total_loss / num_valid_tokens if num_valid_tokens > 0 else total_loss), outputs
        return outputs
    