import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn import TransformerEncoderLayer, TransformerEncoder, MultiheadAttention


class SimplifiedTranscriptionModel(nn.Module):
    def __init__(self, input_dims, n_notes=88, hidden_dim=256):
        super().__init__()
        
        # Onset Detection Network
        self.onset_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        # Frame Prediction Network
        self.frame_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        # Calculate the size after CNN stacks
        self.cnn_output_size = self._get_cnn_output_size(input_dims)
        
        # Prediction heads
        self.onset_pred = nn.Sequential(
            nn.Linear(self.cnn_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_notes),
            nn.Sigmoid()
        )
        
        self.frame_pred = nn.Sequential(
            nn.Linear(self.cnn_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_notes),
            nn.Sigmoid()
        )

    def _get_cnn_output_size(self, input_dims):
        # Helper to calculate flattened size after CNNs
        dummy_input = torch.zeros(1, 1, *input_dims)
        dummy_output = self.onset_stack(dummy_input)
        return dummy_output.numel() // dummy_output.shape[0]

    def forward(self, x):
        # Split processing for onsets and frames
        onset_features = self.onset_stack(x)
        frame_features = self.frame_stack(x)
        
        # Flatten and predict
        onset_features = onset_features.view(onset_features.size(0), -1)
        frame_features = frame_features.view(frame_features.size(0), -1)
        
        onsets = self.onset_pred(onset_features)
        frames = self.frame_pred(frame_features)
        
        return onsets, frames

class Model_SPEC2MIDI(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder_spec2midi = encoder
        self.decoder_spec2midi = decoder
        self.gradient_checkpointing = False
        
    def forward(self, input_spec):
        if self.gradient_checkpointing and self.training:
            enc_vector = checkpoint(lambda x: self.encoder_spec2midi(x), input_spec)
            outputs = checkpoint(lambda x: self.decoder_spec2midi(x), enc_vector)
        else:
            enc_vector = self.encoder_spec2midi(input_spec)
            outputs = self.decoder_spec2midi(enc_vector)
        return outputs

class Encoder_SPEC2MIDI(nn.Module):
    def __init__(self, n_margin, n_frame, n_bin, cnn_channel, cnn_kernel, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.device = device
        self.n_frame = n_frame
        self.n_bin = n_bin
        self.cnn_channel = cnn_channel
        self.cnn_kernel = cnn_kernel
        self.hid_dim = hid_dim
        
        # Optimized CNN layer
        self.conv = nn.Conv2d(1, self.cnn_channel, kernel_size=(1, self.cnn_kernel))
        self.n_proc = n_margin * 2 + 1
        self.cnn_dim = self.cnn_channel * (self.n_proc - (self.cnn_kernel - 1))
        
        # Embeddings
        self.tok_embedding_freq = nn.Linear(self.cnn_dim, hid_dim)
        self.pos_embedding_freq = nn.Embedding(n_bin, hid_dim)
        
        # Replace custom encoder layers with optimized transformer
        encoder_layer = TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=n_heads,
            dim_feedforward=pf_dim,
            dropout=dropout,
            batch_first=True
        )
        self.layers_freq = TransformerEncoder(encoder_layer, n_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.scale_freq = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, spec_in):
        batch_size = spec_in.shape[0]
        
        # Use torch.unfold for efficient sliding window
        spec = spec_in.unfold(2, self.n_proc, 1).permute(0, 2, 1, 3).contiguous()
        
        # Optimize CNN operations
        spec_cnn = spec.reshape(-1, 1, self.n_bin, self.n_proc)
        spec_cnn = self.conv(spec_cnn)
        spec_cnn_freq = spec_cnn.permute(0, 2, 1, 3).reshape(-1, self.n_bin, self.cnn_dim)
        
        # Embeddings
        spec_emb_freq = self.tok_embedding_freq(spec_cnn_freq)
        pos_freq = torch.arange(0, self.n_bin, device=self.device).expand(batch_size*self.n_frame, -1)
        spec_freq = self.dropout((spec_emb_freq * self.scale_freq) + self.pos_embedding_freq(pos_freq))
        
        # Transformer encoding
        spec_freq = self.layers_freq(spec_freq)
        return spec_freq.reshape(batch_size, self.n_frame, self.n_bin, self.hid_dim)

class Decoder_SPEC2MIDI(nn.Module):
    def __init__(self, n_frame, n_bin, n_note, n_velocity, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.device = device
        self.n_note = n_note
        self.n_frame = n_frame
        self.n_velocity = n_velocity
        self.n_bin = n_bin
        self.hid_dim = hid_dim
        
        # Optimized attention modules
        self.freq_attention = MultiheadAttention(hid_dim, n_heads, dropout=dropout, batch_first=True)
        self.time_attention = MultiheadAttention(hid_dim, n_heads, dropout=dropout, batch_first=True)
        
        # Embeddings
        self.pos_embedding_freq = nn.Embedding(n_note, hid_dim)
        self.pos_embedding_time = nn.Embedding(n_frame, hid_dim)
        
        # Output layers
        self.fc_onset = nn.Linear(hid_dim, 1)
        self.fc_offset = nn.Linear(hid_dim, 1)
        self.fc_mpe = nn.Linear(hid_dim, 1)
        self.fc_velocity = nn.Linear(hid_dim, n_velocity)
        
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.scale_time = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, enc_spec):
        batch_size = enc_spec.shape[0]
        enc_spec = enc_spec.reshape(-1, self.n_bin, self.hid_dim)
        
        # Frequency attention
        pos_freq = torch.arange(0, self.n_note, device=self.device).expand(batch_size*self.n_frame, -1)
        midi_freq = self.pos_embedding_freq(pos_freq)
        
        midi_freq, attention_freq = self.freq_attention(
            midi_freq, enc_spec, enc_spec,
            need_weights=True
        )
        
        # Reshape attention for output
        attention_freq = attention_freq.view(batch_size, self.n_frame, -1, self.n_note, self.n_bin)
        
        # Generate frequency-based outputs
        midi_freq_reshaped = midi_freq.view(batch_size, self.n_frame, self.n_note, -1)
        output_onset_freq = self.sigmoid(self.fc_onset(midi_freq_reshaped).squeeze(-1))
        output_offset_freq = self.sigmoid(self.fc_offset(midi_freq_reshaped).squeeze(-1))
        output_mpe_freq = self.sigmoid(self.fc_mpe(midi_freq_reshaped).squeeze(-1))
        output_velocity_freq = self.fc_velocity(midi_freq_reshaped)
        
        # Time-based processing
        midi_time = midi_freq_reshaped.permute(0, 2, 1, 3).reshape(-1, self.n_frame, self.hid_dim)
        pos_time = torch.arange(0, self.n_frame, device=self.device).expand(batch_size*self.n_note, -1)
        midi_time = self.dropout((midi_time * self.scale_time) + self.pos_embedding_time(pos_time))
        
        midi_time, _ = self.time_attention(midi_time, midi_time, midi_time)
        
        # Generate time-based outputs
        midi_time = midi_time.view(batch_size, self.n_note, self.n_frame, -1).permute(0, 2, 1, 3)
        output_onset_time = self.sigmoid(self.fc_onset(midi_time).squeeze(-1))
        output_offset_time = self.sigmoid(self.fc_offset(midi_time).squeeze(-1))
        output_mpe_time = self.sigmoid(self.fc_mpe(midi_time).squeeze(-1))
        output_velocity_time = self.fc_velocity(midi_time)
        
        return (output_onset_freq, output_offset_freq, output_mpe_freq, output_velocity_freq, 
                attention_freq, output_onset_time, output_offset_time, output_mpe_time, 
                output_velocity_time)