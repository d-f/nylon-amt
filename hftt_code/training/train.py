#! python
import json
import numpy as np
from tqdm import tqdm
import torch
import mir_eval


def reshape_for_mir_eval(onset_matrix, offset_matrix, 
                        hop_length=512, sample_rate=44100,
                        min_duration=0.032):

    time_per_frame = hop_length / sample_rate
    
    intervals = []
    pitches = []
    
    for batch_idx in range(onset_matrix.shape[0]):
        for pitch_idx in range(onset_matrix.shape[2]):
            onset_frames = np.where(onset_matrix[batch_idx, :, pitch_idx])[0]
            offset_frames = np.where(offset_matrix[batch_idx, :, pitch_idx])[0]
            
            if len(onset_frames) == 0 or len(offset_frames) == 0:
                continue
                
            for onset_frame in onset_frames:
                next_offsets = offset_frames[offset_frames > onset_frame]
                if len(next_offsets) == 0:
                    offset_frame = onset_frame + max(1, int(min_duration / time_per_frame))
                else:
                    offset_frame = next_offsets[0]
                
                if offset_frame <= onset_frame:
                    offset_frame = onset_frame + max(1, int(min_duration / time_per_frame))
                
                onset_time = onset_frame * time_per_frame
                offset_time = offset_frame * time_per_frame
                
                if offset_time - onset_time < min_duration:
                    offset_time = onset_time + min_duration
                
                frequency = 440 * (2 ** ((pitch_idx - 69) / 12))
                
                intervals.append([onset_time, offset_time])
                pitches.append(frequency)
    
    if not intervals:  
        return np.array([[0, min_duration]]), np.array([440.0]) 
        
    intervals = np.array(intervals)
    pitches = np.array(pitches)
    
    valid_idx = (intervals[:, 1] - intervals[:, 0]) > 0
    intervals = intervals[valid_idx]
    pitches = pitches[valid_idx]
    
    return intervals, pitches


##
## train
##
def train(model, iterator, optimizer,
          criterion_onset_A, criterion_offset_A, criterion_mpe_A, criterion_velocity_A,
          criterion_onset_B, criterion_offset_B, criterion_mpe_B, criterion_velocity_B,
          weight_A, weight_B,
          device, verbose_flag):
    model.train()
    epoch_loss = 0
    
    for i, (input_spec, label_onset, label_offset, label_mpe, label_velocity) in tqdm(enumerate(iterator), total=len(iterator)):
        input_spec = input_spec.to(device, non_blocking=True)
        label_onset = label_onset.to(device, non_blocking=True)
        label_offset = label_offset.to(device, non_blocking=True)
        label_mpe = label_mpe.to(device, non_blocking=True)
        label_velocity = label_velocity.to(device, non_blocking=True)
        # input_spec: [batch_size, n_bins, margin_b+n_frame+margin_f] (8, 256, 192)
        # label_onset: [batch_size, n_frame, n_note] (8, 128, 88)
        # label_velocity: [batch_size, n_frame, n_note] (8, 128, 88)
        if verbose_flag is True:
            print('***** train i : '+str(i)+' *****')
            print('(1) input_spec  : '+str(input_spec.size()))
            print(input_spec)
            print('(1) label_mpe   : '+str(label_mpe.size()))
            print(label_mpe)
            print('(1) label_velocity : '+str(label_velocity.size()))
            print(label_velocity)

        optimizer.zero_grad()
        output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = model(input_spec)
        # output_onset_A: [batch_size, n_frame, n_note] (8, 128, 88)
        # output_onset_B: [batch_size, n_frame, n_note] (8, 128, 88)
        # output_velocity_A: [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)
        # output_velocity_B: [batch_size, n_frame, n_note, n_velocity] (8, 128, 88, 128)

        if verbose_flag is True:
            print('(2) output_onset_A : '+str(output_onset_A.size()))
            print(output_onset_A)
            print('(2) output_onset_B : '+str(output_onset_B.size()))
            print(output_onset_B)
            print('(2) output_velocity_A : '+str(output_velocity_A.size()))
            print(output_velocity_A)
            print('(2) output_velocity_B : '+str(output_velocity_B.size()))
            print(output_velocity_B)

        output_onset_A = output_onset_A.contiguous().view(-1)
        output_offset_A = output_offset_A.contiguous().view(-1)
        output_mpe_A = output_mpe_A.contiguous().view(-1)
        output_velocity_A_dim = output_velocity_A.shape[-1]
        output_velocity_A = output_velocity_A.contiguous().view(-1, output_velocity_A_dim)

        output_onset_B = output_onset_B.contiguous().view(-1)
        output_offset_B = output_offset_B.contiguous().view(-1)
        output_mpe_B = output_mpe_B.contiguous().view(-1)
        output_velocity_B_dim = output_velocity_B.shape[-1]
        output_velocity_B = output_velocity_B.contiguous().view(-1, output_velocity_B_dim)

        # output_onset_A: [batch_size * n_frame * n_note] (90112)
        # output_onset_B: [batch_size * n_frame * n_note] (90112)
        # output_velocity_A: [batch_size * n_note * n_frame, n_velocity] (90112, 128)
        # output_velocity_B: [batch_size * n_note * n_frame, n_velocity] (90112, 128)

        if verbose_flag is True:
            print('(3) output_onset_A : '+str(output_onset_A.size()))
            print('(3) output_onset_B : '+str(output_onset_B.size()))
            print('(3) output_velocity_A : '+str(output_velocity_A.size()))
            print('(3) output_velocity_B : '+str(output_velocity_B.size()))

        label_onset = label_onset.contiguous().view(-1)
        label_offset = label_offset.contiguous().view(-1)
        label_mpe = label_mpe.contiguous().view(-1)
        label_velocity = label_velocity.contiguous().view(-1)
        # label_onset: [batch_size * n_frame * n_note] (90112)
        # label_velocity: [batch_size * n_frame * n_note] (90112)
        if verbose_flag is True:
            print('(4) label_onset   :'+str(label_onset.size()))
            print(label_onset)
            print('(4) label_velocity   :'+str(label_velocity.size()))
            print(label_velocity)

        loss_onset_A = criterion_onset_A(output_onset_A, label_onset)
        loss_offset_A = criterion_offset_A(output_offset_A, label_offset)
        loss_mpe_A = criterion_mpe_A(output_mpe_A, label_mpe)
        loss_velocity_A = criterion_velocity_A(output_velocity_A, label_velocity)
        loss_A = loss_onset_A + loss_offset_A + loss_mpe_A + loss_velocity_A

        loss_onset_B = criterion_onset_B(output_onset_B, label_onset)
        loss_offset_B = criterion_offset_B(output_offset_B, label_offset)
        loss_mpe_B = criterion_mpe_B(output_mpe_B, label_mpe)
        loss_velocity_B = criterion_velocity_B(output_velocity_B, label_velocity)
        loss_B = loss_onset_B + loss_offset_B + loss_mpe_B + loss_velocity_B

        loss = weight_A * loss_A + weight_B * loss_B
        if verbose_flag is True:
            print('(5) loss:'+str(loss.size()))
            print(loss)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


##
## validation
##
def valid(model, iterator,
          criterion_onset_A, criterion_offset_A, criterion_mpe_A, criterion_velocity_A,
          criterion_onset_B, criterion_offset_B, criterion_mpe_B, criterion_velocity_B,
          weight_A, weight_B,
          device,
          metrics
          ):
    model.eval()
    epoch_loss = 0
    
    if metrics:
        precision = 0
        recall = 0
        f1 = 0

    with torch.no_grad():
        for i, (input_spec, label_onset, label_offset, label_mpe, label_velocity) in enumerate(tqdm(iterator)):
            input_spec = input_spec.to(device, non_blocking=True)
            label_onset = label_onset.to(device, non_blocking=True)
            label_offset = label_offset.to(device, non_blocking=True)
            label_mpe = label_mpe.to(device, non_blocking=True)
            label_velocity = label_velocity.to(device, non_blocking=True)

            output_onset_A, output_offset_A, output_mpe_A, output_velocity_A, attention, output_onset_B, output_offset_B, output_mpe_B, output_velocity_B = model(input_spec)

            if metrics:
                est_int, est_pitch = reshape_for_mir_eval(onset_matrix=output_onset_B.detach().cpu().numpy(), offset_matrix=output_offset_B.detach().cpu().numpy())
                ref_int, ref_pitch = reshape_for_mir_eval(onset_matrix=label_onset.detach().cpu().numpy(), offset_matrix=label_onset.detach().cpu().numpy())

                scores = mir_eval.transcription.evaluate(ref_int, ref_pitch, est_int, est_pitch)
                precision += scores["Precision"]
                recall += scores["Recall"]
                f1 += scores["F-measure"]

            output_onset_A = output_onset_A.contiguous().view(-1)
            output_offset_A = output_offset_A.contiguous().view(-1)
            output_mpe_A = output_mpe_A.contiguous().view(-1)
            output_velocity_A_dim = output_velocity_A.shape[-1]
            output_velocity_A = output_velocity_A.contiguous().view(-1, output_velocity_A_dim)

            output_onset_B = output_onset_B.contiguous().view(-1)
            output_offset_B = output_offset_B.contiguous().view(-1)
            output_mpe_B = output_mpe_B.contiguous().view(-1)
            output_velocity_B_dim = output_velocity_B.shape[-1]
            output_velocity_B = output_velocity_B.contiguous().view(-1, output_velocity_B_dim)

            label_onset = label_onset.contiguous().view(-1)
            label_offset = label_offset.contiguous().view(-1)
            label_mpe = label_mpe.contiguous().view(-1)
            label_velocity = label_velocity.contiguous().view(-1)

            loss_onset_A = criterion_onset_A(output_onset_A, label_onset)
            loss_offset_A = criterion_offset_A(output_offset_A, label_offset)
            loss_mpe_A = criterion_mpe_A(output_mpe_A, label_mpe)
            loss_velocity_A = criterion_velocity_A(output_velocity_A, label_velocity)
            loss_A = loss_onset_A + loss_offset_A + loss_mpe_A + loss_velocity_A

            loss_onset_B = criterion_onset_B(output_onset_B, label_onset)
            loss_offset_B = criterion_offset_B(output_offset_B, label_offset)
            loss_mpe_B = criterion_mpe_B(output_mpe_B, label_mpe)
            loss_velocity_B = criterion_velocity_B(output_velocity_B, label_velocity)
            loss_B = loss_onset_B + loss_offset_B + loss_mpe_B + loss_velocity_B

            loss = weight_A * loss_A + weight_B * loss_B

            epoch_loss += loss.item()
    
    if metrics:
        precision /= len(iterator)
        recall /= len(iterator)
        f1 /= len(iterator)

        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)

        save_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        with open("test_performance.json", mode="w") as opened_json:
            json.dump(save_dict, opened_json)


    return epoch_loss, len(iterator)
