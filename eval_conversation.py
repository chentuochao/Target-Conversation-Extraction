from src.metrics.metrics import compute_decay, Metrics
from src.utils import read_audio_file, write_audio_file
import src.utils as utils
import argparse
import os, json, glob
import numpy as np
import torch
import pandas as pd
from src.datasets.dataset_TSE import Dataset
import torchaudio
import time
from collections import Counter
from resemblyzer import VoiceEncoder, preprocess_wav, wav_to_mel_spectrogram
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import mir_eval
def save_audio_file_torch(file_path, wavform, sample_rate = 48000, rescale = True):
    if rescale:
        wavform = wavform/torch.max(wavform)*0.9
    torchaudio.save(file_path, wavform, sample_rate)

def overlap_duration_check(self_speaker, others, shift = 1):
    L_total = 16000*60
    self_time = self_speaker["timestamp"]
    if shift!= 0:
        for i in range(0, len(self_time)):
            self_time[i] = [self_time[i][0]+ shift, self_time[i][1] + shift]
    timestamps = [self_time]
    search_index = [0]
    for o in others:
        timestamps.append(o["timestamp"])
        search_index.append(0)
    
    step = 4000
    # print(len(timestamps), L_total)
    # print(timestamps[0])
    overlap_duration = 0
    valid_duration = 0
    
    diag_info = {}

    duration = []
    for stamp in timestamps:
        du = 0
        for b, e in stamp:
            du += (e - b) 
        duration.append(du)
    duration = np.array(duration)
    duration = duration/np.sum(duration)
    diag_info["occupation"] = duration.tolist() 
    
    for i in range(0, L_total, step):
        begin = i
        end = i + step

        spk_exits = []

        for j in range(len(timestamps)):
            while True:
                idx = search_index[j]
                if idx >= len(timestamps[j]):
                    break
                
                b, e = timestamps[j][idx]
                if end <= b:
                    break
                elif e <= begin:
                    search_index[j] += 1
                    continue
                else:
                    spk_exits.append(j)
                    if e < end:
                        search_index[j] += 1
                    break

        if len(spk_exits) > 1 :#and 0 in spk_exits:
            overlap_duration += step
        if len(spk_exits) > 0:
            valid_duration += step
    overlap_duration = overlap_duration/(valid_duration + 1e-5)
    return overlap_duration

def point_in_box(pos, left, right, top, bottom):
    return pos[0] >= left and pos[0] <= right and pos[1] <= top and pos[1] >= bottom

def calculate_cosine_similarity(vector1, vector2):
    # Ensure both vectors have the same shape
    if vector1.shape != vector2.shape:
        raise ValueError("Input vectors must have the same shape")

    # Reshape the vectors to ensure they are 2D arrays
    vector1 = np.reshape(vector1, (1, -1))
    vector2 = np.reshape(vector2, (1, -1))

    # Calculate cosine similarity
    similarity = cosine_similarity(vector1, vector2)

    return similarity[0, 0]


def get_mixture_and_gt(curr_dir, SHIFT_VALUE = 0):
    
    metadata2 = utils.read_json(os.path.join(curr_dir, 'metadata.json'))
    # print("reverb0", reverb0)
    diags = metadata2["target_dialogue"]
    

    label_path = os.path.join(curr_dir, 'self_label.npy')
    label_path2 = os.path.join(curr_dir, 'other_label.npy')
    if os.path.exists(label_path):
        self_label = np.load(label_path) 
        other_label = np.load(label_path2) 
        label_gt = np.stack([self_label, other_label], axis=0)
    else:
        self_label = None
        other_label = None
        label_gt = None
    
    self_speech = utils.read_audio_file_torch(os.path.join(curr_dir, 'self_speech.wav'), 1)

    other_speech = torch.zeros_like(self_speech)
    for i in range(len(diags) - 1):
        wav = utils.read_audio_file_torch(os.path.join(curr_dir, f'target_speech{i}.wav'), 1)
        other_speech += wav


    if os.path.exists(os.path.join(curr_dir, f'intereference.wav')):
        interfere = utils.read_audio_file_torch(os.path.join(curr_dir, f'intereference.wav'), 1)
    else:
        interfers = metadata2["interference"]
        interfere = torch.zeros_like(self_speech)
        Num_inter = len(interfers)

        for i in range(0, Num_inter):
            interfere += utils.read_audio_file_torch(os.path.join(curr_dir, f'intereference{i}.wav'), 1)

    overlap = overlap_duration_check(diags[0], diags[1:])
    if "diag_info" not in metadata2.keys():
        metadata2["diag_info"] = {}
        metadata2["target_name"], metadata2["interfer_name"] = "xxx", "xxx"
    
    interfer_overlap = []
    for interf in metadata2["interference"]:
        overlap1 = overlap_duration_check(diags[0], [interf])
        interfer_overlap.append(overlap1)
    # print("within conversation overlap", overlap,"interference overlap-", interfer_overlap)
    metadata2["diag_info"]["overlap_ratio"] = overlap
    metadata2["diag_info"]["overlap_ratio_inter"] = interfer_overlap[0]
    aug_id = 0
    # print("reverb1", reverb1)
    reverb_path = os.path.join(curr_dir, f'embed.pt')
    example_wav = utils.read_audio_file_torch(os.path.join(curr_dir, f'example.wav'), 1)


    L = example_wav.shape[-1]

    scale = 0.75
    other_speech = other_speech*scale
    self_speech = self_speech*scale
    gt = self_speech + other_speech
    mixture =  gt + interfere 
    
    embed = torch.load(reverb_path)
    embed = torch.from_numpy(embed)
    example_wav_torch = example_wav
    example_wav = example_wav[0].numpy()
    example_wav = preprocess_wav(example_wav)
    example_mel = wav_to_mel_spectrogram(example_wav)
    example_mel = torch.from_numpy(example_mel)

    inputs = {
        'mixture': mixture.float(),
        'embed': embed.float(),
        'example': example_mel.float(),
        'seq_len': torch.tensor(L, dtype = torch.long)
    }
    
    targets = {
        'self':self_speech[0:1, :].numpy(),
        'other':other_speech[0:1, :].numpy(),
        'target':gt[0:1, :].float(),
        'label_gt': label_gt    }
    
    return inputs, targets, metadata2, example_wav_torch, interfere



def run_basedline(inputs, timestamps):
    mix = inputs["mixture"]
    
    total_sample =mix.shape[-1]
    merged_times = []
    _b = timestamps[0][0]
    _e = timestamps[0][1]
    L = 0
    for i in range(1, len(timestamps) ):
        b, e = timestamps[i]
        if e >= total_sample:  
            e = total_sample
        if b <= _e + 8000:
            _e = e 
        else:
            merged_times.append([_b, _e])
            L += (_e - _b)
            _b = b
            _e = e
    merged_times.append([_b, _e])
    L += (_e - _b)

    masked = torch.ones_like(mix)
    for b, e in timestamps:
        masked[:, b : e ] = 0

    output = masked * mix

    return output.numpy()

def run_testcase(model, inputs, device) -> np.ndarray:
    with torch.no_grad():
        # Create tensor and copy it to the device
        inputs["mixture"] = inputs["mixture"].unsqueeze(0).to(device)
        inputs["embed"] = inputs["embed"].unsqueeze(0).to(device)
        inputs["example"] = inputs["example"].unsqueeze(0).to(device)
        inputs["seq_len"] = inputs["seq_len"].unsqueeze(0).to(device)
        outputs = model(inputs)
        output_target = outputs['output'].squeeze(0)
        if 'output_label' in outputs.keys():
            output_label = outputs['output_label'][0].cpu().numpy()
        else:
            output_label = None
        # Copy to cpu and convert to numpy array
        output_target = output_target.cpu().numpy()
        mixture = inputs["mixture"][0].cpu().numpy()
        
        return mixture, output_target, output_label

def main(args: argparse.Namespace):
    device = 'cuda' if args.use_cuda else 'cpu'
    
    np.random.seed(0)
    # test_set = Dataset([args.test_dir], reverb_embed = 1)

    # sample_dirs = sorted(glob.glob(os.path.join(args.test_dir, '*')))
    # Load model
    model = utils.load_torch_pretrained(args.run_dir).model
    model_name = args.run_dir.split('/')[-2]
    model = model.to(device)
    model.eval()
    
    # Initialize metrics
    snr = Metrics('snr')
    snr_i = Metrics('snr_i')
    
    si_snr = Metrics('si_snr')
    si_snr_i = Metrics('si_snr_i')
    
    si_sdr = Metrics('si_sdr')
    si_sdr_i = Metrics('si_sdr_i')

    # pesq = Metrics('PESQ')
    # stoi = Metrics('STOI') 633
    records = []

    sisdris = []
    sisdris2 = []
    sisdris3 = []
    num_pick_right = 0
    num_pick_wrong = 0
    tse_dataset = {
        "LibriTTS": [],
        "SpokenWoz": [],
        "AMI": [],
        "ASR": [],
        "AliMeeting": [],
        "xxx": []
    }
    bad_spk = []
    save_dir = [ 52, 68, 87] #[2, 5, 8, 10, 12, 17, 25, 28, 63, 94]
    x1 = []
    x2 = []
    y = []
    overlaps = []
    overlaps2 = []
    sisdris4 = []

    for i in range(0, 100):
        # print(f"Sample: {i} ----------")
        
        # curr_dir = os.path.join(args.test_dir, "{:05d}".format(i))
        curr_dir = os.path.join(args.test_dir, "{:05d}".format(i))
        embed_dir = os.path.join(curr_dir, "embed.pt")
        if not os.path.exists(embed_dir):
            continue
        inputs, targets, metadata, example_wav, interfer = get_mixture_and_gt(curr_dir, SHIFT_VALUE = args.shift)
        if metadata["interfer_name"] != metadata["target_name"]:
            continue
        # Run inference
        mixture, output_target, output_label = run_testcase(model, inputs, device)
        interfer = interfer.numpy()
        target_speech = targets['target'].numpy()
        row = {}
        # Compute SNR-based metrics for the case where there is at least one target speaker                       
        
        if output_label is not None:
            output_label = (output_label > 0.5)
            label_gt = targets['label_gt']
            eq_matrix = np.equal(output_label, label_gt)
            acc = np.sum(eq_matrix == 1)/(eq_matrix.size)
        else:
            acc = 0
        # Input SNR & SNR
        row['input_snr'] = snr(est=mixture[0:1], gt=target_speech, mix=mixture[0:1]).item()
        row['snri'] = snr_i(est=output_target, gt=target_speech, mix=mixture[0:1]).item()

        # Input SI-SDR & SI-SDRi

        row['input_sisdr_target'] = si_sdr(est=mixture[0:1], gt=target_speech, mix=mixture[0:1]).item()
        row['sisdri_target'] = si_sdr_i(est=output_target, gt=target_speech, mix=mixture[0:1]).item()


        mix_exclude = mixture[0:1] - targets["self"]
        output_exclude = output_target - targets["self"]

        row['sisdri_interfer_ex'] = si_sdr_i(est=output_exclude, gt=interfer, mix=mix_exclude).item()
        row['sisdri_other_ex'] = si_sdr_i(est=output_exclude, gt=targets["other"], mix=mix_exclude).item()

        row['snri_interfer'] = snr_i(est=output_target, gt=interfer + targets["self"], mix=mixture[0:1]).item()
        row['snri_other'] = snr_i(est=output_target, gt=targets["other"] + targets["self"], mix=mixture[0:1]).item()
        
        row['sisdri_interfer'] = si_sdr_i(est=output_target, gt=interfer, mix=mixture[0:1]).item()
        row['sisdri_other'] = si_sdr_i(est=output_target, gt=targets["other"], mix=mixture[0:1]).item()
        
        tse_dataset[metadata["interfer_name"]].append(row['sisdri_target'])
        diag_info = metadata["diag_info"]
        

        self_duration = 0
        for b, e in metadata["target_dialogue"][0]["timestamp"]:
            self_duration += (e - b)
        self_duration = self_duration/16000
        row["self_duration"] = self_duration
        row["overlap_ratio"] = diag_info["overlap_ratio"]
        row["overlap_ratio_inter"] = diag_info["overlap_ratio_inter"]
        overlaps.append(diag_info["overlap_ratio"])
        overlaps2.append(diag_info["overlap_ratio_inter"])
        x1.append( diag_info["overlap_ratio_inter"] - diag_info["overlap_ratio"])
        y.append(row['sisdri_target'])
        print(curr_dir, diag_info["overlap_ratio"],  self_duration, "Target speech: input sisdr = ", row['input_sisdr_target'])
        print("sisdr i= ", row['sisdri_target'], row['sisdri_other'], row['sisdri_interfer'])
        if row['snri_interfer'] > row['snri_other']: #row['sisdri_interfer'] > 0: #delta_sir[1] > 0:
            num_pick_wrong += 1
        # num_pick_wrong = 1
        
        if i in save_dir:
            bad_spk.append(metadata["target_dialogue"][0]["spk_id"])
            save_folder = f"../TCE_samples2/{i}"
            os.makedirs(save_folder, exist_ok=True)
            save_audio_file_torch(f"{save_folder}/mix.wav", torch.from_numpy(mixture[0:1]), sample_rate = args.sr,rescale = False)
            save_audio_file_torch(f"{save_folder}/example.wav", example_wav, sample_rate = args.sr,rescale = False)
            # save_audio_file_torch(f"{save_folder}/self.wav", torch.from_numpy(targets["self"]), sample_rate = args.sr,rescale = False)
            # save_audio_file_torch(f"{save_folder}/other.wav", torch.from_numpy(targets["other"],), sample_rate = args.sr,rescale = False)
            save_audio_file_torch(f"{save_folder}/output_target.wav", torch.from_numpy(output_target), sample_rate = args.sr,rescale = False)
            save_audio_file_torch(f"{save_folder}/interfer.wav", torch.from_numpy(interfer), sample_rate = args.sr,rescale = False)
            save_audio_file_torch(f"{save_folder}/target_speech.wav", torch.from_numpy(target_speech), sample_rate = args.sr,rescale = False)

        records.append(row)
        sisdris2.append(row['sisdri_target'])
        sisdris3.append(row['sisdri_target'] + row['input_sisdr_target'] )
        sisdris4.append(row['sisdri_interfer'])

    print(num_pick_wrong)
    print("sisdri = ", np.mean(sisdris2), "interfert = ", np.mean(sisdris4))
    print("sisdro = ", np.mean(sisdris3))
    print("overlaps", np.mean(overlaps),  np.std(overlaps), np.mean(overlaps2))

    results_df = pd.DataFrame.from_records(records)
    
    # Save DataFrame
    results_csv_path = f'./output/result_{model_name}.csv' 
    results_df.to_csv(results_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir',
                        type=str,
                        help="Path to test dataset")
    parser.add_argument('run_dir',
                        type=str,
                        help='Path to model run')

    parser.add_argument('--sr',
                        type=int,
                        default=16000,
                        help='Project sampling rate')



    parser.add_argument('--shift',
                        type=int,
                        default=0,
                        help='shift pertube')
    
    parser.add_argument('--use_cuda',
                        action='store_true',
                        help='Whether to use cuda')



    main(parser.parse_args())



