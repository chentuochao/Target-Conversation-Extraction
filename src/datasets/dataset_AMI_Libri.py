"""
Torch dataset object for synthetically rendered
spatial data
"""
import json
import random

from typing import Tuple
from pathlib import Path

import torch
import numpy as np
import os

import src.utils as utils
from resemblyzer import VoiceEncoder, preprocess_wav, wav_to_mel_spectrogram

MAX_LEN = 50

class Dataset(torch.utils.data.Dataset):
    """
    Dataset of mixed waveforms and their corresponding ground truth waveforms
    recorded at different microphone.

    Data format is a pair of Tensors containing mixed waveforms and
    ground truth waveforms respectively. The tensor's dimension is formatted
    as (n_microphone, duration).

    Each scenario is represented by a folder. Multiple datapoints are generated per
    scenario. This can be customized using the points_per_scenario parameter.
    """
    def __init__(self, input_dir, n_mics=1, sr=8000,
                 sig_len = 30, downsample = 1,
                 split = 'val', output_conversation = 0, batch_size = 8, clean_embed=False, noise_dir = None):
        super().__init__()

        self.dirs = []
        for _dir in input_dir:
            dir_list = sorted(list(Path(_dir).glob('[0-9]*')))
            for dest in dir_list:
                meta_path = os.path.join(dest, 'metadata.json')
                embed_path = os.path.join(dest, 'embed.pt')
                if os.path.exists(meta_path) and os.path.exists(embed_path) :
                    self.dirs.append(dest)

        self.noise_dir = noise_dir
        self.noise_dirs = []
        if noise_dir is not None:
            dir_list = sorted(list(Path(noise_dir).glob('[0-9]*')))
            for dest in dir_list:
                self.noise_dirs.append(dest)
            
        # Physical params
        self.clean_embed = clean_embed
        self.n_mics = n_mics
        self.sig_len = int(sig_len*sr/downsample)
        self.sr = sr
        self.downsample = downsample
        self.scales = [-3, 3]
        self.output_conversation = output_conversation
        # Data augmentation
        ### calculate the stat
        self.batch_size = batch_size
        self.split = split
        print(self.split, (len(self.dirs)//batch_size)*batch_size)

    def __len__(self) -> int:
        return (len(self.dirs)//self.batch_size)*self.batch_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mixed_data - M x T
            target_voice_data - M x T
            window_idx_one_hot - 1-D
        """
        #print("Fetch idx: ", idx)
        curr_dir = self.dirs[idx%len(self.dirs)]
        return self.get_mixture_and_gt(curr_dir)


    def get_mixture_and_gt(self, curr_dir):
        if not self.clean_embed:
            metadata = utils.read_json(os.path.join(curr_dir, 'aug_meta.json'))
            aug = metadata["aug"]
        else:
            aug = [0]
        metadata2 = utils.read_json(os.path.join(curr_dir, 'metadata.json'))
        
        # print(curr_dir, metadata2["target_name"], metadata2["interfer_name"])
        # reverb0 = metadata2["room_info"]["absorption"], metadata2["room_info"]["max_order"]
        # print("reverb0", reverb0)
        self_speech = utils.read_audio_file_torch(os.path.join(curr_dir, 'self_speech.wav'), 1)
        if os.path.exists(os.path.join(curr_dir, f'intereference.wav')):
            interfere = utils.read_audio_file_torch(os.path.join(curr_dir, f'intereference.wav'), 1)
            scale = 0.8
        else:
            interfers = metadata2["interference"]
            interfere = torch.zeros_like(self_speech)
            for i in range(0, len(interfers)):
                interfere += utils.read_audio_file_torch(os.path.join(curr_dir, f'intereference{i}.wav'), 1)
            scale = 1

        if self.noise_dir is not None:
            # print(curr_dir)
            noise_dir = str(curr_dir).split('/')
            noise_dir = os.path.join(self.noise_dir, noise_dir[3], noise_dir[4])          
            # print(noise_dir)  
            BG = np.random.uniform(low = 0.5, high = 1.6 ) * utils.read_audio_file_torch(os.path.join(noise_dir, 'BG.wav'), 1)
            interfere += BG

        other_speech = torch.zeros_like(self_speech)
        if self.output_conversation:
            diags = metadata2["target_dialogue"]
            for i in range(len(diags) - 1):
                wav = utils.read_audio_file_torch(os.path.join(curr_dir, f'target_speech{i}.wav'), 1)
                other_speech += wav

        if self.split == "train":
            aug_id = np.random.choice([i for i in range(len(aug))])
        else:
            aug_id = 0


        other_speech = other_speech*scale
        self_speech = self_speech*scale
        gt = self_speech + other_speech
        mixture =  gt + interfere 
        
        # reverb1 = aug[aug_id]["room_info"]["absorption"], aug[aug_id]["room_info"]["max_order"]
        # print("reverb1", reverb1)
        MAX_FRAME = MAX_LEN*100 ## 10ms for each mel frame
        if self.clean_embed:
            reverb_path = os.path.join(curr_dir, f'embed.pt')
            L = 0
            example_mel = torch.zeros(1, 100)
        else:
            reverb_path = os.path.join(curr_dir, f'embed_aug{aug_id}.pt')
            example_wav = utils.read_audio_file_torch(os.path.join(curr_dir, f'example_aug{aug_id}.wav'), 1)
            example_wav = example_wav[0].numpy()
            example_wav = preprocess_wav(example_wav)
            example_mel = wav_to_mel_spectrogram(example_wav)
            example_mel = torch.from_numpy(example_mel)
            L = example_mel.shape[0]
            if L < MAX_FRAME:
                example_mel = torch.nn.functional.pad(example_mel, (0, 0, 0, MAX_FRAME - example_mel.shape[0]))
            else:
                example_mel = example_mel[:MAX_FRAME, :]
                L = MAX_FRAME

            # print(example_wav.shape, example_mel.shape)
        
        embed = torch.load(reverb_path)
        embed = torch.from_numpy(embed)

        inputs = {
            'mixture': mixture.float(),
            'embed': embed.float(),
            'example': example_mel.float(),
            'seq_len': torch.tensor(L, dtype = torch.long)
        }
        
        self.output_conversation
        targets = {
            'target': gt[0:1, :].float(),
            'self_speech': self_speech[0:1, :].float(),
            'other_speech': other_speech[0:1, :].float()
        }
        
        return inputs, targets

