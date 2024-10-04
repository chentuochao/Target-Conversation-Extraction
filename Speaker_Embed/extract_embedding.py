"""
Generate NEMO embeddings with titanet with audio samples in the format:

root_dir
|-- subset 0
│   |-- speaker0
│   │   ├── audio0.wav
│   │   ├── audio1.wav
│   │   ├── ...
│   |-- speaker1
│   │   ├── audio0.wav
│   │   ├── audio1.wav
│   │   ├── ...
│   ├── ...
|-- subset 1
│   |-- speaker0
│   │   ├── audio0.wav
│   │   ├── audio1.wav
│   │   ├── ...
|   |-- speaker1
...
"""

import argparse
import os

import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from tqdm import tqdm

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import os
from datasets.WhitePinkBrownPerturbation import get_colored_noise
from datasets.dataset_util import augment_reverbration
import librosa
import src.utils as utils
from sklearn.metrics.pairwise import cosine_similarity
import json
import multiprocessing.dummy as mp


def handle_error(e):
    print("Error happen " + "!"*30)
    print(e)

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

def check_zeros(wav):
    assert(np.max(np.abs(wav)) > 1e-4)


def aug_folder(dset, s, speaker_model, args):
    sample_num = int(str(dset).split('/')[-1])
    # print(f'Processing {dset}...')
    meta_path = os.path.join(dset, 'metadata.json')
    if not os.path.exists(meta_path):
        print(dset, "Not exist")
        return
    metadata = utils.read_json(meta_path)

    audio_path = os.path.join(dset, "example.wav")
    wav = preprocess_wav(audio_path)
    check_zeros(wav)
    emb = speaker_model.embed_utterance(wav)
    torch.save(emb, os.path.join(dset, 'embed.pt'))

    if os.path.exists(os.path.join(dset, "example_denoised.wav")):
        audio_path = os.path.join(dset, "example_denoised.wav")
        wav = preprocess_wav(audio_path)
        check_zeros(wav)
        emb = speaker_model.embed_utterance(wav)
        torch.save(emb, os.path.join(dset, 'embed_denoised.pt'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/MixLibriSpeech/librispeech_scaper_fmt')
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--aug', type=int, default=0)

    args = parser.parse_args()

    # Load model
    speaker_model = VoiceEncoder()
    speaker_model.eval()
    print(speaker_model.device)

    

    split = ["train"]#, "val"]
    
    for s in split:
        split_dir = os.path.join(args.data_dir, s)

        dirs = sorted(list(Path(split_dir).glob('[0-9]*')))
        pbar = tqdm(total=len(dirs))
        pool = mp.Pool(8)
        callback_fn = lambda _: pbar.update()

        for dset in dirs:
            num = str(dset).split('/')[-1]
            #if int(num) < 7000:
            #    continue
            #print(int(num) )
            pool.apply_async(aug_folder,
                            args=(dset, s, speaker_model, args ),
                            callback=callback_fn,
                            error_callback=handle_error)
        pool.close()
        pool.join()
        pbar.close()
       
if __name__ == '__main__':
    main()
