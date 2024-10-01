import os
import sys

import argparse
import json
from typing import List
from numpy.random import randint, uniform
from pathlib import Path
import tqdm
import random
import glob
import multiprocessing.dummy as mp
import src.utils as utils
import scipy
import scipy.spatial 
import numpy as np
import librosa
import pyroomacoustics as pra
import soundfile as sf
from scipy.io.wavfile import write as wav_write
import matplotlib.pyplot as plt
import time

from datasets.dataset_loader import AMI_Dataset, ASR_Dataset
from datasets.dataset_util import overlap_duration_check

def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)

def handle_error(e):
    print("Error happen " + "!"*30)
    print(e)

def check_zeros(wav):
    assert(np.max(np.abs(wav)) > 1e-4)

def save_sample(idx, split, self_speech, target_speech, interfer, interfer_individuals, BG, dialogue, interference, noise_type, spk_info, room_info, reverb_example, room_info_ex, spk_info_ex, args, target_name, interfer_name, diag_info = None):
    sample_dir = os.path.join(args.save_dir, split, '{:05d}'.format(idx))
    # print(sample_dir)
    os.makedirs(sample_dir, exist_ok=True)
    
    meta_data = {}
    meta_data["target_name"] = target_name 
    meta_data["interfer_name"] = interfer_name

    ### save vctk
    if spk_info is not None:
        meta_data["spk_info"] = spk_info
    if room_info is not None:
        meta_data["room_info"] = room_info
    if room_info_ex is not None:
        meta_data["room_info_ex"] = room_info_ex
    if spk_info_ex is not None:
        meta_data["spk_info_ex"] = spk_info_ex

    meta_data["noise_type"] = noise_type
    if diag_info is not None:
        meta_data["diag_info"] = diag_info
    target_dialogue = dialogue
    meta_data["target_dialogue"] = []
    
    spk = target_dialogue["self_speech"]
    spk_meta = {}
    
    exits_spk = []

    if "conversation_id" in spk.keys():
        spk_meta["conversation_id"] = spk["conversation_id"]

    if "index" in spk.keys():
        spk_meta["index"] = spk["index"]

    spk_meta["spk_id"] = spk["spk_id"]
    exits_spk.append(spk["spk_id"])
    spk_meta["timestamp"] = spk["timestamp"]
    spk_meta["text"] = spk["text"]
    
    meta_data["target_dialogue"].append(spk_meta)
    

    for spk in target_dialogue["others"]:
        spk_meta = {}
        spk_meta["spk_id"] = spk["spk_id"]
        ### check the speaker is not appear in bother target and interfertece
        if spk["spk_id"] in exits_spk:
            print("Speaker overlap!!!!!!", spk["spk_id"], exits_spk)
        assert(spk["spk_id"] not in exits_spk)
        exits_spk.append(spk["spk_id"])
        spk_meta["timestamp"] = spk["timestamp"]
        spk_meta["text"] = spk["text"]

        meta_data["target_dialogue"].append(spk_meta)

    meta_data["interference"] = []
    for i, inter in enumerate(interference):
        spk_meta = {}
        spk_meta["spk_id"] = inter["spk_id"]
        if inter["spk_id"] in exits_spk:
            print("Speaker overlap!!!!!!", inter["spk_id"], exits_spk)
        assert(inter["spk_id"] not in exits_spk)
        exits_spk.append(inter["spk_id"])
        spk_meta["timestamp"] = inter["timestamp"]
        spk_meta["text"] = inter["text"]

        meta_data["interference"].append(spk_meta)

    # print(mixture.shape)
    # utils.write_audio_file(os.path.join(sample_dir, f"mixture.wav"), mixture, args.sr)
    check_zeros(self_speech)
    utils.write_audio_file(os.path.join(sample_dir, f"self_speech.wav"), self_speech, args.sr)
    for i in range(len(target_speech)):
        utils.write_audio_file(os.path.join(sample_dir, f"target_speech{i}.wav"), target_speech[i], args.sr)
    # utils.write_audio_file(os.path.join(sample_dir, f"intereference.wav"), interfer, args.sr)
    for i in range(len(interference)):
        utils.write_audio_file(os.path.join(sample_dir, f"intereference{i}.wav"), interfer_individuals[i], args.sr)

    if args.BG:
        utils.write_audio_file(os.path.join(sample_dir, f"BG.wav"), BG, args.sr)

    check_zeros(target_dialogue["self_speech"]["clean_example"])
    utils.write_audio_file(os.path.join(sample_dir, f"example.wav"), target_dialogue["self_speech"]["clean_example"], args.sr)
    if reverb_example is not None:
        utils.write_audio_file(os.path.join(sample_dir, f"example_reverb.wav"), reverb_example, args.sr)

    metadata_path = os.path.join(sample_dir, "metadata.json")
    with open(metadata_path, 'w', encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4)


def mixing(dialogue, intereference, noise, args, target_name, interfer_name):
    target_scales = [0.5, 0.8]
    interference_scales = [0.3, 0.6]

    voice_lists = []
    check_zeros(dialogue["self_speech"]["speech"])
    voice_lists.append(dialogue["self_speech"]["speech"]) ## self speech
    clean_example = dialogue["self_speech"]["clean_example"]
    check_zeros(clean_example)
    n_target = len(dialogue["others"])
    for dat in dialogue["others"]:
        voice_lists.append(dat["speech"]) ## target speech
    
    for interfer in intereference:
        voice_lists.append(interfer["speech"]) ## intereference speech
    

    ### whether apply reverberation augments
    reverberation = (np.random.rand() < args.reverb) 
    if reverberation: ### add reverberation to enhance the dataset
        voice_lists, room_info, spk_info = augment_reverbration(voice_lists, args, args.total_samples)
        # reverb_example, room_info_ex, spk_info_ex = augment_reverbration([clean_example], args, clean_example.shape[-1])
        # reverb_example = reverb_example[0]
    else:
        room_info, spk_info = None, None
    reverb_example, room_info_ex, spk_info_ex = None, None, None

    self_speech = voice_lists[0]*np.random.uniform(low = target_scales[0], high = target_scales[1])
    
    target_speech = []
    targets = np.zeros_like(self_speech)
    for i in range(1, n_target+1):
        sp = voice_lists[i]*np.random.uniform(low = target_scales[0], high = target_scales[1])
        target_speech.append(sp)
        targets += sp

    interfer = np.zeros_like(self_speech)
    interfer_individuals = np.zeros( (len(intereference), self_speech.shape[-1]))
    inter_i = 0
    for i in range(n_target + 1, len(voice_lists)):
        inte = voice_lists[i][0] *np.random.uniform(low = interference_scales[0], high = interference_scales[1])
        interfer_individuals[inter_i, :] = inte
        interfer += inte
        inter_i += 1

    # print(self_speech.shape, targets.shape, interfer.shape, noise.shape)
    mixture = self_speech + targets + interfer + noise
    
    scale = 1/np.abs(mixture).max()*np.random.uniform(low = 0.6, high = 0.9)

    mixture *= scale
    self_speech *= scale
    for i in range(n_target):
        target_speech[i] *= scale
    interfer *= scale
    noise *= scale
    interfer_individuals *= scale

    return self_speech, target_speech, interfer, interfer_individuals, noise, room_info, spk_info, reverb_example, room_info_ex, spk_info_ex




def generate_sample(idx, split, AMI_dataloader, ASR_dataloader, args, target_option, interference_option):
    ### self speech and target speech
    target_choice = []
    targer_prob = []
    for d in target_option.keys():
        target_choice.append(d)
        targer_prob.append(target_option[d])
    
    target_name = np.random.choice(target_choice, p = targer_prob)

    interfer_choice = []
    interfer_prob = []
    for d in interference_option.keys():
        interfer_choice.append(d)
        interfer_prob.append(interference_option[d])
    
    interfer_name = np.random.choice(interfer_choice, p = interfer_prob)


    '''
    dialogue: 
    "self_speech" = {
        "speech",
        "timestamp",
        "text",
        "spk_id"
        "clean_example"
    }
    "others" = [
        {
            "speech",
            "timestamp",
            "text",
            "spk_id"
        }
    ]


    interference = [
        {
            "speech",
            "timestamp",
            "text",
            "spk_id"
        }
    ]
    '''
    interference = []

    if target_name == "ASR" and interfer_name == "ASR":
        outputs = ASR_dataloader.get_conversation(2)
        target_dialogue = outputs[0]
        interfer_dialogue = outputs[1]
        interference.append(interfer_dialogue["self_speech"])
        for spk in interfer_dialogue["others"]:
            interference.append(spk)
    elif target_name == "AMI" and interfer_name == "AMI":
        outputs = AMI_dataloader.get_conversation(2)
        target_dialogue = outputs[0]
        interfer_dialogue = outputs[1]

        interference.append(interfer_dialogue["self_speech"])
        for spk in interfer_dialogue["others"]:
            interference.append(spk)
    else:
        raise ValueError("Invalid datase", target_name, interfer_name)        
    # print(target_name, interfer_name)

    # if len(interference) > 3:
    #     interference = interference[:3]

    ### background noise
    noise = np.zeros((1, args.total_samples))
    noise_type = "none"  

    # mix the self speech target and interference and noise
    diag_info = overlap_duration_check(target_dialogue)
    # if diag_info["overlap_ratio"] > 0.3:
    #     print("warning too much overlap", diag_info["overlap_ratio"],)
    self_speech, target_speech, interfer, interfer_individuals, BG, room_info, spk_info, reverb_example, room_info_ex, spk_info_ex  = mixing(target_dialogue, interference, noise, args, target_name, interfer_name )
    

    ### save the data to the disk
    save_sample(idx, split, self_speech, target_speech, interfer, interfer_individuals, BG, target_dialogue, interference, noise_type, spk_info, room_info, reverb_example, room_info_ex, spk_info_ex, args, target_name, interfer_name, diag_info)
    


def main(args: argparse.Namespace):
    seed_all(args.seed)
    args.total_samples = int(args.duration*args.sr)
    with open(args.config, 'rb') as f:
        data_config = json.load(f) 

    for split in ["train", "val", "test"]:
        n_outputs = getattr(args, "n_outputs_" + split)
        if n_outputs <= 0: 
            continue
        print(split)


        ### initialize the dataset
        dataset_name = data_config.keys()



        if "AMI" in dataset_name:
            AMI_dataloader = AMI_Dataset(data_config["AMI"], split, args) 
        else:
            AMI_dataloader = None 

        if "ASR" in dataset_name:
            ASR_dataloader = ASR_Dataset(data_config["ASR"], split, args) 
        else:
            ASR_dataloader = None 


        pbar = tqdm.tqdm(total=n_outputs)
        pool = mp.Pool(args.n_workers)
        callback_fn = lambda _: pbar.update()
        # for i in range(n_outputs):
        #     print(i, '-'*10)
        #     generate_sample(i, split, AMI_dataloader, LibriTTS_loader, MMCSG_dataloader, ASR_dataloader, AliMeeting_dataloader, noise_loader, args, data_config["target_prob"], data_config["interference_prob"])
        for i in range(n_outputs):
            pool.apply_async(generate_sample,
                            args=(i, split, AMI_dataloader,  ASR_dataloader, args, data_config["target_prob"], data_config["interference_prob"]),
                            callback=callback_fn,
                            error_callback=handle_error)
        pool.close()
        pool.join()
        pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help="dataset configuration files")

    parser.add_argument('save_dir',
                        type=str,
                        help="Directory to save files")

    parser.add_argument('--n_outputs_train', type=int, default=8000)
    parser.add_argument('--n_outputs_test', type=int, default=0)
    parser.add_argument('--n_outputs_val', type=int, default=1000)

    parser.add_argument('--n_workers', type=int, default=32)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--duration', type=float, default=60)
    parser.add_argument('--reverb', type=float, default=0)
    parser.add_argument('--BG', type=float, default=0)

    main(parser.parse_args())






