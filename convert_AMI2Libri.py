import numpy as np
import random
import os
import sys
import csv
import argparse
import json
from typing import List
from pathlib import Path
import multiprocessing.dummy as mp
import numpy as np
import librosa
import glob
import src.utils as utils
from scipy.signal import butter, lfilter
from tqdm import tqdm
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import torch

LibriTTS_config = {
    "train_folder": "/scr/LibriTTS/train-clean-360/",
    "val_folder": "/scr/LibriTTS/dev-clean/",
    "test_folder": "/scr/LibriTTS/test-clean/",
    "sr": 16000
}


def handle_error(e):
    print("Error happen " + "!"*30)
    print(e)


def high_pass_filter(wav, sampling_rate, order=7):
    CUTOFF = 50
    nyquist = 0.5 * sampling_rate
    normal_cutoff = CUTOFF / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    wav = lfilter(b, a, wav)
    return wav

def get_envelope(signal):
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope

def check_zeros(wav):
    assert(np.max(np.abs(wav)) > 1e-4)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


class LibriTTS_Dataset(object):
    def __init__(self, config, split):
        if split == "train":
            self.tts_list = self.list_tts_folders(config["train_folder"])
        elif split == "val":
            self.tts_list = self.list_tts_folders(config["val_folder"])
        elif split == "test":
            self.tts_list = self.list_tts_folders(config["test_folder"])
        else:
            raise ValueError("Invalid split")

        self.sr = config["sr"]
        self.split = split
        self.total_sample = self.sr * 60

    def list_tts_folders(self, directory):
        folder_names = {}

        for folder in os.listdir(directory):
            spk_director = os.path.join(directory, folder)
            tmp_list = []
            if not folder.isnumeric():
                continue
            if os.path.isdir(spk_director):
                for folder2 in os.listdir(spk_director):
                    section_director = os.path  .join(spk_director, folder2)
                    if folder2.isnumeric() and os.path.isdir(spk_director):
                        tmp_list.append(section_director)
            if len(tmp_list) > 0:
                folder_names[folder] = tmp_list
        return folder_names


    def replace_sample(self, timestamps, need_example, existing_speaker = [], interfer = False):
        tts_lists = list(self.tts_list.keys())
        speakers = np.random.choice(tts_lists, 10, replace=False) #random.sample(args.all_voices, n_voices + 5)
        output = np.zeros((1, self.total_sample))
        if interfer:
            scale = np.random.uniform(low = 0.3, high = 0.6)
        else:
            scale = np.random.uniform(low = 0.4, high = 0.8)
        ### merge the close timestamp
        merged_times = []
        _b = timestamps[0][0]
        _e = timestamps[0][1]
        L = 0
        for i in range(1, len(timestamps) ):
            b, e = timestamps[i]
            if e >= self.total_sample:  
                e = self.total_sample
            if b <= _e + 1600:
                _e = e 
            else:
                merged_times.append([_b, _e])
                L += (_e - _b)
                _b = b
                _e = e
        merged_times.append([_b, _e])
        L += (_e - _b)

        spk_pick = None
        speech_audio_pick = []
        speech_text_pick = [] 
        speech_duration_pick = [] 
        
        for speaker_id in speakers:
            files_lists = []
            voice_dir = self.tts_list[speaker_id]
            voice_identity = speaker_id.split('/')[-1]
            if "tts_" + voice_identity in existing_speaker:
                continue

            random.shuffle(voice_dir)
            for story_id in voice_dir:
                files = glob.glob(os.path.join(story_id, '*.wav'))
                files_lists.extend(files)

            total_length = 0
            speech_audio = []
            speech_text = []
            speech_duration = []

            for i in range(0, len(files_lists)):
                wav_raw, _ = librosa.load(files_lists[i], sr = self.sr)
                wav_raw = high_pass_filter(wav_raw, self.sr)
                wav_raw, _ = librosa.effects.trim(wav_raw, top_db=40)
                if wav_raw.shape[-1] < 1600:
                    print("Invalid get voice and retry")
                    continue
                wav_raw = wav_raw/np.max(np.abs(wav_raw)) * scale
                untter_name = files_lists[i].split(".")[0]
                text_name = untter_name + ".normalized.txt"
                with open(text_name) as f:
                    contents = f.read()
                ### append speech and interval
                speech_audio.append(wav_raw)
                speech_text.append(contents)
                speech_duration.append(wav_raw.shape[-1])
                total_length += wav_raw.shape[-1]

                if total_length >= L + need_example*50*self.sr + 8000:
                    break

            if total_length < L + need_example*30*self.sr + 8000:
                print("TTS too shore, try another")
                continue
            else:
                spk_pick = "tts_" + speaker_id
                speech_audio_pick = speech_audio
                speech_text_pick = speech_text 
                speech_duration_pick = speech_duration 
                break
        Libri_audio = np.concatenate(speech_audio_pick, axis = 0)
        wavform = moving_average(np.abs(Libri_audio), window_size = 200)
        wavform = np.roll(wavform, 200)
        assert(Libri_audio.shape[-1] == wavform.shape[-1])

        assert(L < Libri_audio.shape[-1])
        index = 0
        new_times = []
        for i in range(len(merged_times)):
            b, e = merged_times[i]
            # print(b, e, output.shape)
            assert(index + (e - b) < Libri_audio.shape[-1])
            
            search_begin = max([index + (e - b) - 800, index + 1600]  )
            search_end = min([index + (e - b) + 800, wavform.shape[-1]])
            if search_end > search_begin:
                valley_point = np.argmin(wavform[search_begin:search_end]) + search_begin
                assert(valley_point - index > 1200)
                if b + (valley_point - index) > output.shape[-1]:
                    output[:, b:e] = Libri_audio[index:index + (e - b)]
                    index = index + (e - b)
                    new_times.append([b, e])
                else:
                    output[:, b:b+(valley_point - index)] = Libri_audio[index:valley_point]
                    new_times.append([b, int(b+(valley_point - index))])
                    index = valley_point
                    
            else:
                output[:, b:e] = Libri_audio[index:index + (e - b)]
                index = index + (e - b)
                new_times.append([b, e])
        if need_example:
            example = Libri_audio[index + 1600:]
            assert(example.shape[-1] > 20*self.sr)
        else:
            example = None

        assert(spk_pick is not None)
        return spk_pick, example, output, new_times


def convert_sample(in_dir, out_dir, sample_id, dataset, sr = 16000, replace_prob = 0.5):
    curr_dir = os.path.join(in_dir, sample_id)
    save_dir = os.path.join(out_dir, sample_id)
    os.makedirs(save_dir, exist_ok=True)
    
    metadata = utils.read_json(os.path.join(curr_dir, 'metadata.json'))
    target =  metadata["target_dialogue"]  
    save_meta = {}

    used_spk = []

    ##### interferance
    save_meta["interference"] = []
    interference = metadata["interference"]
    inter_audio = utils.read_audio_file_torch(os.path.join(curr_dir, 'intereference.wav'), 1).numpy()
    num_interfer = min([len(interference), 2])
    for i in range(0, num_interfer):
        spko = interference[i]
        timestamps = interference[i]["timestamp"]
        if np.random.rand() < replace_prob:
            spk_pick, example, output, merged_times = dataset.replace_sample(timestamps, 0, existing_speaker = used_spk, interfer = True)
            spko["spk_id"] = spk_pick
            used_spk.append(spk_pick)
            spko["timestamp"] = merged_times
        else:
            output = np.zeros_like(inter_audio)
            for b,e in timestamps:
                output[:, b:e] = inter_audio[:, b:e]
            used_spk.append(spko["spk_id"])

        utils.write_audio_file(os.path.join(save_dir, f"intereference{i}.wav"), output, sr)
        save_meta["interference"].append(spko)


    save_meta["target_dialogue"] = []
    ### self speaker
    self_spk = target[0]
    timestamps = self_spk["timestamp"]
    if np.random.rand() < replace_prob:
        spk_pick, example, output, merged_times = dataset.replace_sample(timestamps, 1, existing_speaker = used_spk)
        self_spk["spk_id"] = spk_pick
        assert(spk_pick not in used_spk)
        used_spk.append(spk_pick)
        self_spk["timestamp"] = merged_times
    
    else:
        print("cannot be here")
        output = utils.read_audio_file_torch(os.path.join(curr_dir, 'self_speech.wav'), 1).numpy()
        example = utils.read_audio_file_torch(os.path.join(curr_dir, 'example.wav'), 1).numpy()
        assert(self_spk["spk_id"] not in used_spk)
        used_spk.append(self_spk["spk_id"])
    check_zeros(output)
    check_zeros(example)

    utils.write_audio_file(os.path.join(save_dir, f"self_speech.wav"), output, sr)
    utils.write_audio_file(os.path.join(save_dir, f"example.wav"), example, sr)
    save_meta["target_dialogue"].append(self_spk)

    ### OTHER SPEAKERS IN CONVERSATION
    for i in range(1, len(target)):
        spko = target[i]
        timestamps = target[i]["timestamp"]
        if np.random.rand() < replace_prob:
            spk_pick, example, output, merged_times = dataset.replace_sample(timestamps, 0, existing_speaker = used_spk)
            spko["spk_id"] = spk_pick
            spko["timestamp"] = merged_times
            assert(spk_pick not in used_spk)
            used_spk.append(spk_pick)
        else:
            print("cannot be here")
            output = utils.read_audio_file_torch(os.path.join(curr_dir, f'target_speech{i-1}.wav'), 1).numpy()
            assert(spko["spk_id"] not in used_spk)
            used_spk.append(spko["spk_id"])
        # check_zeros(output)
        utils.write_audio_file(os.path.join(save_dir, f"target_speech{i-1}.wav"), output, sr)
        save_meta["target_dialogue"].append(spko)

    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, 'w', encoding="utf-8") as f:
        json.dump(save_meta, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/MixLibriSpeech/librispeech_scaper_fmt')
    parser.add_argument('--save_dir', type=str, default='data/MixLibriSpeech/librispeech_scaper_fmt')
    parser.add_argument('--replace_prob', type=float, default=0.5)


    args = parser.parse_args()

    split = ["train", "val"]
    spk_id = {}

    np.random.seed(0)
    random.seed(0)
    
    for s in split:
        dataset = LibriTTS_Dataset(LibriTTS_config, s)
        in_dir = os.path.join(args.data_dir, s)
        out_dir = os.path.join(args.save_dir, s)
        dirs = sorted(list(Path(in_dir).glob('[0-9]*')))


        pbar = tqdm(total=len(dirs))
        pool = mp.Pool(32)
        callback_fn = lambda _: pbar.update()

        for dir in dirs:
            sample_id = str(dir).split('/')[-1]
            # convert_sample(in_dir, out_dir, sample_id, dataset, 16000, args.replace_prob)
            pool.apply_async(convert_sample,
                args=(in_dir, out_dir, sample_id, dataset, 16000, args.replace_prob),
                callback=callback_fn,
                error_callback=handle_error)
            
        pool.close()
        pool.join()
        pbar.close()