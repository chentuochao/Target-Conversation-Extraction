import os
import sys
import csv
import argparse
import json
from typing import List

import numpy as np
import librosa
import random
import glob
from datasets.WhitePinkBrownPerturbation import get_colored_noise
import noisereduce as nr
from datasets.dataset_util import augment_reverbration, overlap_duration_check

import src.utils as utils
from scipy.signal import butter, lfilter

def filter_set_by_keys(original_set, key_sublist):
    return {key: original_set[key] for key in key_sublist if key in original_set}

def denoise(signal, noise_sample, sr, stationary=False, n_jobs=1):
    return nr.reduce_noise(y=signal, sr=sr, y_noise=noise_sample, stationary=stationary, n_jobs=n_jobs)

def high_pass_filter(wav, sampling_rate, order=7):
    CUTOFF = 20
    nyquist = 0.5 * sampling_rate
    normal_cutoff = CUTOFF / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    wav = lfilter(b, a, wav)
    return wav


class ASR_Dataset(object):
    def __init__(self, config, split, args):
        self.sr = args.sr
        self.split = split
        self.wav_folder = config["wav_folder"]
        self.txt_folder = config["txt_folder"]
        self.split_tsv = config[split + "_tsv"]
        self.total_samples = args.total_samples
        self.diag_lists = []
        with open(self.split_tsv, 'r', newline='', encoding='utf-8') as tsvfile:
            # Assuming the first line contains headers, adjust if necessary
            reader = csv.reader(tsvfile, delimiter='\t')
            init = False
            for row in reader:
                if init == False:
                    init =True
                    continue
                unit = row[0].split('.wav')[0]
                self.diag_lists.append(unit)

    def load_txt(self, file_path):
        data_list = []
        speakers = []
        with open(file_path, 'r', encoding='utf-8') as txtfile:
            for line in txtfile:
                # Split the line by tab to extract values
                values = line.strip().split('\t')

                # Extract the desired values and convert to the appropriate types
                time_range = eval(values[0])  # Convert string representation of list to actual list
                spk_id = values[1]
                content = values[3]
                if spk_id != "G00000000" and spk_id not in speakers:
                    speakers.append(spk_id)
                # Append the extracted values to the data_list
                data_list.append([time_range, spk_id, content])
        return data_list, speakers

    def generate_dialog(self, diag_info, audio):

        num_turns = len(diag_info)
        MAX_LEN = 80*self.sr + self.total_samples

        if (num_turns < 2):
            print("not enough turn number !!!! ")
            return None, None ### fail and regenerate

        ###
        L = int(self.sr*diag_info[-1][0][1])
        if L <= MAX_LEN:
            print("not long enough !!!! ")
            return None, None ### fail and regenerate

        diag_wavs = {}
        diag_index = {}
        diag_text = {}
        speakers = []

        for trial in range(0, 25):
            diag_wavs = {}
            diag_index = {}
            diag_text = {}
            speakers = []

            start_t = np.random.randint(low = 0, high = L - self.total_samples)
            end_t = start_t + self.total_samples
            ### extract the dialogue
            for utter_id in range(0, num_turns):
                timestamp = diag_info[utter_id][0]
                start,end = int(self.sr*timestamp[0]), int(self.sr*timestamp[1])

                if end < start_t:
                    continue
                if start > end_t:
                    break
                # print(trial, utter_id, start, end, start_t, end_t)
                spk_id = diag_info[utter_id][1]
                text = diag_info[utter_id][2]

                if spk_id == "G00000000": ### G00000000 means invalid speech
                    continue

                s_i = max([start_t, start])
                e_i = min([end_t, end])

                if spk_id not in speakers:
                    speakers.append(spk_id)
                    diag_wavs[spk_id] = np.zeros((self.total_samples, ))
                    diag_index[spk_id] = []
                    diag_text[spk_id] = []
                diag_wavs[spk_id][s_i - start_t: e_i - start_t] = audio[s_i:e_i]
                diag_index[spk_id].append([s_i - start_t, e_i - start_t])
                diag_text[spk_id].append(text)
            if (len(speakers) <= 1):
                continue
            else:
                break

        if (len(speakers) <= 1):
            print("not enough speaker number !!!! ")
            return None, None ### fail and regenerate

        self_spk = speakers[0]
        other_spk = speakers[1]
        clean_example = []
        len_example = 0
        ### extract example
        for utter_id in range(0, num_turns):
            timestamp = diag_info[utter_id][0]
            start,end = int(self.sr*timestamp[0]), int(self.sr*timestamp[1])
            text = diag_info[utter_id][2]
            spk_id = diag_info[utter_id][1]
            if end < start_t or start > end_t:
                if spk_id == self_spk and '[+]' not in text:
                    len_example += (end - start)
                    clean_example.append(audio[start:end])
            if len_example > 60*self.sr:
                break
        if (len_example <= 20*self.sr):
            print("not enough clean exmaple !!!! ")
            return None, None ### fail and regenerate

        clean_example = np.concatenate(clean_example, axis = 0)
        clean_example = clean_example/np.abs(clean_example).max()
        self_data = {}
        self_data["clean_example"] = clean_example[np.newaxis, :]
        self_wav = diag_wavs[self_spk]
        self_wav = self_wav/np.abs(self_wav).max()
        self_data["speech"] = self_wav[np.newaxis, :]
        self_data["spk_id"] = self_spk
        self_data["timestamp"] = diag_index[self_spk]
        self_data["text"] = diag_text[self_spk]

        target_data = {}
        other_wav = diag_wavs[other_spk]
        other_wav = other_wav/np.abs(other_wav).max()
        target_data["speech"] = other_wav[np.newaxis, :]
        target_data["spk_id"] = other_spk
        target_data["timestamp"] = diag_index[other_spk]
        target_data["text"] = diag_text[other_spk]

        return self_data, target_data

    def get_conversation(self, n_diag):
        diag_choices = np.random.choice(self.diag_lists, n_diag + 5, replace=False)
        voices_data = []
        used_speakers = []

        for diag in diag_choices:
            txt_file = os.path.join(self.txt_folder, diag + ".txt")
            diag_info, speakers = self.load_txt(txt_file)

            overlap = False
            for spk in speakers:
                if spk in used_speakers:
                    overlap = True 
                    break
            if overlap:
                print("overlap speaker and remove it!!!!", spk, used_speakers)
                continue 

            wav_file = os.path.join(self.wav_folder, diag + ".wav")
            audio, _ = librosa.load(wav_file, sr=self.sr, mono = True)
            audio = high_pass_filter(audio, self.sr)

            self_data, target_data = self.generate_dialog(diag_info, audio)
            if self_data is not None:
                # print("audio finish: ", audio.shape)
                voices_data.append(
                    {
                        "self_speech": self_data,
                        "others": [target_data]
                    }
                )
                for spk in speakers:
                    used_speakers.append(spk)


            if len(voices_data) == n_diag:
                return voices_data
        return voices_data



class AMI_Dataset(object): ## todo
    def __init__(self, config, split, args):
        if split == 'train':
            self.train_folder = config['train_folder']
            # self.train_folder = '/gscratch/intelligentsystems/common_datasets/ami/ami-dataset/1min/train/'
        elif split == 'val':
            self.train_folder = config['val_folder']
            # self.train_folder = '/gscratch/intelligentsystems/common_datasets/ami/ami-dataset/1min/val/'
        elif split == 'test':
            self.train_folder = config['test_folder']
            # self.train_folder = '/gscratch/intelligentsystems/common_datasets/ami/ami-dataset/1min/test/'

        self.clean_folder = config['clean_folder']
        # self.clean_folder = '/gscratch/intelligentsystems/qirui/ami-datasets_clean_example/'

        self.sr = args.sr
        self.total_samples = args.total_samples
        # self.sr = 16000

    def select_random_conv_sub_id(self, conv_ids : list, chosen_conv_super_ids : id):
        conv_ids = [conv_id for conv_id in conv_ids if conv_id[:-1] not in chosen_conv_super_ids]
        # if no conv_id remains, return None
        if len(conv_ids) == 0:
            return None

        # randomly choose a conv id
        chosen_id = random.choice(conv_ids)
        conv_path = os.path.join(self.train_folder, chosen_id)

        conv_sub_ids = [item for item in os.listdir(conv_path) if os.path.isdir(os.path.join(conv_path, item))]
        chosen_sub_id = random.choice(conv_sub_ids)

        return chosen_sub_id

    def choose_self_speech_spk_id(self, full_conv_path : str):
        significant_speakers = []
        with open(os.path.join(full_conv_path, 'meta_data.json'), 'r') as file:
            significant_speakers = json.load(file)['meta_data']['significant_speakers']
        # chosen_self_speech_spk_id = significant_speakers[np.random.choice(np.arange(len(significant_speakers)))]
        chosen_self_speech_spk_id = random.choice(significant_speakers)
        return chosen_self_speech_spk_id

    def choose_clean_speech(self, spk_id : str, conv_id : str):
        conv_super_id = conv_id[:-1] # ES2002b -> ES2002
        other_conv_ids = [item for item in os.listdir(self.clean_folder) if item.startswith(conv_super_id) and item != conv_id]

        # randomly select from conv_ids different from the given conv_id for clean speech
        conv_id_for_clean_speech = random.choice(other_conv_ids)

        target_wav_name = conv_id_for_clean_speech + '_' + spk_id + '.wav'
        audio_path = os.path.join(self.clean_folder, conv_id_for_clean_speech, target_wav_name)

        # no clean sample found
        if not os.path.exists(audio_path):
            print("cannnot find clean example", audio_path)
            return None
        try:
            clean_speech, _ = librosa.load(audio_path, sr=self.sr)
            clean_speech = high_pass_filter(clean_speech, self.sr)
        except:
            print("wrong infifnite exmaple ", audio_path)
            return None
        # clean speech shorter than 60 seconds
    
        if clean_speech.shape[0] < 30 * self.sr:
            print("clean example is not long", audio_path)
            return None

        max_len = 60
        if clean_speech.shape[0] > max_len * self.sr:
            # clip 60 seconds from the audio
            clip_from = random.choice(range(clean_speech.shape[0] - max_len * self.sr))
            clean_speech = clean_speech[clip_from : clip_from+max_len*self.sr]

        return clean_speech


    def convert_to_dict(self, speech, speech_data, spk_id, clean_audio=None, conv_id=None):
        result = {}
        if speech.shape[-1] < self.total_samples/2:
            raise ValueError("load wav not very short!!!!")
        elif speech.shape[-1] < self.total_samples:
            speech = np.pad(speech, (0, self.total_samples - speech.shape[-1]), mode='constant' )

        result['speech'] = speech[np.newaxis, :self.total_samples]
        result['timestamp'] = [[int(word['start'] * self.sr), int(word['end'] * self.sr)] for word in speech_data['words']]
        text_list = [word['text'] for word in speech_data['words']]
        result['text'] = ' '.join(text_list)
        result['spk_id'] = spk_id
        if clean_audio is not None:
            result['clean_example'] = clean_audio[np.newaxis, :]
        if conv_id is not None:
            result['conversation_id'] = conv_id
        return result


    def zero_out_mask(self, timestamps : list, len_waveform, sr):
        time = np.arange(0, len_waveform)

        zero_out_mask = np.zeros_like(time)

        for start_time, end_time in timestamps:
            zero_out_mask[(time >= start_time*sr) & (time < end_time*sr)] = 1

        return zero_out_mask


    def get_conversation(self, n_diag):
        '''
            n_diag -- number of diag
            return List[Dict()] - diag_list
            Dict(): - per conversation
            (1) clean example from other audio
                (select random one but it should > 3,4s ???????) as self speech
            (3) list n keys (n is speaker number)
                key - value:
                    speech (30 s)
                    timing
                    text
                    speaker id
            {
                "self speech":

                    {
                        "speech":
                        "timing":
                        "text":
                        "speaker_id":

                        "clean example":
                        "conversation_id":

                    } // one speaker

                "others":
                [
                    {
                        "speech":
                        "timing":
                        "text":
                        "speaker_id":
                    } // one speaker

                ]
            }
        '''
        conv_ids = [item for item in os.listdir(self.train_folder) if os.path.isdir(os.path.join(self.train_folder, item))]
        chosen_conv_super_ids = set()
        chosen_speaker_ids = set()
        output_list = []
        while len(output_list) < n_diag:
            dialogue = dict()
            # select dialogue/conversation
            conv_sub_id = self.select_random_conv_sub_id(conv_ids, chosen_conv_super_ids)
            # no conversation left
            if conv_sub_id is None:
                print('reached maximum sample number')
                break
            conv_id = conv_sub_id[:conv_sub_id.find('_')]
            conv_super_id = conv_id[:-1] # super_id: ES2010, id: ES2010b, sub_id: ES2010b_0
            full_conv_path = os.path.join(self.train_folder, conv_id, conv_sub_id)

            #############  Create self speech ################
            # select self speech speaker
            self_spk_id = self.choose_self_speech_spk_id(full_conv_path)
            # check if the self speech speaker is repetitive
            if self_spk_id in chosen_speaker_ids:
                continue

            # select clean speech
            clean_audio = self.choose_clean_speech(self_spk_id, conv_id)

            # resample if no clean sample found
            if clean_audio is None:
                continue

            # normalize and trim audio
            clean_audio /= np.abs(clean_audio).max()
            L_clean = max([clean_audio.shape[-1], self.sr*120])
            clean_audio = clean_audio[:L_clean]

            # get self speech audio
            with open(os.path.join(full_conv_path, 'meta_data.json'), 'r') as file:
                meta_data = json.load(file)['meta_data']

            # get information from meta data
            audio_dir_path = meta_data['audio_dir_path']
            start_idx = int(meta_data['start'] * self.sr)
            end_idx = int(meta_data['end'] * self.sr)

            # load, clip and normalize audio
            self_speech_audio, sr = librosa.load(os.path.join(audio_dir_path, conv_id + '_' + self_spk_id + '.wav'), sr=self.sr)
            self_speech_audio = self_speech_audio[start_idx : end_idx]
            self_speech_audio = high_pass_filter(self_speech_audio, self.sr)
            self_speech_audio /= np.abs(self_speech_audio).max()

            # apply zero mask
            with open(os.path.join(full_conv_path, 'speaker_' + self_spk_id + '.json')) as file:
                words_data = json.load(file)['words']
            timestamps = []
            for word_data in words_data:
                timestamps.append((float(word_data['start']), float(word_data['end'])))
            zero_mask = self.zero_out_mask(timestamps, self_speech_audio.shape[0], self.sr)
            self_speech_audio = self_speech_audio * zero_mask

            # get self speech data
            with open(os.path.join(full_conv_path, 'speaker_' + self_spk_id + '.json'), 'r') as file:
                self_speech_data = json.load(file)

            # get self speech dictionary
            self_speech_dict = self.convert_to_dict(self_speech_audio, self_speech_data, self_spk_id, clean_audio, conv_sub_id)
            dialogue['self_speech'] = self_speech_dict

            #############  Create others #####################
            others = []
            spk_ids = []

            # get all other speaker ids
            for item in os.listdir(full_conv_path):
                if item.startswith('speaker_') and item.endswith('.json'):
                    spk_id = item[item.find('_')+1 : item.find('.')]
                    if spk_id == self_spk_id:
                        continue
                    spk_ids.append(spk_id)

            # check if the other speakers are repetitive
            repetitives = sum([speaker_id in chosen_speaker_ids for speaker_id in spk_ids])
            if repetitives > 0:
                continue

            # get others speech dictionary
            for spk_id in spk_ids:
                # load, clip and normalize audio
                try:
                    # print(os.path.join(full_conv_path, 'speaker_' + spk_id + '.wav'))
                    speech_audio, sr = librosa.load(os.path.join(audio_dir_path, conv_id + '_' + spk_id + '.wav'), sr=self.sr)
                    speech_audio = speech_audio[start_idx : end_idx]
                    speech_audio = high_pass_filter(speech_audio, self.sr)
                    speech_audio /= np.abs(speech_audio).max()
                except:
                    speech_audio = np.zeros_like(speech_audio)

                # apply zero mask
                with open(os.path.join(full_conv_path, 'speaker_' + spk_id + '.json')) as file:
                    words_data = json.load(file)['words']
                timestamps = []
                for word_data in words_data:
                    timestamps.append((float(word_data['start']), float(word_data['end'])))
                zero_mask = self.zero_out_mask(timestamps, speech_audio.shape[0], self.sr)
                speech_audio = speech_audio * zero_mask

                # get speech data
                with open(os.path.join(full_conv_path, 'speaker_' + spk_id + '.json'), 'r') as file:
                    speech_data = json.load(file)

                speech_dict = self.convert_to_dict(speech_audio, speech_data, spk_id)
                others.append(speech_dict)
            dialogue['others'] = others

            # append to output
            output_list.append(dialogue)
            chosen_conv_super_ids.add(conv_super_id)
            chosen_speaker_ids.add(self_spk_id)
            for spk_id in spk_ids:
                chosen_speaker_ids.add(spk_id)

        return output_list

    # def get_conversation_givelen(self, n_diag, total_sample):
    #     '''
    #         total_sample: 20 * sr = 20 * 16000
    #     '''
    #     conversations = self.get_conversation(n_diag)
    #     # traverse through every conversation to extract 20 seconds
    #     for conv in conversations:
    #         timestamp = conv['self_speech']['timestamp']
    #         chosen_start_time = -1
    #         # traverse through every word and check if the 20 second clip is valid
    #         for i in range(len(timestamp)):
    #             starttime = timestamp[i][0]
    #             endtime = starttime + total_sample
    #             # append length of self speaking time within the 20 seconds
    #             total_self_speak_time = 0
    #             for j in range(i, len(timestamp)):
    #                 total_self_speak_time += timestamp[j][1] - timestamp[j][0]
    #                 if timestamp[j][1] >= endtime:
    #                     break
    #             # print(i, starttime, total_self_speak_time, 3 * self.sr)
    #             if total_self_speak_time >= 3 * self.sr:
    #                 # choose the 20 second clip: i to j-1
    #                 chosen_start_time = starttime
    #                 break
    #         # record the chosen start time of this conversation clip
    #         assert(chosen_start_time >= 0)
    #         if chosen_start_time + total_sample <= conv['self_speech']['speech'].shape[-1]:
    #             startime = chosen_start_time
    #             endtime = startime + total_sample
    #         else:
    #             startime = conv['self_speech']['speech'].shape[-1] - total_sample
    #             endtime = conv['self_speech']['speech'].shape[-1]

    #         assert(endtime <= conv['self_speech']['speech'].shape[-1])
    #         chosen_idxs = [i for i in range(len(conv['self_speech']['timestamp'])) if conv['self_speech']['timestamp'][i][0] >= startime and conv['self_speech']['timestamp'][i][0] < endtime]
    #         conv['self_speech']['speech'] = conv['self_speech']['speech'][:, starttime : endtime]
    #         conv['self_speech']['timestamp'] = [  [conv['self_speech']['timestamp'][i][0] - startime, conv['self_speech']['timestamp'][i][1] - startime] for i in chosen_idxs]
    #         conv['self_speech']['text'] = [conv['self_speech']['text'][i] for i in chosen_idxs]

    #         for other_idx in range(len(conv['others'])):
    #             chosen_idxs = [i for i in range(len(conv['others'][other_idx]['timestamp'])) if conv['others'][other_idx]['timestamp'][i][0] >= startime and conv['others'][other_idx]['timestamp'][i][0] < endtime]
    #             conv['others'][other_idx]['speech'] = conv['others'][other_idx]['speech'][:, starttime : endtime]
    #             conv['others'][other_idx]['timestamp'] = [ [conv['others'][other_idx]['timestamp'][i][0] - starttime, conv['others'][other_idx]['timestamp'][i][1] - starttime] for i in chosen_idxs]
    #             conv['others'][other_idx]['text'] = [conv['others'][other_idx]['text'][i] for i in chosen_idxs]

    #     return conversations

