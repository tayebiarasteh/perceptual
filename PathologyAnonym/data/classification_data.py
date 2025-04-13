"""
classification_data.py
Created on Feb 5, 2024.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""

import glob
import os
import pdb
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import webrtcvad
import struct
from tqdm import tqdm
import random
from math import ceil, isnan
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
import math
import torch
from torch.utils.data import Dataset
from scipy.signal import get_window
from librosa.filters import mel
from scipy import signal

from config.serde import read_config

import warnings
warnings.filterwarnings('ignore')


# Global variables
int16_max = (2 ** 15) - 1
epsilon = 1e-15




class classification_tisvcontentbased_data_preprocess():
    def __init__(self, cfg_path="/PATH/PathologyAnonym/config/config.yaml"):
        self.params = read_config(cfg_path)



    def get_mel_content(self, input_df, output_df_path, exp_name):
        """
        References:
            https://github.com/RF5/simple-autovc/
            https://github.com/CODEJIN/AutoVC/

        Parameters
        ----------
        """

        mel_basis_hifi = mel(self.params['preprocessing']['sr'], 1024, fmin=0, fmax=8000, n_mels=80).T
        b, a = self.butter_highpass(30, self.params['preprocessing']['sr'], order=5)

        # lower bound of utterance length
        utter_min_len = (self.params['preprocessing']['tisv_frame'] * self.params['preprocessing']['hop'] +
                         self.params['preprocessing']['window']) * self.params['preprocessing']['sr']

        final_dataframe = pd.DataFrame([])

        for index, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):

            # Read audio file
            utter_path = os.path.join(self.params['file_path'], row['relative_path'])
            x, fs = sf.read(utter_path)

            if x.shape[0] < utter_min_len:
                continue

            x = librosa.resample(x, fs, self.params['preprocessing']['sr'])
            # Remove drifting noise
            y = signal.filtfilt(b, a, x)
            # Ddd a little random noise for model roubstness
            wav = y * 0.96 + (np.random.RandomState().rand(y.shape[0]) - 0.5) * 1e-06
            # Compute spect
            D = self.pySTFT(wav).T
            # Convert to mel and normalize
            D_mel = np.dot(D, mel_basis_hifi)
            S = np.log(np.clip(D_mel, 1e-5, float('inf'))).astype(np.float32)

            os.makedirs(os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name,
                                     os.path.dirname(row['relative_path'])), exist_ok=True)

            rel_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name,
                                    os.path.dirname(row['relative_path']),
                                    os.path.basename(row['relative_path']).replace('.wav', '.npy'))
            S = S.transpose(1,0)
            np.save(rel_path, S)

            # add to the new dataframe
            tempp = pd.DataFrame([[os.path.join('tisv_preprocess', exp_name,
                                                os.path.dirname(row['relative_path']),
                                                os.path.basename(row['relative_path']).replace('.wav', '.npy')),
                                   row['speaker_id'], row['subset'], row['gender'], row['location'], row['age_y'],
                                   row['microphone'], row['patient_control'], row['automatic_WRR'],
                                   x.shape[0] / self.params['preprocessing']['sr'],
                                   row['user_id'], row['session'], row['father_tongue'], row['mother_tongue'],
                                   row['test_type'], row['mic_room'], row['diagnosis']
                                   ]],
                                 columns=['relative_path', 'speaker_id', 'subset', 'gender', 'location', 'age_y',
                                          'microphone',
                                          'patient_control', 'automatic_WRR', 'file_length', 'user_id', 'session',
                                          'father_tongue',
                                          'mother_tongue', 'test_type', 'mic_room', 'diagnosis'])
            final_dataframe = final_dataframe.append(tempp)

        # to check the criterion of 8 utters after VAD
        final_dataframe = self.csv_speaker_trimmer(final_dataframe)
        # sort based on speaker id
        final_data = final_dataframe.sort_values(['relative_path'])
        final_data.to_csv(output_df_path, sep=';', index=False)


    def get_mel_content_anonym(self, input_df, output_df_path, exp_name, old_exp_name, anonym_method='PEAKS_anonymized'):
        """
        References:
            https://github.com/RF5/simple-autovc/
            https://github.com/CODEJIN/AutoVC/

        Parameters
        ----------
        """

        mel_basis_hifi = mel(self.params['preprocessing']['sr'], 1024, fmin=0, fmax=8000, n_mels=80).T
        b, a = self.butter_highpass(30, self.params['preprocessing']['sr'], order=5)

        # lower bound of utterance length
        utter_min_len = (self.params['preprocessing']['tisv_frame'] * self.params['preprocessing']['hop'] +
                         self.params['preprocessing']['window']) * self.params['preprocessing']['sr']

        final_dataframe = pd.DataFrame([])

        for index, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):

            row_relative_path = row['relative_path'].replace('.npy', '.wav')
            row_relative_path = row_relative_path.replace('tisv_preprocess/dysarthria_70_30_contentmel/PEAKS', 'PEAKS')
            row_relative_path = row_relative_path.replace('tisv_preprocess/dysglossia_70_30_contentmel/PEAKS', 'PEAKS')
            row_relative_path = row_relative_path.replace('tisv_preprocess/dysphonia_70_30_contentmel/PEAKS', 'PEAKS')
            row_relative_path = row_relative_path.replace('tisv_preprocess/CLP_70_30_contentmel/PEAKS', 'PEAKS')
            row_relative_path = row_relative_path.replace('tisv_preprocess/multiclass_62_16/PEAKS', 'PEAKS')
            row_relative_path = row_relative_path.replace('PEAKS/', anonym_method + '/')

            # Read audio file
            utter_path = os.path.join(self.params['file_path'], row_relative_path)
            x, fs = sf.read(utter_path)

            if x.shape[0] < utter_min_len:
                continue

            x = librosa.resample(x, fs, self.params['preprocessing']['sr'])
            # Remove drifting noise
            y = signal.filtfilt(b, a, x)
            # Ddd a little random noise for model roubstness
            wav = y * 0.96 + (np.random.RandomState().rand(y.shape[0]) - 0.5) * 1e-06
            # Compute spect
            D = self.pySTFT(wav).T
            # Convert to mel and normalize
            D_mel = np.dot(D, mel_basis_hifi)
            S = np.log(np.clip(D_mel, 1e-5, float('inf'))).astype(np.float32)

            os.makedirs(os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name,
                                     os.path.dirname(row_relative_path)), exist_ok=True)

            rel_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name,
                                    os.path.dirname(row_relative_path),
                                    os.path.basename(row_relative_path).replace('.wav', '.npy'))
            S = S.transpose(1,0)
            np.save(rel_path, S)

            # add to the new dataframe
            tempp = pd.DataFrame([[os.path.join('tisv_preprocess', exp_name,
                                                os.path.dirname(row_relative_path),
                                                os.path.basename(row_relative_path).replace('.wav', '.npy')),
                                   row['speaker_id'], row['subset'], row['gender'], row['location'], row['age_y'],
                                   row['microphone'], row['patient_control'], row['automatic_WRR'],
                                   x.shape[0] / self.params['preprocessing']['sr'],
                                   row['user_id'], row['session'], row['father_tongue'], row['mother_tongue'],
                                   row['test_type'], row['mic_room'], row['diagnosis']
                                   ]],
                                 columns=['relative_path', 'speaker_id', 'subset', 'gender', 'location', 'age_y',
                                          'microphone',
                                          'patient_control', 'automatic_WRR', 'file_length', 'user_id', 'session',
                                          'father_tongue',
                                          'mother_tongue', 'test_type', 'mic_room', 'diagnosis'])
            final_dataframe = final_dataframe.append(tempp)

        # to check the criterion of 8 utters after VAD
        final_dataframe = self.csv_speaker_trimmer(final_dataframe)
        # sort based on speaker id
        final_data = final_dataframe.sort_values(['relative_path'])
        final_data.to_csv(output_df_path, sep=';', index=False)





    def pad_seq(self, x, base=32):
        len_out = int(base * math.ceil(float(x.shape[0]) / base))
        len_pad = len_out - x.shape[0]
        assert len_pad >= 0
        return torch.nn.functional.pad(x, (0, 0, 0, len_pad), value=0), len_pad

    def butter_highpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def pySTFT(self, x, fft_length=1024, hop_length=256):
        x = np.pad(x, int(fft_length // 2), mode='reflect')
        noverlap = fft_length - hop_length
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
        strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        fft_window = get_window('hann', fft_length, fftbins=True)
        result = np.fft.rfft(fft_window * result, n=fft_length).T
        return np.abs(result)






class Dataloader_disorder(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', experiment_name='name'):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.file_base_dir = self.params['file_path']
        self.sampling_val = 180

        if mode == 'train':
            self.main_df = pd.read_csv(os.path.join(self.file_base_dir, "tisv_preprocess", experiment_name, "train_" + experiment_name + ".csv"), sep=';')
        elif mode == 'valid':
            self.main_df = pd.read_csv(os.path.join(self.file_base_dir, "tisv_preprocess", experiment_name, "test_" + experiment_name + ".csv"), sep=';')
        elif mode == 'test':
            self.main_df = pd.read_csv(os.path.join(self.file_base_dir, "tisv_preprocess", experiment_name, "test_" + experiment_name + ".csv"), sep=';')

        self.main_df = self.main_df[self.main_df['file_length'] > 3]

        self.speaker_list = self.main_df['speaker_id'].unique().tolist()




    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.speaker_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        output_tensor = []

        # select a speaker
        selected_speaker = self.speaker_list[idx]
        selected_speaker_df = self.main_df[self.main_df['speaker_id'] == selected_speaker]

        # randomly select M utterances from the speaker
        shuff_selected_speaker_df = selected_speaker_df.sample(frac=1).reset_index(drop=True)

        shuff_selected_speaker_df = shuff_selected_speaker_df[:self.params['Network']['M']]

        # return M utterances
        for index, row in shuff_selected_speaker_df.iterrows():
            # select a random utterance
            utterance = np.load(os.path.join(self.file_base_dir, row['relative_path']))

            # randomly sample a fixed specified length
            id = np.random.randint(0, utterance.shape[1] - self.sampling_val, 1)
            utterance = utterance[:, id[0]:id[0] + self.sampling_val]

            output_tensor.append(utterance)

        while len(output_tensor) < self.params['Network']['M']:
            output_tensor.append(utterance)

        output_tensor = np.stack((output_tensor, output_tensor, output_tensor), axis=1) # (n=M, c=3, h=melsize, w=sampling_val)
        output_tensor = torch.from_numpy(output_tensor) # (M, c, h) treated as (n, h, w)


        # one hot
        if shuff_selected_speaker_df['patient_control'].values[0] == 'patient':
            label = torch.ones((self.params['Network']['M']), 2)
            label[:, 0] = 0
        elif shuff_selected_speaker_df['patient_control'].values[0] == 'control':
            label = torch.zeros((self.params['Network']['M']), 2)
            label[:, 0] = 1
        elif shuff_selected_speaker_df['mic_room'].values[0] == 'control_group_plantronics':
            label = torch.zeros((self.params['Network']['M']), 2)
            label[:, 0] = 1
        elif shuff_selected_speaker_df['mic_room'].values[0] == 'maxillofacial':
            label = torch.zeros((self.params['Network']['M']), 2)
            label[:, 0] = 0

        label = label.float()

        return output_tensor, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        full_length = len(self.main_df)

        disease_length = sum(self.main_df['patient_control'].values == 'patient')
        output_tensor = (full_length - disease_length) / (disease_length + epsilon)

        output_tensor = torch.Tensor([output_tensor])
        return output_tensor
