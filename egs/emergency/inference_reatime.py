# -*- coding: utf-8 -*-
# @Author   : jeffcheng
# @Time     : 2021/9/1 - 15:13
# @Reference: a inference script for single audio, heavily base on demo.py and traintest.py
import os
import sys
import csv
import argparse

import numpy as np
import torch
import torchaudio

import warnings
warnings.filterwarnings("ignore")

import pyaudio
import torch.nn.functional as F
import time

from datetime import datetime
now = datetime.now()

CHUNK = 960
RATE = 16000


p=pyaudio.PyAudio()
stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK)

torchaudio.set_audio_backend("soundfile")       # switch backend
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from src.models import ASTModel

# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'


def make_features(waveform, mel_bins, target_length=1024):
    sr = 16000

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-2.2162132)) / (4.137818 * 2)
    return fbank


def load_label(label_csv):
    with open(label_csv, 'r', encoding="UTF-8") as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels


if __name__ == '__main__':
    # assume each input spectrogram has 100 time frames
    input_tdim = 1024

    record_sec = 3
    label_dim = 9
    
    checkpoint_paths = './ckpt/ckpt_{}/audio_model_50_mixup0_noise.pth'.format(str(label_dim))

    label_csv = './data/Emergency_class_labels_indices_{}.csv'.format(str(label_dim))       # label and indices for audioset data
    # 1. load the best model and the weights
    ast_mdl = ASTModel(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
    
    print(f'[*INFO] load checkpoint: {checkpoint_paths}')
    
    checkpoint = torch.load(checkpoint_paths, map_location=torch.device('cpu'))
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)

    # GPU가 가능한 환경이면 활용
    #audio_model = audio_model.to(torch.device("cuda:0"))
    audio_model.eval()     
    
    labels = load_label(label_csv)
    print('[*INFO] predice results:')
    with torch.no_grad():
        if not os.path.exists('real_time_result/'):
            os.mkdir('real_time_result/')
        f = open('real_time_result/' + str(now.month) + str(now.day) + '-' + str(now.hour) + str(now.minute) + '.csv', 'w', encoding="UTF-8-sig")
        wr = csv.writer(f)
        wr.writerow(['Real time Audio Classificaiton'])
        wr.writerow(['filename', '1', '1', '2', '2', '3', '3'])


        t = 1
        while(True):
            print("\n{} Step\n#########################################################".format(str(t)))
            print("Recording {} sec".format(str(record_sec)), end='')
            for i in range(int(RATE / CHUNK * record_sec)):
                piece = np.fromstring(stream.read(CHUNK),dtype=np.int16)
                if i == 0:
                    waveform = piece
                else:
                    waveform = np.concatenate((waveform, piece), axis=0)

                if i % 16 == 0 and i != 0:
                    print('.', end='')

            waveform = torch.Tensor(waveform).unsqueeze(0)
            print('\nWaveform Length:', len(waveform[0]))

            # feature를 FFT 수행
            feats = make_features(waveform, mel_bins=128)           
            feats_data = feats.expand(1, input_tdim, 128)          

            print('\n! Inferencing')
            output = audio_model.forward(feats_data)
            #output = torch.sigmoid(output)
            output = F.softmax(output)
            result_output = output.data.cpu().numpy()[0]

            sorted_indexes = np.argsort(result_output)[::-1]

            row_result = []
            row_result.append(str(now.month) + str(now.day) + str(now.hour) + str(now.minute))
            for k in range(9):
                """if k == 0:
                    if result_output[sorted_indexes[k]] > 0.80:
                        print('{}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]],
                                                result_output[sorted_indexes[k]]))
                    else:
                        print("Silence")"""
                print('{}: {:.2f}%'.format(np.array(labels)[sorted_indexes[k]],
                                        result_output[sorted_indexes[k]] * 100))
                row_result.append(np.array(labels)[sorted_indexes[k]])
                row_result.append(result_output[sorted_indexes[k]])
            wr.writerow(row_result)
            print("#########################################################")
            t += 1
            time.sleep(1)
        f.close()