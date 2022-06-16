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


torchaudio.set_audio_backend("soundfile")       # switch backend
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from src.models import ASTModel

# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'


def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)
    print(len(waveform))
    print(len(waveform[1]))
    print(type(waveform))
    
    #waveform = torch.Tensor(nr.reduce_noise(y=waveform, sr=sr))

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

    # !!! normalize 
    #fbank = (fbank - (-2.2162132)) / (4.137818 * 2)
    return fbank


def load_label(label_csv):
    with open(label_csv, 'r', encoding='') as f:
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

    parser = argparse.ArgumentParser(description='Example of parser:'
                                                 'python inference.py --audio_path ./test_data/8.전기사고_244939 --model_path ./ckpt/best_audio_model_97.1.pth')

    parser.add_argument("--model_path", default='ckpt_17/best_audio_model_no_norm.pth', type=str, 
                        help="the trained model you want to test")
    parser.add_argument('--audio_path', default='./evaluate_data/', 
                        help='the audio you want to predict, sample rate 16k.',
                        type=str)

    args = parser.parse_args()

    label_csv = './data/Emergency_class_labels_indices_17.csv'       # label and indices for audioset data


    # assume each input spectrogram has 100 time frames
    input_tdim = 1024

    # 1. load the best model and the weights
    base_model_path = './ckpt/'

    ckpt_list = os.listdir(base_model_path)

    for ckpts in ckpt_list:
        checkpoint_path = base_model_path + ckpts
        ast_mdl = ASTModel(label_dim=int(ckpts[-2:]), input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
        ckpt_list = os.listdir(checkpoint_path)
        for ckpt in ckpt_list:
            checkpoint_paths = checkpoint_path + '/' + ckpt
            print(f'[*INFO] load checkpoint: {checkpoint_paths}')
            checkpoint = torch.load(checkpoint_paths, map_location='cpu')
            audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
            audio_model.load_state_dict(checkpoint)

            #audio_model = audio_model.to(torch.device("cuda:0"))
            audio_model.eval()     
            
            # 2. make feature for predict
            audio_paths = os.listdir(args.audio_path)
            audio_paths.sort()
                                            # set the eval model
            
            labels = load_label(label_csv)
            print('[*INFO] predice results:')
            with torch.no_grad():
                if not os.path.exists('test_result/' + ckpts):
                    os.mkdir('test_result/' + ckpts)
                f = open('test_result/' + ckpts + '/' + ckpt + '.csv', 'w', newline='', encoding='utf-8-sig')
                wr = csv.writer(f)
                wr.writerow([ckpt])
                wr.writerow(['filename', '1', '1', '2', '2', '3', '3'])
                for audio_path in audio_paths:
                    # 오디오 데이터
                    
                    feats = make_features(args.audio_path + audio_path, mel_bins=128)           # shape(1024, 128)

                    # 3. feed the data feature to model
                    feats_data = feats.expand(1, input_tdim, 128)           # reshape the feature

                    output = audio_model.forward(feats_data)
                    output = torch.sigmoid(output)
                    result_output = output.data.cpu().numpy()[0]

                    # 4. map the post-prob to label

                    sorted_indexes = np.argsort(result_output)[::-1]

                    row_result = []
                    print(audio_path)
                    row_result.append(audio_path)
                    # Print audio tagging top probabilities
                    for k in range(3):
                        if k == 0:
                            print('{}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]],
                                                    result_output[sorted_indexes[k]]))
                        #row_result.append([np.array(labels)[sorted_indexes[k]],
                        #                        result_output[sorted_indexes[k]]])
                        row_result.append(np.array(labels)[sorted_indexes[k]])
                        row_result.append(result_output[sorted_indexes[k]])
                    wr.writerow(row_result)
                    print()
            f.close()