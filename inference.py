# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
import os
import glob
import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

from vocoders.env import AttrDict
from vocoders.hifigan_model import Generator as HiFiGAN


HIFIGAN_CONFIG = './vocoders/config.json'
HIFIGAN_CHECKPT = './vocoders/vocoder.pth'


def extract_digits(f):
    digits = "".join(filter(str.isdigit, f))
    return int(digits) if digits else -1


def latest_checkpoint_path(dir_path, regex="G_[0-9]*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: extract_digits(f))
    x = f_list[-1]
    print(f"latest_checkpoint_path:{x}")
    return x



def tts(args):
    if not isinstance(args.speaker_id, type(None)):
        assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([args.speaker_id]).cuda()
    else:
        spk = None
    
    print('Initializing Grad-TTS...')
    log_dir = params.log_dir
    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    
    model_path = latest_checkpoint_path(log_dir)
    checkpoint_dict = torch.load(model_path, map_location="cpu")
    generator.load_state_dict(checkpoint_dict["state_dict"])
    _ = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')
    
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()
    
    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'Synthesizing {i} text...', end=' ')
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
            
            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                   stoc=False, spk=spk, length_scale=0.91)
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            
            write(f'./out/sample_{i}.wav', 22050, audio)

    print('Done. Check out `out` folder for samples.')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default="synthesis.txt", help='path to a file with texts to synthesize')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None, help='speaker id for multispeaker model')
    args = parser.parse_args()
    
    tts(args)
