import os
import glob
import json
import datetime as dt
import gradio as gr
from gradio import components
import torch
import numpy as np
import nltk
import params
from model import GradTTS
from text import text_to_sequence, sequence_to_text
from text.symbols import symbols
from utils import intersperse
from scipy.io.wavfile import write as write_wav

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


def save_wav(audio_array, filename):
    write_wav(filename, 22050, audio_array)


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

    
def tts(text, temperature=1.5, length_scale=0.91, timesteps=10):
    spk = None # LJSpeech has just 1 speaker
    test_parts = []
    sent_text = nltk.sent_tokenize(text)
    silenceshort = np.zeros(int((float(500) / 1000.0) * 22050), dtype=np.int16)
    
    for j, text in enumerate(sent_text):
        with torch.no_grad():
            # print(f'Synthesizing {j} text...', end=' ')
            sequence = text_to_sequence(text)
            sq_to_txt = sequence_to_text(sequence)
            print(sq_to_txt)
            x = torch.LongTensor(intersperse(sequence, len(symbols))).cuda()[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
            
            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=timesteps, temperature=temperature,
                                                   stoc=False, spk=spk, length_scale=length_scale)
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            test_parts += [audio]
            test_parts += [silenceshort.copy()]
            
    save_wav(np.concatenate(test_parts), 'out/combined_audio.wav')
    
    return 'out/combined_audio.wav'
    
    
if __name__ == "__main__":
    input_text = gr.components.Textbox(lines=20, label="Text")
    gr.Interface(
        fn=tts,
        inputs=[input_text],
        outputs=components.Audio(type='filepath', label="Generated Speech"),
        live=False
    ).launch()