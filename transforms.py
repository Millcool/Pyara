"""
Module of transformations for audio signal
"""
import torch
import torchaudio
import random
from torch import distributions
import librosa.effects
from config import CFG

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=CFG.SAMPLE_RATE,
    n_fft=1024,
    win_length = 1024,
    hop_length=256,
    n_mels= CFG.mels,
    window_fn = torch.hann_window,
    center=False
)

MFCC_spectrogram = torchaudio.transforms.MFCC(
    sample_rate=CFG.SAMPLE_RATE,
    n_mfcc = CFG.mels,
    melkwargs={
        "n_fft": 1024,
        "n_mels": CFG.mels,
        "hop_length": 256,
        "mel_scale": "htk",
        'win_length': 1024,
        'window_fn': torch.hann_window,
        'center':False
    },
)

def probability_augmentetion(prob):
    return True if random.random() < prob else False

def audio_augmentations(wav):
    # Gausian Nooise augmentation
    if probability_augmentetion(0.5):
        noiser = distributions.Normal(0, 0.05)
        wav = wav + noiser.sample(wav.size())
        wav = torch.clamp(wav, -1, 1)
    # Time stratching - boost
    if probability_augmentetion(0.25):
        wav = librosa.effects.time_stretch(wav.numpy().squeeze(), rate=random.uniform(1, 2))
        wav = torch.from_numpy(wav)

    # Time stratching - slowing down
    elif probability_augmentetion(0.25):
        wav = librosa.effects.time_stretch(wav.numpy().squeeze(), rate=random.uniform(0.5, 1))
        wav = torch.from_numpy(wav)

    # Volume
    if probability_augmentetion(0.2):
        # quiet
        if probability_augmentetion(0.5):
            valer = torchaudio.transforms.Vol(gain=random.uniform(0.2, 1), gain_type='amplitude')
            wav = valer(wav)
        # Louder
        else:
            valer = torchaudio.transforms.Vol(gain=random.uniform(1, 5), gain_type='amplitude')
            wav = valer(wav)

    if probability_augmentetion(0.01):
        if probability_augmentetion(0.5):
            shifter = torchaudio.transforms.PitchShift(sample_rate=16000, n_steps=-1)
            wav = shifter(wav)
        else:
            shifter = torchaudio.transforms.PitchShift(sample_rate=16000, n_steps=1)
            wav = shifter(wav)

    return wav

