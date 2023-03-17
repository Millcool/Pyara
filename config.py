"""
Module with Config and imports
"""
# Main imports
import glob
import sys
import warnings
import os
import time
import random
from collections import defaultdict

# DS library imports

from scipy import stats
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchaudio
from tqdm.auto import tqdm
import copy
import shutil

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d
# Import visualizations
import matplotlib.pyplot as plt
from IPython import display
import wandb

warnings.filterwarnings("ignore")


class CFG:
    """
    Class with main variables, which we can modify
    """
    JUST_PREDICT = False
    DEBUG = False
    FULL_DATA = True
    wandb_on = False
    seed = 101
    model_name = 'LSTM'
    archive_name = 'Audio'
    train_bs = 128
    valid_bs = 128
    mels = 80
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 48000
    num_item_all = 1000 if DEBUG else 611829  # 611829  45235
    num_test = 10 if DEBUG else 301  # 1000
    print_every = 1 if DEBUG else 50  # 500
    epochs = 25 if DEBUG else 40  # 35
    ###############################################
    lr = 0.002
    criterion = nn.CrossEntropyLoss()
    scheduler = None # 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = int(30000 / train_bs * epochs) + 50
    T_0 = 25
    warmup_epochs = 0
    wd = 1e-6
    n_accumulate = max(1, 32 // train_bs)
    num_classes = 2
    classes = [0, 1]
    activation = None  # 'softmax'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_path = "E:/Audio/ASV/clips/"
    save_path = "./result/"
    train_path = "C:/Users/79671/Desktop/ML/Datasets/Audio/ASVspoof2021/clips/"  # "/app/data/"
    csv_path = 'valid'
    best_model_w = f''
    wandb_project_name = "Docker"
    wandb_run_name = f"{model_name}, Epochs: {epochs}, Samples: {num_item_all},BS: {train_bs}, MY_PC"
