import torch
from torch import nn


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
    width = 300
    train_bs = 128
    valid_bs = 128
    mels = 80
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 48000
    num_item_all = 1000 if DEBUG else 45235  # 611829  45235
    num_test = 10 if DEBUG else 301  # 1000
    print_every = 1 if DEBUG else 50  # 500
    epochs = 25 if DEBUG else 40  # 35
    ###############################################
    lr = 0.003
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
    test_path = "C:/Users/79671/Desktop/ML/Datasets/Audio_datasets/ASVspoof2021/equal_audio"
    save_path = "./result/"
    train_path = "C:/Users/79671/Desktop/ML/Datasets/Audio_datasets/ASVspoof2021/equal_audio"  # "/app/data/"
    csv_path = 'equal_dataset'
    best_model_w = f''
    wandb_project_name = "LSTM_GMM"
    wandb_run_name = f"{model_name}, Epochs: {epochs}, Samples: {num_item_all},BS: {train_bs}, MY_PC"


#%%
