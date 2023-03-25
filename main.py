import os

import pandas as pd
import torch
from torch import nn

from Model.model import LSTM
from Model.test_model import test_model
from Model.train_model import train_model
from Wandb.Wandb_functions import wandb_init, wandb_login
from config import CFG
from dataloader import prepare_loaders

#%%
wandb_login()

data = pd.read_csv(CFG.csv_path, sep = '\t' ) #pd.read_csv(CFG.csv_path, sep = '\\t', header=None)

train_loader, valid_loader, test_loader = prepare_loaders(data)

device = CFG.device
print(f"Device: {device}, Available: {torch.cuda.is_available()}, Pytorch_verion: {torch.__version__}")
model = LSTM().to(device)
model.eval()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
criterion = nn.CrossEntropyLoss()

try:
    #shutil.rmtree('./result')
    #local_time = time.ctime().replace(' ', '_').replace(':', '.')
    directory = f'results/result'
    os.mkdir(directory)
    print('PC DIR CREATED')
except Exception:
    print("DIR NOT CREATED")
    pass

run = wandb_init()




train_model(model,optimizer,train_loader,valid_loader, criterion, directory)


test_model(test_loader, model)

run.finish()

#%%

#%%
