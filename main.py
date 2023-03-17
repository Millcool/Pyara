from config import *
from test_model import *
from train_model import *
from transforms import *
from Wandb_functions import *
from metrics import *
from dataset import *
from dataloader import *
from model import *


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

time.sleep(120)
time.sleep(30)
for i in range(1000):
    time.sleep(0.1)
    print("Hi",i)
time.speep(20)

train_model(model,optimizer,train_loader,valid_loader, criterion, directory)


test_model(test_loader, model)

run.finish()