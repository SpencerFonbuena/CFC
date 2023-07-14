import torch
import wandb
import glob
import os
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import torch.cuda.amp as amp

from torch.utils.data import DataLoader
from modules.data_process import Create_Dataset
from modules.crossformer import Crossformer
from sklearn.preprocessing import StandardScaler
from modules.loss import Myloss
from modules.hyperparameters import Hyperparameters as hp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from modules.tools import pre_process

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
scaler = StandardScaler()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # select device CPU or GPU
print(f'use device: {DEVICE}')

'''
datafile, window_size, pred_size, split, scaler, mode = str
'''

wandb.init(project='garb LIV', name="01")
if torch.cuda.is_available():
    path = '/root/datasets/Stocks'
else:
    path = '/Users/spencerfonbuena/Documents/Stocks'




'''colossalai.launch_from_torch(
    config='DNNM/modules/hyperparameters.py',
)'''


def pipeline(datafile):
    train_dataset = Create_Dataset(datafile=datafile, window_size=hp.lookback, pred_size=hp.pred_size, split=hp.split, mode='train')
    test_dataset = Create_Dataset(datafile=datafile, window_size=hp.lookback, pred_size=hp.pred_size, split=hp.split, mode='test')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=hp.num_workers,pin_memory=True,  drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=hp.batch_size, shuffle=False, num_workers=hp.num_workers,pin_memory=True)


    return train_dataloader, test_dataloader

'''
d_ff, n_heads, e_layers, 
                dropout, baseline, device):

'''

def model():
    net = Crossformer(data_dim=hp.data_dim, in_len=hp.lookback, out_len=hp.pred_size, seg_len=hp.seg_len,
                      win_size=hp.win_size, d_model=hp.d_model, baseline=hp.baseline, d_ff=hp.d_ff, n_heads=hp.n_heads, 
                      e_layers=hp.e_layers, dropout=hp.dropout, factor=hp.factor, device=DEVICE).to(DEVICE)
    return net

def main():
    net = model()
    loss_function = Myloss()
    optimizer = optim.AdamW(net.parameters(), lr = hp.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=1, threshold=.0001, )
    gradscaler = amp.GradScaler()
    net.train()
    for epochs in tqdm(range(10)):
        for datafolder in glob.glob(os.path.join(path, 'data*')):
            for datafile in glob.glob(os.path.join(datafolder, '*.txt')):
                with open(os.path.join(os.getcwd(), datafile), 'r') as f:
                    print(datafile)
                    df = pre_process(datafile=datafile)
                    df = scaler.fit_transform(df)
                    if len(df) >= 1000:
                        train_dataloader, test_dataloader = pipeline(df)
                        for i, (x,y) in enumerate(train_dataloader):
                            x, y = x.to(DEVICE), y.to(DEVICE)
                            print(x.shape, y.shape)
                            with amp.autocast(dtype=torch.float16):
                                y_pred = net(x)
                                loss = loss_function(y_pred, y)
                            gradscaler.scale(loss).backward()
                            if i % 4 == 0:
                                gradscaler.step(optimizer=optimizer)
                                gradscaler.update()
                                optimizer.zero_grad()
                        wandb.log({'Loss': loss})
                        wandb.log({'Epoch': epochs})

 
                pre = y_pred.cpu().detach().numpy()[0,:,0]
                ys = y.cpu().detach().numpy()[0,:,0]
                fig, ax = plt.subplots()
                ax.plot(pre, label='predictions')
                ax.plot(ys, label ='actual')
                plt.legend()
                wandb.log({'train plot': wandb.Image(fig)})
                plt.close()
                #test(net=net, dataloader=test_dataloader, optimizer=optimizer, loss_function=loss_function)
        scheduler.step(loss.mean())
def test(net, dataloader, optimizer, loss_function):
    net.eval()
    for epochs in range(10):
        for i, (x,y) in enumerate(dataloader):
            optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = net(x)
            if i % 20 == 0:
                pre = y_pred.cpu().detach().numpy()[0,:,0]
                ys = y.cpu().detach().numpy()[0,:,0]
                fig, ax = plt.subplots()
                ax.plot(pre, label='predictions')
                ax.plot(ys, label ='actual')
                plt.legend()
                wandb.log({'test plot': wandb.Image(fig)})
                plt.close()

if __name__ == '__main__':
    main()