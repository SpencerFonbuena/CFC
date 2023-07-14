import numpy as np
import torch
import json
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj=='type1':
        lr_adjust = {2: args.learning_rate * 0.5 ** 1, 4: args.learning_rate * 0.5 ** 2,
                     6: args.learning_rate * 0.5 ** 3, 8: args.learning_rate * 0.5 ** 4,
                     10: args.learning_rate * 0.5 ** 5}
    elif args.lradj=='type2':
        lr_adjust = {5: args.learning_rate * 0.5 ** 1, 10: args.learning_rate * 0.5 ** 2,
                     15: args.learning_rate * 0.5 ** 3, 20: args.learning_rate * 0.5 ** 4,
                     25: args.learning_rate * 0.5 ** 5}
    else:
        lr_adjust = {}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class StandardScaler():
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args

def string_split(str_for_split):
    str_no_space = str_for_split.replace(' ', '')
    str_split = str_no_space.split(',')
    value_list = [eval(x) for x in str_split]

    return value_list

def create_labels(df):

    A = 0
    C = 0
    labels = np.array([])
    
    for i in range(0, (len(df))):


        #find 1 percent and 2 percent above and below
        #print(df[A])
        one_low = df[A] * .99
        two_low = df[A] * .98
        one_high = df[A] * 1.01
        two_high = df[A] * 1.02

        #print(f'1 low: {one_low} | 2 low: {two_low} | 1 high: {one_high} | 2 high: {two_high}')
        #initialize the label counter
        label_counter = A

        #this is to make sure that once it either enters the "gone up by one percent" or "gone down by 1 percent"
        #it doesn't enter the other while loops
        pathway = 0

        try:
            #look for the instance when the price increases or decreases by 1 percent
            while df[label_counter] >= one_low and df[label_counter] <= one_high:
                label_counter += 1
                #print(df[label_counter])
            #If the price moved up 1 pecent first, this while loop will trigger and check if it is a two to one, or a one to one trade
            while df[label_counter] >= one_low and df[label_counter] <= two_high:
                label_counter += 1
                pathway = 1
                #print(df[label_counter])
            #Check if price has increased two percent
            if df[label_counter] >= two_high:
                labels = np.append(labels, 2)
                pathway = 1
                #print(df[label_counter])
            #check if price has reversed back down to the one percent marker
            if df[label_counter] <= one_low and pathway == 1:
                labels = np.append(labels, 1)
                #print(df[label_counter])
            
            #if the price moved down 1 pecent first, this will check if it is a two to one, or a one to one trade
            while df[label_counter] <= one_high and df[label_counter] >= two_low and pathway != 1:
                label_counter += 1
                pathway = 2
                #print(df[label_counter])
        
            #check if the price has continued down two percent
            if df[label_counter] <= two_low and pathway != 1:
                labels = np.append(labels, 0)
                #print(df[label_counter])
            #check if price reversed back up to the 1 percent above marker
            if df[label_counter] >= one_high and pathway != 1:
                labels = np.append(labels, 1)
                #print(df[label_counter])
            
            #temporarily store the last label that was added to "labels=[]"
            C = labels[-1]

        except:
            break
        #increment the graph by one time interval
        A += 1 

    #Create an array with the last value before the classification algorithm stopped
    array_append = []
    while A < len(df):
        array_append = np.append(array_append, C)
        A += 1
        

    labels = np.append(labels, array_append)
    return labels

def pre_process(datafile):
    df = pd.read_csv(datafile, sep=',', index_col=0, header=None, names=["Date", 'Open', 'High', 'Low', 'Close', 'Volume'])
    labels = create_labels(df['Close'])
    df['Open'] = df['Open'].pct_change()
    df['High'] = df['High'].pct_change()
    df['Low'] = df['Low'].pct_change()
    df['Close'] = df['Close'].pct_change()
    df['Volume'] = df['Volume'].pct_change()
    df['Labels'] = labels
    return df[1:]