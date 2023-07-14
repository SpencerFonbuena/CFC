import  torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

seed = 10
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#print(f'Use device {DEVICE}')

#wandb.init(project='mach39', name='03')

#Create_Dataset class that inherits the attributes and methods of torch.utils.data.Dataset
class Create_Dataset(Dataset):
    def __init__(self, datafile, window_size, pred_size, split, mode = str ): # datafile -> csv file | window_size -> # of timesteps in each example | split -> The percent of data you want for training
        
  
        #Create the training data
        datafile = pd.DataFrame(datafile)
        
        rawtrainingdata = datafile.drop(columns=[5]).to_numpy()
        #create the labels
        rawtraininglabels = datafile[5].to_numpy()
        # [End pre-processing]


        # [General Functionalities]

        #The data we want will depend if we are in train, validate, or test mode.
        self.mode = mode
        #create a split value to separate validate from training
        self.split = int(len(datafile) * split)

        # [End Functionalities]




  
        
        # [Window the dataset]
        
        ''#First array is a row vector, which broadcasts with dataset_array
        window_array = np.array([np.arange(window_size)])
        #This is a column vector (as shown by the reshape, to have a 1 in the column dimension) that broadcasts with window_array
        dataset_array = np.array(np.arange(len(rawtrainingdata)-window_size - pred_size + 1)).reshape(len(rawtrainingdata)-window_size - pred_size + 1, 1)
        #broadcast the data together
        indexdata = window_array + dataset_array
        # Index into the raw training data with our preset windows to create datasets quickly
        trainingdata = rawtrainingdata[indexdata]
        #[End windowing dataset]


        # [Window the dataset]
        
        #First array is a row vector, which broadcasts with dataset_array
        window_array = np.array([np.arange(pred_size)])
        #This is a column vector (as shown by the reshape, to have a 1 in the column dimension) that broadcasts with window_array
        dataset_array = np.array(np.arange(len(rawtrainingdata)- pred_size-window_size+ 1)).reshape(len(rawtrainingdata) - pred_size - window_size+ 1, 1)
        #broadcast the data together
        indexlabeldata = window_array + dataset_array + window_size
        # Index into the raw training data with our preset windows to create datasets quickly
        labeldata = rawtraininglabels[indexlabeldata]
   
        #[End windowing dataset]''




        #[beginning of creating test data and labels]

        #create the training data and labels
        self.trainingdata = torch.tensor(trainingdata[:self.split]).to(torch.float32)
        self.traininglabels = torch.tensor(labeldata[:self.split]).to(torch.float32)
        #can't call the iterate through a torch, so this is to create all of the weights
        

        #[end of creating training data and labels]




        #[beginning of creating validation data and labels]
        
        #create the validation data and labels
        self.valdata = torch.tensor(trainingdata[self.split:]).to(torch.float32)
        self.vallabels = torch.tensor(labeldata[self.split:]).to(torch.float32)
        #[end of creating validation data and labels]
        
        


    
    
    def __getitem__(self, index):
        if self.mode == 'train':
            return self.trainingdata[index], self.traininglabels[index]
        elif self.mode == 'test':
            return self.valdata[index], self.vallabels[index]
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.trainingdata)
        if self.mode == 'test':
            return len(self.vallabels)

        
        