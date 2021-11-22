#!/usr/bin/env python
# coding: utf-8

# In[133]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install albumentations')

import pandas as pd
import math, random
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import sklearn
import shutil
import seaborn as sns
import os
import json 
import cv2
import albumentations as albu

from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import random_split
from torch.nn import init
from matplotlib import pyplot as plt
from matplotlib.mlab import window_hanning, specgram
from matplotlib.colors import LogNorm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from torchsummary import summary as summary


# In[142]:


import os
print('=====================================')
print('============= For Train =============')
print('=====================================')
for dirname, _, filenames in os.walk('/mnt/dataset'):
    print('=====================================')
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
print('=====================================')
print('=========== For Inference ===========')
print('=====================================')        
for dirname2, _, filenames2 in os.walk('/mnt/VOICE/dataset'):
    print('=====================================')
    for filename2 in filenames2:
        print(os.path.join(dirname2, filename2))


# In[143]:


# This block is for checking the structure of the given dataset

def plot_Melspectrogram(data, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title('Melspectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(data, cmap='viridis', interpolation= 'gaussian', aspect=aspect, vmin=1e-4, vmax=1e-1)
  # im = axs.imshow(data, cmap='viridis', interpolation= 'gaussian', aspect=aspect, vmin=1e-25, vmax=1e-2)
  # cmap='viridis', 'viridis_r', 'inferno', 'inferno_r', 'plasma', plt.cm.Blues, plt.cm.Blues_r, 'BrBG', 'BrBG_r'
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show()
    
    
# ■■■■■ Loading and Checking the [number] of dataset for training 
train_idx = []
for filename in os.listdir("/mnt/dataset/source/"): 
    if filename.startswith("data"):
        #print('=========== file name cut ===========')
        #print(filename[4:-8])
        train_idx.append(filename[4:-8])
train_idx_arr_all = np.asarray(train_idx)
train_idx = pd.DataFrame(train_idx_arr_all, columns = ['num'])
#print(train_idx)
print('\n\n=========== train_data_num(Used) ===========')
print(train_idx.head())
print(train_idx.shape)


# ■■■■■ Loading and Checking the [ID] of dataset for training 
print('\n\n=========== train_data_ID (Used) ===========')

train_id_all= []
for idx in train_idx_arr_all:
    with open('/mnt/dataset/metadata/data' +  str(idx)  +'_metadata.json', 'r') as f:
        #print(idx)
        train_meta_json = json.load(f)
        train_id = train_meta_json['diagnosis']       # Train id data

        if train_id == 'Normal':
            train_id = 0
        if train_id == 'Cancer':
            train_id = 1
        if train_id == 'Cyst_and_Polyp':
            train_id = 2
        if train_id == 'Nodules':
            train_id = 3
        if train_id == 'Functional_dysphonia':
            train_id = 4
        if train_id == 'Paralysis':
            train_id = 5 
        train_id_all = np.append(train_id_all, train_id)
train_id_all = np.asarray(train_id_all)
print(train_id_all)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■ Concatenating and Checking the [Num - ID] of dataset for training ■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

print('\n\n=========== train_data_ [Num-ID] (Used) ===========')
train_idx_arr_all = np.asarray(train_idx_arr_all)
train_id_all = np.asarray(train_id_all)


train_num_id_all = np.vstack((train_idx_arr_all, train_id_all))
train_num_id_all = train_num_id_all.transpose()
train_num_id_all = train_num_id_all.astype(np.float32)
train_num_id_all = train_num_id_all.astype(np.int32)
train_num_id_all = np.asarray(train_num_id_all)
train_num_id_all = pd.DataFrame(train_num_id_all, columns = ['num', 'id'])

allData_part = []
for id_idx in range(0, 6) :
    allData_part.append(train_num_id_all[train_num_id_all['id'] == id_idx].sample(n=200, replace = True))
train_idx_class_bal = pd.concat(allData_part, axis=0, ignore_index=False)
train_idx_class_bal = pd.DataFrame(train_idx_class_bal, columns = ['num', 'id'])
train_idx_class_bal = train_idx_class_bal.reset_index(drop = True, inplace=False)
print(train_idx_class_bal)  
print(train_idx_class_bal.shape)


# ■■■■■ Loading and Checking the [number] of dataset for training (Not Used)
train_data_num = pd.read_csv('/mnt/dataset/answer_format.csv')
train_data_num = train_data_num.values[:, :].astype('int32')
train_data_num = pd.DataFrame(train_data_num, columns = ['num', 'dummy'])
train_data_num = train_data_num.loc[:, ['num']]

print('\n\n=========== train_data_num(Not used) ===========')
print(train_data_num.head())
print(train_data_num.shape)
# ■■■■■ Loading and Checking the [csv] of dataset for training 
train_data_mel_example = pd.read_csv('/mnt/dataset/source/data892_mel.csv')
train_data_mel_example = train_data_mel_example.values[:,1:]
print('\n\n=========== train_data_Mel ===========')
print(train_data_mel_example.shape)
print(train_data_mel_example)
print(max(map(max, train_data_mel_example)))
print(min(map(min, train_data_mel_example)))
plot_Melspectrogram(train_data_mel_example, title=None, ylabel='freq_bin', aspect='auto', xmax=None)


# ■■■■■ Loading and Checking the [images] of dataset for training 
train_data_mel_example =cv2.imread('/mnt/dataset/images/data892.png', cv2.IMREAD_UNCHANGED)
print(train_data_mel_example.shape)
print('\n\n=========== train_data_spectrogram_example ===========')
plt.imshow(train_data_mel_example)
plt.show()

# ■■■■■ Loading and Checking the [metadata.json] of dataset for training 
# ■ Sample
with open('/mnt/dataset/metadata/data892_metadata.json', 'r') as f:
    train_data_meta_json_example = json.load(f)
print('\n\n=========== train_data_metadata.json_example ===========')
print('"wav_code", "source_type" --> are not unnecessary')
print(json.dumps(train_data_meta_json_example, indent="\t") )

# ■ All
path_to_meta_json = '/mnt/dataset/metadata/'
meta_json_files = [pos_json for pos_json in os.listdir(path_to_meta_json) if pos_json.endswith('.json')]
# print(meta_json_files)


# ■■■■■ Loading and Checking the [annotation.json] of dataset for training 
# ■ Sample
with open('/mnt/dataset/annotation/data892_annotation.json', 'r') as f:
    train_data_anno_json_example = json.load(f)
print('\n\n=========== train_data_anno.json_example ===========')
print('"startFrame", "endFrame" --> x-coordinate value for column of csv or image')
print(json.dumps(train_data_anno_json_example, indent="\t") )

# ■ All
path_to_anno_json = '/mnt/dataset/annotation/'
anno_json_files = [pos_json for pos_json in os.listdir(path_to_anno_json) if pos_json.endswith('.json')]
# print(anno_json_files)


# In[144]:


batchsize = 8
mel_time_size = 2048
mel_freq_size = 256

train_data_path = '/mnt/dataset'

class train_SoundData(Dataset):
    def __init__(self, df, data_path, transforms=None):
        self.df = df
        self.data_path = str(data_path)
        self.transforms = transforms

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        train_mel_path = self.data_path + '/source/data' + str(self.df.loc[idx, 'num']).zfill(3) + '_mel.csv'
        #print(idx)
        train_mel = pd.read_csv(train_mel_path)
        train_mel = train_mel.values[:,:] 
        
        if self.transforms is not None:
            augmented = self.transforms(image=train_mel)
            train_mel = augmented['image']   
            train_mel = train_mel.astype(np.float32)  # Train melspectrogram data
            
        with open(self.data_path +'/metadata/data' +  str(self.df.loc[idx, 'num']).zfill(3)  +'_metadata.json', 'r') as f:
            train_meta_json = json.load(f)
        train_id = train_meta_json['diagnosis']       # Train id data
       
        if train_id=='Normal':
            train_id=0
        if train_id=='Cancer':
            train_id=1
        if train_id=='Cyst_and_Polyp':
            train_id=2
        if train_id=='Nodules':
            train_id=3
        if train_id=='Functional_dysphonia':
            train_id=4
        if train_id=='Paralysis':
            train_id=5 
        train_id = np.asarray(train_id)
                
        
        #  Normal (정상) 0 / Cancer (후두암) 1/ Cyst_and_Polyp (물혹/폴립) 2/ Nodules (성대결절)  3
        #  Functional_dysphonia (기능성 음성질환) 4/ Paralysis (성대 마비) 5
        
        
        return train_mel, train_id


resize_transforms = albu.Compose([albu.Resize(mel_time_size, mel_freq_size, always_apply=True), ])

train_data       = train_SoundData(train_idx_class_bal, train_data_path, resize_transforms)  # train_idx  : not balanced
train_dataset    = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True)

def trash_check():
    trash_check= 0
    for batch_idx_trash, (mels_trash, targets_trash) in enumerate(train_dataset):
        trash_check+= 1
        if trash_check ==1 :
            print('\n\n=========== trash_check',trash_check, '===========') 
            print(mels_trash)
            print(targets_trash)
        else:
            break;
trash_check()


# In[215]:


def conv_1(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,1,1),
        nn.ReLU(),
    )
    return model

def conv_1_3(in_dim,mid_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,mid_dim,1,1),
        nn.ReLU(),
        nn.Conv2d(mid_dim,out_dim,3,1,1),
        nn.ReLU()
    )
    return model
    
def conv_1_5(in_dim,mid_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,mid_dim,1,1),
        nn.ReLU(),
        nn.Conv2d(mid_dim,out_dim,5,1,2),
        nn.ReLU()
    )
    return model
    

def max_3_1(in_dim,out_dim):
    model = nn.Sequential(
        nn.MaxPool2d(3,1,1),
        nn.Conv2d(in_dim,out_dim,1,1),
        nn.ReLU(),
    )
    return model

class inception_module(nn.Module):
    def __init__(self,in_dim,out_dim_1,mid_dim_3,out_dim_3,mid_dim_5,out_dim_5,pool):
        super(inception_module,self).__init__()
        self.conv_1 = conv_1(in_dim,out_dim_1)
        self.conv_1_3 = conv_1_3(in_dim,mid_dim_3,out_dim_3)
        self.conv_1_5 = conv_1_5(in_dim,mid_dim_5,out_dim_5)
        self.max_3_1 = max_3_1(in_dim,pool)

    def forward(self,x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_1_3(x)
        out_3 = self.conv_1_5(x)
        out_4 = self.max_3_1(x)
        output = torch.cat([out_1,out_2,out_3,out_4],1)
        return output
    
# ----------------------------
# Melspectro Classifier Model
# ----------------------------
class Melspectro_classifier(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        
#         self.layer_1 = nn.Sequential(
#             nn.Conv2d(1,16,7,2,3),
#             #nn.MaxPool2d(3,2,1),
#             nn.Conv2d(16,16*3,3,1,1),
#             #nn.MaxPool2d(3,2,1),
#         )
#         self.layer_2 = nn.Sequential(
#             inception_module(16*3,64,96,128,16,32,32),
#             inception_module(256,128,128,192,32,96,64),
#             nn.MaxPool2d(3,2,1),
#         )
#         self.layer_3 = nn.Sequential(
#             inception_module(480,192,96,208,16,48,64),
#             inception_module(512,160,112,224,24,64,64),
#             inception_module(512,128,128,256,24,64,64),
#             inception_module(512,112,144,288,32,64,64),
#             inception_module(528,256,160,320,32,128,128),
#             nn.MaxPool2d(3,2,1),
#         )
#         self.layer_4 = nn.Sequential(
#             inception_module(832,256,160,320,32,128,128),
#             inception_module(832,384,192,384,48,128,128), 
#             nn.AvgPool2d(16,1),
#         )
#         self.layer_5 = nn.Dropout2d(0.4)
#         self.fc_layer1 = nn.Linear(1024*8,1024*4)
#         self.fc_layer2 = nn.Linear(1024*4,1024)
#         self.fc_layer3 = nn.Linear(1024,6)
#         self.print_shape = True

        self.layer_1 = nn.Sequential(
            nn.Conv2d(1,3,7,2,3),
            nn.MaxPool2d(3,2,1),
            nn.Conv2d(3,6,3,1,1),
            nn.MaxPool2d(3,2,1),
            nn.AvgPool2d(32,32),
        )
        self.layer_2 = nn.Sequential(
            inception_module(6,16,24,32,4,8,8),
            inception_module(64,8,8,12,2,6,4),
            nn.MaxPool2d(3,2,1),
        )
        self.layer_3 = nn.Sequential(
            inception_module(30,16,24,32,4,8,8),
            inception_module(64,8,8,12,2,6,4),
            nn.MaxPool2d(3,2,1),
            nn.AvgPool2d(8,57),
            

        )

        self.layer_5 = nn.Dropout2d(0.3)
        self.fc_layer1 = nn.Linear(30,16)
        self.fc_layer3 = nn.Linear(48,6)
        self.ReLU = nn.ReLU()
        self.print_shape = True

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        if self.print_shape: print('input----',x.shape)
        out = self.layer_1(x)
        if self.print_shape: print('layer 1----', out.shape)
#         out = self.layer_2(out)
#         if self.print_shape: print('layer 2----',out.shape)
#         out = self.layer_3(out)
#         if self.print_shape: print('layer 3----',out.shape)
        out = self.layer_5(out)
        if self.print_shape: print('layer 5----',out.shape)
        out = out.view(out.size(0),-1)
        if self.print_shape: print('layer before FC----',out.shape)
#         out = self.fc_layer1(out)
#         out = self.ReLU(out)
#         if self.print_shape: print('layer FC1----',out.shape)
        out = self.fc_layer3(out)
        if self.print_shape: print('layer FC3----',out.shape); self.print_shape = False

        return out
    
Mel_classifier = Melspectro_classifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device =  "cpu"
#Mel_classifier = Mel_classifier.to(device)

# summary(Mel_classifier, input_size=(1,256,2048), batch_size=8)

# Check that it is on Cuda
print('\n\n=========== Where the model is located ===========')
next(Mel_classifier.parameters()).device


# In[216]:


for i in Mel_classifier.named_children():
    print(i)


# In[217]:


num_epochs = 30
train_loss_values = np.zeros([num_epochs])

# ----------------------------
# Saver
# ----------------------------
def save_checkpoint(state, is_best, filename='Mel_classifier_incept1.pt'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_incept1.pt')

        
# ----------------------------
# Training 
# ----------------------------
def training(model, train_dataset, num_epochs):
    # Loss Function, Optimizer and Scheduler
    model.train()
    best_acc1 = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                  lr_lambda=lambda epoch: 0.96 ** epoch)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     from torch.optim.lr_scheduler import StepLR
#     scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    
    
    # Repeat for each epoch
    for epoch in range(num_epochs):
        print('\n=========== Epoch',epoch,  '===========')
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        train_loss_tot = 0
        
        
        # Repeat for each batch in the training set
        for i, data in enumerate(train_dataset):
            # Get the mels features and target labels, and put them on the GPU
            #print( data[0])
            #print( data[1])
            mels, labels = data[0].to(device), data[1].to(device)
            
            #■
            mels = torch.unsqueeze(mels, 1)
            
            # Normalize the mels
            mels_m, mels_s = mels.mean(), mels.std()
            mels = (mels - mels_m) / mels_s
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(mels)
            loss = criterion(outputs, labels)
            train_loss_tot += loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
  
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            acc = correct_prediction / total_prediction
        
        train_loss_tot /= len(mels)
        train_loss_values[epoch] = train_loss_tot
        print('train_loss_values[epoch] | ',train_loss_values[epoch])
    save_checkpoint(Mel_classifier , 1)
    print('Finished Training')


print('\n\n=====================================================')
print('================= Starting Training =================')
print('=====================================================\n')
# (n_samples, channels, height, width)
training(Mel_classifier, train_dataset, num_epochs)


# In[ ]:


x_len = np.arange(len(train_loss_values))
plt.plot( x_len ,train_loss_values, marker = '.',label="Train loss")
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# In[166]:


# It's for testing for training set (Not proper validation, Temporary validation)
tot_test_dataset = torch.utils.data.DataLoader(train_data, batch_size=24, shuffle=True)

def tot_testing(model, tot_test_dataset):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        model.eval()
        for data in tot_test_dataset:
            print(data)
            # Get the input features and target labels, and put them on the GPU
            mels, labels = data[0].to(device), data[1].to(device)
            labels = labels.cpu()
            #■
            mels = torch.unsqueeze(mels, 1)

            # Normalize the mels
            mels_m, mels_s = mels.mean(), mels.std()
            mels = (mels - mels_m) / mels_s

            # Get predictions
            outputs = model(mels)
            outputs = outputs.cpu()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
            acc = correct_prediction / total_prediction

            print(classification_report(labels, prediction, digits = 6))
            confmat = confusion_matrix(y_true=labels, y_pred=prediction)
            sns.heatmap(confmat, annot = True, fmt ='d',cmap = 'Blues')
            plt.xlabel('predicted label')
            plt.ylabel('true label')
            plt.show()

    print(f'Accuracy: {acc:.6f}, Total items: {total_prediction}')

tot_testing(Mel_classifier, tot_test_dataset)


# In[ ]:




