#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install albumentations')
get_ipython().system(' apt-get install -y libsndfile-dev')

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
import torchaudio
import copy

from torchaudio import transforms
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import random_split
from torch.nn import init
from matplotlib import pyplot as plt
from matplotlib.mlab import window_hanning, specgram
from matplotlib.colors import LogNorm
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
np.random.seed(1024)


# In[2]:


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


# In[3]:


#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class AudioUtil():
#   # ----------------------------
#   # Load an audio file. Return the signal as a tensor and the sample rate
#   # ----------------------------
#   @staticmethod
#   def open(audio_file):
#     sig, sr = torchaudio.load(audio_file)
#     return (sig, sr)

# ----------------------------
# Since Resample applies to a single channel, we resample one channel at a time
# ----------------------------
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
          # Nothing to do
          return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
          # Resample the second channel and merge both channels
          retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
          resig = torch.cat([resig, retwo])

        return ((resig, newsr))
# ----------------------------
# Convert the given audio to the desired number of channels
# ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
          # Nothing to do
          return aud

        if (new_channel == 1):
          # Convert from stereo to mono by selecting only the first channel
          resig = sig[:1, :]
        else:
          # Convert from mono to stereo by duplicating the first channel
          resig = torch.cat([sig, sig])

        return ((resig, sr))
# ----------------------------
# Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
# ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if (sig_len > max_len):
          # Truncate the signal to the given length
          sig = sig[:, :max_len]

        elif (sig_len < max_len):
          # Length of padding to add at the beginning and end of the signal
          pad_begin_len = random.randint(0, max_len - sig_len)
          pad_end_len = max_len - sig_len - pad_begin_len

          # Pad with 0s
          pad_begin = torch.zeros((num_rows, pad_begin_len))
          pad_end = torch.zeros((num_rows, pad_end_len))

          sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

# ----------------------------
# Shifts the signal to the left or right by some percent. Values at the end
# are 'wrapped around' to the start of the transformed signal.
# ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)
    def melspectro_gram(aud, n_mels=64, n_fft=1024, hop_len=512):
        sig,sr = aud
        top_db = 100
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        #spec = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_len)(sig)
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)

        return (spec)

    def spectro_augment(spec, max_mask_pct=0.06, n_freq_masks= 4, n_time_masks=4):
        #print('spec shape', spec.shape)
        n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        #print('mask_value', mask_value)
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec
    
# This block is for checking the structure of the given dataset

def plot_Melspectrogram(data, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title('Melspectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(data, cmap='viridis', interpolation= 'gaussian', aspect=aspect, vmin=0, vmax=2)
    # im = axs.imshow(data, cmap='viridis', interpolation= 'gaussian', aspect=aspect, vmin=1e-25, vmax=1e-2)
    # cmap='viridis', 'viridis_r', 'inferno', 'inferno_r', 'plasma', plt.cm.Blues, plt.cm.Blues_r, 'BrBG', 'BrBG_r'
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show()
    
    
    
    
    
    


# In[4]:


n_mels = 256

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
print('\n\n=========== train_data_ID (Used) ===========')
print(train_id_all)


train_age_all= []
for idx in train_idx_arr_all:
    with open('/mnt/dataset/metadata/data' +  str(idx)  +'_metadata.json', 'r') as f:
        #print(idx)
        train_meta_json = json.load(f)
        train_age = train_meta_json['age']       

        if train_age <= 10:
            train_age = 0
        elif train_age <= 20:
            train_age = 1    
        elif train_age <= 40:
            train_age = 2 
        elif train_age <= 60:
            train_age = 3 
        elif train_age <= 70:
            train_age = 4
        elif train_age > 70:
            train_age = 5 
        train_age_all = np.append(train_age_all, train_age)
train_age_all = np.asarray(train_age_all)
print('\n\n=========== train_data_age (Used) ===========')
print(train_age_all)


train_sex_all= []
for idx in train_idx_arr_all:
    with open('/mnt/dataset/metadata/data' +  str(idx)  +'_metadata.json', 'r') as f:
        #print(idx)
        train_meta_json = json.load(f)
        train_sex = train_meta_json['sex']       

        if train_sex  == 'M':
            train_sex = 0
        if train_sex  == 'F':
            train_sex = 1
        train_sex_all = np.append(train_sex_all, train_sex)
train_sex_all = np.asarray(train_sex_all)
print('\n\n=========== train_data_sex (Used) ===========')
print(train_sex_all)
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ■■■■■ Concatenating and Checking the [Num - ID] of dataset for training ■■■■■
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

print('\n\n=========== train_data_ [Num-ID] (Used) ===========')
train_idx_arr_all = np.asarray(train_idx_arr_all)
train_id_all = np.asarray(train_id_all)
# train_age_all
# train_sex_all

train_num_id_all = np.vstack((train_idx_arr_all, train_age_all, train_sex_all , train_id_all)) # 번호 | 나이 | 성별 | 질병
train_num_id_all = train_num_id_all.transpose()
train_num_id_all = train_num_id_all.astype(np.float32)
train_num_id_all = train_num_id_all.astype(np.int32)
train_num_id_all = np.asarray(train_num_id_all)
train_num_id_all = pd.DataFrame(train_num_id_all, columns = ['num', 'age', 'sex', 'id'])

allData_part = []
for id_idx in range(0, 6) :
    allData_part.append(train_num_id_all[train_num_id_all['id'] == id_idx].sample(n=400, replace = True))
train_idx_class_bal = pd.concat(allData_part, axis=0, ignore_index=False)
train_idx_class_bal = pd.DataFrame(train_idx_class_bal, columns = ['num', 'age', 'sex', 'id'])
train_idx_class_bal = train_idx_class_bal.reset_index(drop = True, inplace=False)
print(train_idx_class_bal)  
print(train_idx_class_bal.shape)


# ■■■■■ Loading and Checking the [number] of dataset for training (Not Used)
# train_data_num = pd.read_csv('/mnt/dataset/answer_format.csv')
# train_data_num = train_data_num.values[:, :].astype('int32')
# train_data_num = pd.DataFrame(train_data_num, columns = ['num', 'dummy'])
# train_data_num = train_data_num.loc[:, ['num']]

# print('\n\n=========== train_data_num(Not used) ===========')
# print(train_data_num.head())
# print(train_data_num.shape)
# ■■■■■ Loading and Checking the [csv] of dataset for training 
train_data_mel_example = pd.read_csv('/mnt/dataset/source/data892_mel.csv')
train_data_mel_example = -0.05 * np.log(train_data_mel_example.values[:,1:])
train_data_mel_example = AudioUtil.spectro_augment(torch.Tensor(train_data_mel_example), max_mask_pct=0.06, n_freq_masks=4, n_time_masks=4)

print('\n\n=========== train_data_Mel ===========')
print(train_data_mel_example.shape)
print(train_data_mel_example)
print(max(map(max, train_data_mel_example)))
print(min(map(min, train_data_mel_example)))
plot_Melspectrogram(train_data_mel_example, title=None, ylabel='freq_bin', aspect='auto', xmax=None)

# train_audio_est = torchaudio.transforms.InverseMelScale(train_data_mel_example, sample_rate: int = 44100, n_fft: int = 2048, hop_length: Optional[int] = 1024, n_mels: int = 256)
# train_mel_est = torchaudio.transforms.MelScale(train_audio_est, sample_rate: int = 44100, n_fft: int = 2048, hop_length: Optional[int] = 1024, n_mels: int = 256)
# train_audio_est = torchaudio.transforms.InverseMelScale(train_data_mel_example, sample_rate=44100, hop_length=1024, n_mels=256)
# train_mel_est = torchaudio.transforms.MelScale(train_audio_est, sample_rate= 44100, n_fft=2048, hop_length= 1024, n_mels=256)







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


# In[1]:


batchsize = 64
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

#         train_audio_est = torchaudio.transforms.InverseMelScale(mel)
        
        if self.transforms is not None:
            augmented = self.transforms(image=train_mel)
            train_mel = augmented['image']   
            train_mel = train_mel.astype(np.float32)  # Train melspectrogram data
            
        train_mel = -0.05 * np.log(train_mel+0.0001)
        train_mel = AudioUtil.spectro_augment(torch.Tensor(train_mel), max_mask_pct=0.06, n_freq_masks=4, n_time_masks=4)
        train_mel = train_mel.numpy()
#         noise_mean = 0
#         noise_var = 0.1
#         noise_sigma = noise_var**0.5
#         noise_gauss = np.random.normal(noise_mean,noise_sigma,(256,2048,1))
#         noise_gauss = noise_gauss.reshape(256,2048,1)
#         train_mel = train_mel + noise_gauss
#         train_mel[train_mel<=0] = 0.0001
        
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
        
        train_age= []
        with open(self.data_path +'/metadata/data' +  str(self.df.loc[idx, 'num']).zfill(3)  +'_metadata.json', 'r') as f:
            #print(idx)
            train_meta_json = json.load(f)
            train_age = train_meta_json['age']       

            if train_age <= 10:
                train_age = 0
            elif train_age <= 20:
                train_age = 1    
            elif train_age <= 40:
                train_age = 2 
            elif train_age <= 60:
                train_age = 3 
            elif train_age <= 70:
                train_age = 4
            elif train_age > 70:
                train_age = 5 
        train_age = np.asarray(train_age).astype(np.float32)


        train_sex= []
        with open(self.data_path +'/metadata/data' +  str(self.df.loc[idx, 'num']).zfill(3)  +'_metadata.json', 'r') as f:
            #print(idx)
            train_meta_json = json.load(f)
            train_sex = train_meta_json['sex']       

            if train_sex  == 'M':
                train_sex = 0
            if train_sex  == 'F':
                train_sex = 1
        train_sex = np.asarray(train_sex).astype(np.float32)
                
        
        
        
        return train_mel,train_age, train_sex, train_id


resize_transforms = albu.Compose([albu.Resize(mel_time_size, mel_freq_size, always_apply=True), ])

train_data       = train_SoundData(train_idx_class_bal, train_data_path, resize_transforms)  # train_idx  : not balanced
train_dataset    = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True)

# def trash_check():
#     trash_check= 0
#     for batch_idx_trash, (mels_trash, age_trash, sex_trash, id_trash) in enumerate(train_dataset):
#         trash_check+= 1
#         if trash_check ==1 :
#             print('\n\n=========== trash_check',trash_check, '===========') 
#             print(mels_trash)
#             print(age_trash)
#             print(sex_trash)
#             print(id_trash)
#         else:
#             break;
# trash_check()


# In[6]:



# ----------------------------
# Melspectro Classifier Model
# ----------------------------
class Melspectro_classifier(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        # ■ Starting from single channel
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Second Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers += [self.conv5, self.relu5, self.bn5]
        
        # Second Convolution Block
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(256)
        init.kaiming_normal_(self.conv6.weight, a=0.1)
        self.conv6.bias.data.zero_()
        conv_layers += [self.conv6, self.relu6, self.bn6]
        self.conv = nn.Sequential(*conv_layers)
        
        
        # Second Convolution Block
        self.conv7 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu7 = nn.ReLU()
        self.bn7 = nn.BatchNorm2d(512)
        init.kaiming_normal_(self.conv7.weight, a=0.1)
        self.conv7.bias.data.zero_()
        conv_layers += [self.conv7, self.relu7, self.bn7]
        self.conv = nn.Sequential(*conv_layers)

#         # Linear Classifier
#         self.ap = nn.AdaptiveAvgPool2d(output_size=1)
#         self.lin11 = nn.Linear(in_features=256, out_features=128) 
#         self.relu11 = nn.ReLU()
#         self.lin21 = nn.Linear(in_features=128, out_features=64)
#         self.relu21 = nn.ReLU() 
        
#         self.lin12 = nn.Linear(in_features=2, out_features=16)
#         self.relu12 = nn.ReLU()
#         self.lin22 = nn.Linear(in_features=16, out_features=8)
#         self.relu22 = nn.ReLU()
        
#         self.lin3 = nn.Linear(in_features=64+8, out_features=6) 


        #Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.layer_0 = nn.Dropout2d(0.3)

        self.lin11 = nn.Linear(in_features=512, out_features=256) 
        self.relu11 = nn.ReLU()
        self.lin21 = nn.Linear(in_features=256, out_features=64)
        self.relu21 = nn.ReLU() 
        
        self.lin12 = nn.Linear(in_features=2, out_features=16)
        self.relu12 = nn.ReLU()
        self.lin22 = nn.Linear(in_features=16, out_features=8)
        self.relu22 = nn.ReLU()
        
        self.lin3 = nn.Linear(in_features=64+8, out_features=6) 

        #  Normal (정상) / Cancer (후두암) / Cyst_and_Polyp (물혹/폴립) / Nodules (성대결절) 
        #  Functional_dysphonia (기능성 음성질환) / Paralysis (성대 마비)

        # Wrap the Convolutional Blocks
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x, x_meta):
        
        # Run the convolutional blocks
        x = self.conv(x)
        
        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = self.layer_0(x)
        x = x.view(x.shape[0], -1)
        
        # Linear layer
        x      = self.relu11(self.lin11(x))
        x      = self.relu21(self.lin21(x))
        x_meta = self.relu12(self.lin12(x_meta))
        x_meta = self.relu22(self.lin22(x_meta))
    
        x      = torch.cat([x , x_meta], dim=1)
        x      = self.lin3(x)

        # Final output
        return x
    
    
Mel_classifier = Melspectro_classifier()
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
Mel_classifier = Mel_classifier.to(device)
# Check that it is on Cuda
print('\n\n=========== Where the model is located ===========')
next(Mel_classifier.parameters()).device


# In[7]:


for i in Mel_classifier.named_children():
    print(i)


# In[8]:


num_epochs = 20
train_loss_values = np.zeros([num_epochs])

# ----------------------------
# Saver
# ----------------------------
def save_checkpoint(state, is_best, filename='db512_dataaug_dropout_Mel_classifier.pt'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'db512_dataaug_dropout_model_best.pt')

        
# ----------------------------
# Training 
# ----------------------------
def training(model, train_dataset, num_epochs):
    # Loss Function, Optimizer and Scheduler
    best_acc1 = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                  lr_lambda=lambda epoch: 0.95 ** epoch)
    
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
            mels, age, sex, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
            
            #■
            mels = torch.unsqueeze(mels, 1)
            
            # Normalize the mels
            mels_m, mels_s = mels.mean(), mels.std()
            mels = (mels - mels_m) / mels_s
            
            age = age.reshape(-1,1)
            sex = sex.reshape(-1,1)
            
            age_m, age_s = age.mean(), age.std()
            age = (age - age_m) / age_s
            
            sex_m, sex_s = sex.mean(), sex.std()
            sex = (sex - sex_m) / sex_s
                        
#             print('memls type : ', mels.shape)
#             print('age type : ', age.shape)
#             print('sex type : ', sex.shape)

            age_sex = torch.cat([age , sex], dim=1)
 #           print('age_sex type : ', age_sex.shape)

            # Zero the parameter gradients
            optimizer.zero_grad()


            
            # forward + backward + optimize
            outputs = model(mels, age_sex)
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


# In[11]:


x_len = np.arange(len(train_loss_values))
plt.plot( x_len ,train_loss_values, marker = '.',label="Train loss")
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# In[14]:



# It's for testing for training set (Not proper validation, Temporary validation)
tot_test_dataset = torch.utils.data.DataLoader(train_data, batch_size=40, shuffle=True)

def tot_testing(model, tot_test_dataset):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        model.eval() # for not using a dropout

        for data in tot_test_dataset:
            # Get the input features and target labels, and put them on the GPU

            mels, age, sex, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
            #■
            mels = torch.unsqueeze(mels, 1)
            
            # Normalize the mels
            mels_m, mels_s = mels.mean(), mels.std()
            mels = (mels - mels_m) / mels_s
            
            age = age.reshape(-1,1)
            sex = sex.reshape(-1,1)
            
            age_m, age_s = age.mean(), age.std()
            age = (age - age_m) / age_s
            
            sex_m, sex_s = sex.mean(), sex.std()
            sex = (sex - sex_m) / sex_s
                        
#             print('memls type : ', mels.shape)
#             print('age type : ', age.shape)
#             print('sex type : ', sex.shape)

            age_sex = torch.cat([age , sex], dim=1)
            val_pred = model(mels, age_sex)
            val_pred = val_pred.cpu()

            # Get the predicted class with the highest score
            _, prediction = torch.max(val_pred, 1)

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




