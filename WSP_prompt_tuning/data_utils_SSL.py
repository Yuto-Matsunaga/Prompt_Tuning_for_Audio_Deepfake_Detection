import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from random import randrange
import random
import math


___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

def genSpoof_list(dir_meta, is_train=False, is_eval=False, is_dev=False, dataset_name='LA'):

    assert [is_train, is_dev, is_eval].count(True) == 1
    if dataset_name == 'LA':
        if is_train:
            split_name = 'progress'
        elif is_dev:
            split_name = 'hidden_track'
        elif is_eval:
            split_name = 'eval'
    elif dataset_name in ['HABLA', 'VCC_ENG', 'VCC_FIN', 'VCC_MAN', 'VCC_GER', 'In_The_Wild']:
        if is_train:
            split_name = 'train'
        elif is_dev:
            split_name = 'dev'
        elif is_eval:
            split_name = 'test'
    else:
        raise NotImplementedError()
    
    d_meta = dict()
    file_list = list()
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        if dataset_name == 'LA':
            _,key,_,_,_,label,_,split = line.strip().split()
        elif dataset_name in ['HABLA', 'VCC_ENG', 'VCC_FIN', 'VCC_MAN', 'VCC_GER', 'In_The_Wild']:
            _,key,lang,source,method,target,label,split = line.strip().split()
        else:
            raise NotImplementedError()
        if split == split_name:
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
    return d_meta, file_list


def gen_atacklabels(dir_meta, is_train=False, is_eval=False, is_dev=False, dataset_name='LA'):

    assert [is_train, is_dev, is_eval].count(True) == 1
    if dataset_name == 'LA':
        if is_train:
            split_name = 'progress'
        elif is_dev:
            split_name = 'hidden_track'
        elif is_eval:
            split_name = 'eval'
    elif dataset_name in ['HABLA', 'VCC_ENG', 'VCC_FIN', 'VCC_MAN', 'VCC_GER', 'In_The_Wild']:
        if is_train:
            split_name = 'train'
        elif is_dev:
            split_name = 'dev'
        elif is_eval:
            split_name = 'test'
    else:
        raise NotImplementedError()
    
    method_dict = dict()
    file_list = list()
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        if dataset_name == 'LA':
            source,key,_,_,method,label,_,split = line.strip().split()
        elif dataset_name in ['HABLA', 'VCC_ENG', 'VCC_FIN', 'VCC_MAN', 'VCC_GER', 'In_The_Wild']:
            _,key,lang,source,method,target,label,split = line.strip().split()
        else:
            raise NotImplementedError()
        if split == split_name:
            file_list.append(key)
            tmp_label = 1 if label == 'bonafide' else 0
            method_dict[key] = [method, source, tmp_label]
    return method_dict

def genSpoof_list_LA19( dir_meta,is_train=False,is_eval=False,LA21=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    elif (LA21):
        for line in l_meta:
             _,key,_,_,_,label,_,_ = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    else:
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list



def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			

class Dataset_ASVspoof2021_train(Dataset):
    def __init__(self,args,list_IDs, labels, base_dir, n_samples = None):
        '''self.list_IDs	: list of strings (each string: utt key),
            self.labels      : dictionary (key: utt key, value: label integer)'''
            
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo=args["algo"]
        self.args=args
        self.cut=64600 # take ~4 sec audio (64600 samples)
        if n_samples is not None:
            bonafide_sample = math.ceil(n_samples/10)
            spoof_sample= n_samples - bonafide_sample
            filtered_filenames = [filename for filename, label in self.labels.items() if label == 0]
            spoof_samples = random.sample(filtered_filenames, spoof_sample)
            filtered_filenames = [filename for filename, label in self.labels.items() if label == 1]
            bonafide_samples = random.sample(filtered_filenames, bonafide_sample)
            self.list_IDs = random.sample(spoof_samples + bonafide_samples, n_samples)
            self.labels = {k:self.labels[k] for k in self.list_IDs}

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X,fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000) 
        X_pad= pad(X,self.cut)
        x_inp= Tensor(X_pad)
        target = self.labels[utt_id]
        
        return x_inp, torch.tensor(target)
    
    def get_IDlist(self):
        return self.list_IDs
    def get_label(self):
        return self.labels
    
            
class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self,  list_IDs, labels, base_dir, n_samples = None):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.labels = labels
        self.cut=64600 # take ~4 sec audio (64600 samples)
        if n_samples is not None:
            self.list_IDs = random.sample(self.list_IDs, n_samples)

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, utt_id, target

class Dataset_HABLA_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, n_samples = None):
        self.args = args
        self.algo = args["algo"]
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600
        if n_samples is not None:
            # self.list_IDs = random.sample(self.list_IDs, n_samples)
            # self.labels = {k:self.labels[k] for k in self.list_IDs}
            bonafide_sample = math.ceil(n_samples/3)
            spoof_sample= n_samples - bonafide_sample
            filtered_filenames = [filename for filename, label in self.labels.items() if label == 0]
            spoof_samples = random.sample(filtered_filenames, spoof_sample)
            filtered_filenames = [filename for filename, label in self.labels.items() if label == 1]
            bonafide_samples = random.sample(filtered_filenames, bonafide_sample)
            self.list_IDs = random.sample(spoof_samples + bonafide_samples, n_samples)
            self.labels = {k:self.labels[k] for k in self.list_IDs}

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(os.path.join(self.base_dir, 'train_dev', utt_id+'.wav'), sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, torch.tensor(target)
    
    def get_IDlist(self):
        return self.list_IDs
    
class Dataset_HABLA_train_even(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, method_dict, n_samples = None):
        self.args = args
        self.algo = args["algo"]
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600
        if n_samples is not None:
            if n_samples < 50:
                bonafide_sample = math.ceil(n_samples/2)
                spoof_sample= n_samples - bonafide_sample
                filtered_filenames = [filename for filename, label in self.labels.items() if label == 0]
                spoof_samples = random.sample(filtered_filenames, spoof_sample)
                filtered_filenames = [filename for filename, label in self.labels.items() if label == 1]
                bonafide_samples = random.sample(filtered_filenames, bonafide_sample)
                self.list_IDs = random.sample(spoof_samples + bonafide_samples, n_samples)
                self.labels = {k:self.labels[k] for k in self.list_IDs}
            else:
                bonafide_sample = math.ceil(n_samples/3)
                spoof_sample= n_samples - bonafide_sample
                bonafide_methods = {k: v[0] for k, v in method_dict.items() if v[2] == 1}
                spoof_methods = {k: v[0] for k, v in method_dict.items() if v[2] == 0}
                        
                # spoofs     
                group_data = {}
                key_size = 0
                for filename, methods in spoof_methods.items():
                    group_key = tuple(methods)
                    if group_key not in group_data:
                        group_data[group_key] = []
                        key_size = key_size+1
                    group_data[group_key].append(filename)
                
                atack_n_speakers = [(spoof_sample + i) // key_size for i in range(key_size)]
                spoof_samples = list()
                sample_keys = 0
                for _,filenames in group_data.items():
                    selected_file = random.sample(filenames, atack_n_speakers[sample_keys])
                    spoof_samples.extend(selected_file)
                    sample_keys = sample_keys+1
                spoof_samples
                # bonafides 
                group_data = {}
                key_size = 0
                for filename, methods in bonafide_methods.items():
                    group_key = tuple(methods)
                    if group_key not in group_data:
                        group_data[group_key] = []
                        key_size = key_size+1
                    group_data[group_key].append(filename)
                
                n_speakers = [(bonafide_sample + i) // key_size for i in range(key_size)]
                bonafide_samples = list()
                sample_keys = 0
                for _, filenames in group_data.items():
                    selected_file = random.sample(filenames, n_speakers[sample_keys])
                    bonafide_samples.extend(selected_file)
                    sample_keys = sample_keys+1

                self.list_IDs = random.sample(spoof_samples + bonafide_samples, n_samples)
                self.labels = {k:self.labels[k] for k in self.list_IDs}

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(os.path.join(self.base_dir, 'train_dev', utt_id+'.wav'), sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, torch.tensor(target)
    
    def get_IDlist(self):
        return self.list_IDs
    def get_label(self):
        return self.labels
    
class Dataset_HABLA_eval_for_eer(Dataset):
    def __init__(self, args, list_IDs, labels,base_dir, n_samples = None):
        self.args = args
        self.algo = args["algo"]
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.labels = labels
        self.cut = 64600
        if n_samples is not None:
            self.list_IDs = random.sample(self.list_IDs, n_samples)

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(os.path.join(self.base_dir, 'train_dev', utt_id+'.wav'), sr=16000)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, utt_id, target

class Dataset_VCC_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir,method_dict=None, n_samples = None):
        self.args = args
        self.algo = args["algo"]
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600
        if n_samples is not None:
            if n_samples<50:
                bonafide_sample = math.ceil(n_samples/2)
                spoof_sample= n_samples - bonafide_sample
                filtered_filenames = [filename for filename, label in self.labels.items() if label == 0]
                spoof_samples = random.sample(filtered_filenames, spoof_sample)
                filtered_filenames = [filename for filename, label in self.labels.items() if label == 1]
                bonafide_samples = random.sample(filtered_filenames, bonafide_sample)
                self.list_IDs = random.sample(spoof_samples + bonafide_samples, n_samples)
                self.labels = {k:self.labels[k] for k in self.list_IDs}
            else:
                bonafide_sample = math.ceil(n_samples/12)
                spoof_sample= n_samples - bonafide_sample
                bonafide_methods = {k: v[0] for k, v in method_dict.items() if v[2] == 1}
                spoof_methods = {k: v[0] for k, v in method_dict.items() if v[2] == 0}
                        
                # spoofs     
                group_data = {}
                key_size = 0
                for filename, methods in spoof_methods.items():
                    group_key = tuple(methods)
                    if group_key not in group_data:
                        group_data[group_key] = []
                        key_size = key_size+1
                    group_data[group_key].append(filename)
                
                atack_n_speakers = [(spoof_sample + i) // key_size for i in range(key_size)]
                spoof_samples = list()
                sample_keys = 0
                for _,filenames in group_data.items():
                    selected_file = random.sample(filenames, atack_n_speakers[sample_keys])
                    spoof_samples.extend(selected_file)
                    sample_keys = sample_keys+1
                spoof_samples
                # bonafides 
                group_data = {}
                key_size = 0
                for filename, methods in bonafide_methods.items():
                    group_key = tuple(methods)
                    if group_key not in group_data:
                        group_data[group_key] = []
                        key_size = key_size+1
                    group_data[group_key].append(filename)
                
                n_speakers = [(bonafide_sample + i) // key_size for i in range(key_size)]
                bonafide_samples = list()
                sample_keys = 0
                for _, filenames in group_data.items():
                    selected_file = random.sample(filenames, n_speakers[sample_keys])
                    bonafide_samples.extend(selected_file)
                    sample_keys = sample_keys+1

                self.list_IDs = random.sample(spoof_samples + bonafide_samples, n_samples)
                self.labels = {k:self.labels[k] for k in self.list_IDs}

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        # The folder name is eval, but all the data are in eval, and the partition is correct with list_IDs.
        X, fs = librosa.load(os.path.join(self.base_dir, 'eval', utt_id+'.wav'), sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target
    
    def get_IDlist(self):
        return self.list_IDs
    def get_label(self):
        return self.labels

class Dataset_ITW_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, n_samples = None):
        self.args = args
        self.algo = args["algo"]
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600
        if n_samples is not None:
            bonafide_sample = math.ceil(n_samples/1.6)
            spoof_sample= n_samples - bonafide_sample
            filtered_filenames = [filename for filename, label in self.labels.items() if label == 0]
            spoof_samples = random.sample(filtered_filenames, spoof_sample)
            filtered_filenames = [filename for filename, label in self.labels.items() if label == 1]
            bonafide_samples = random.sample(filtered_filenames, bonafide_sample)
            self.list_IDs = random.sample(spoof_samples + bonafide_samples, n_samples)
            self.labels = {k:self.labels[k] for k in self.list_IDs}

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(os.path.join(self.base_dir, utt_id+'.wav'), sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target
    
    def get_IDlist(self):
        return self.list_IDs
    def get_label(self):
        return self.labels

class Dataset_HABLA_VCC_eval(Dataset):
    def __init__(self, list_IDs, labels, base_dir, n_samples = None):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.labels = labels
        self.cut = 64600 # take ~4 sec audio (64600 samples)
        if n_samples is not None:
            self.list_IDs = random.sample(self.list_IDs, n_samples)

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(os.path.join(self.base_dir, 'eval', utt_id+'.wav'), sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, utt_id, target
    
class Dataset_ITW_eval(Dataset):
    def __init__(self, list_IDs,labels,base_dir, n_samples = None):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.labels = labels
        self.cut = 64600 # take ~4 sec audio (64600 samples)
        if n_samples is not None:
            self.list_IDs = random.sample(self.list_IDs, n_samples)

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(os.path.join(self.base_dir, utt_id+'.wav'), sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, utt_id, target

class Dataset_ASVspoof2019_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut=64600 # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+'flac/'+utt_id+'.flac', sr=16000)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id  


#--------------RawBoost data augmentation algorithms---------------------------##



# def process_Rawboost_feature(feature, sr,args,algo):
    
#     # Data process by Convolutive noise (1st algo)
#     if algo==1:

#         feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
#     # Data process by Impulsive noise (2nd algo)
#     elif algo==2:
        
#         feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
#     # Data process by coloured additive noise (3rd algo)
#     elif algo==3:
        
#         feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
#     # Data process by all 3 algo. together in series (1+2+3)
#     elif algo==4:
        
#         feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
#                  args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
#         feature=ISD_additive_noise(feature, args.P, args.g_sd)  
#         feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
#                 args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

#     # Data process by 1st two algo. together in series (1+2)
#     elif algo==5:
        
#         feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
#                  args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
#         feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

#     # Data process by 1st and 3rd algo. together in series (1+3)
#     elif algo==6:  
        
#         feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
#                  args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
#         feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

#     # Data process by 2nd and 3rd algo. together in series (2+3)
#     elif algo==7: 
        
#         feature=ISD_additive_noise(feature, args.P, args.g_sd)
#         feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
#     # Data process by 1st two algo. together in Parallel (1||2)
#     elif algo==8:
        
#         feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
#                  args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
#         feature2=ISD_additive_noise(feature, args.P, args.g_sd)

#         feature_para=feature1+feature2
#         feature=normWav(feature_para,0)  #normalized resultant waveform
 
#     # original data without Rawboost processing           
#     else:
        
#         feature=feature
    
#     return feature
