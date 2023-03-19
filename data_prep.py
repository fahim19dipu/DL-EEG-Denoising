# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 21:56:18 2022

@author: user
"""

import numpy as np
import os
from scipy import io
import ntpath
import math


def get_rms(records):
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
def load_data(filepath):
    mat = io.loadmat(filepath,struct_as_record=True, squeeze_me=False)
    filename = path_leaf(filepath)
    filename, file_extension = os.path.splitext(filename)
    
    mat=np.array(mat[filename])                 # Signal value (time domain)

    return mat

def random_signal(signal,combin_num):
    # Random disturb and augment signal
    random_result=[]

    for i in range(combin_num):
        random_num = np.random.permutation(signal.shape[0])
        shuffled_dataset = signal[random_num, :]
        shuffled_dataset = shuffled_dataset.reshape(signal.shape[0],signal.shape[1])
        random_result.append(shuffled_dataset)
    
    random_result  = np.array(random_result)

    return  random_result
# creating the dataset variable

                                            # Loading data from matlab files
def prepare_data(dirname):
    EOG = load_data(dirname+'/Dataset/EOG_all_epochs.mat') 
    EEG= load_data(dirname+'/Dataset/EEG_all_epochs.mat') 
    
    np.random.shuffle(EEG)
    np.random.shuffle(EOG)
    
    print(EEG.shape)
    
    print(EOG.shape)
    EEG = EEG[0:EOG.shape[0]]                   # taking Eqal number of EEG and EOG
    timepoint = EOG.shape[1]                            # trial size
                                                
    train_num = round(.8 * EEG.shape[0])        # taking 80% of trial for training
    #val_num = round(.2 * EEG.shape[0])          # taking 20% of trial for validation
    test_num = round(.2 * EEG.shape[0])         # taking rest 20% of trial for testing

    train_eeg = EEG[0 : train_num, :]
    train_eog = EOG[0 : train_num, :]
    
    # val_eeg = EEG[train_num : train_num + val_num, :]
    # val_eog = EOG[train_num : train_num + val_num, :]
    
    
    test_eeg = EEG[train_num :  :]
    test_eog = EOG[train_num :  :]
    
    print(train_eeg.shape)
    #print(val_eeg.shape)   
    print(test_eeg.shape)                   
    combin_num = 10
    EEG_train = random_signal(train_eeg, combin_num).reshape(combin_num * train_eeg.shape[0], timepoint)
    ### trating EOG as noise
    NOISE_train = random_signal(train_eog, combin_num).reshape(combin_num * train_eog.shape[0], timepoint) 
    
    print(EEG_train.shape)
    print(NOISE_train.shape)
    
    SNR_train_dB = np.random.uniform(-7, 2, (EEG_train.shape[0]))
    print(SNR_train_dB.shape)
    SNR_train = 10 ** (0.1 * (SNR_train_dB))
    
    ##############################              combin eeg and noise for training set 
    noiseEEG_train=[]
    NOISE_train_adjust=[]
    for i in range (EEG_train.shape[0]):
        eeg=EEG_train[i].reshape(EEG_train.shape[1])
        noise=NOISE_train[i].reshape(NOISE_train.shape[1])
    
        coe=get_rms(eeg)/(get_rms(noise)*SNR_train[i])
        noise = noise*coe
        neeg = noise+eeg
    
        NOISE_train_adjust.append(noise)
        noiseEEG_train.append(neeg)
    
    noiseEEG_train=np.array(noiseEEG_train)
    NOISE_train_adjust=np.array(NOISE_train_adjust)    
    
    # variance for noisy EEG
    EEG_train_end_standard = []
    noiseEEG_train_end_standard = []
    
    for i in range(noiseEEG_train.shape[0]):
        # Each epochs divided by the standard deviation
        eeg_train_all_std = EEG_train[i] / np.std(noiseEEG_train[i])
        EEG_train_end_standard.append(eeg_train_all_std)
    
        noiseeeg_train_end_standard = noiseEEG_train[i] / np.std(noiseEEG_train[i])
        noiseEEG_train_end_standard.append(noiseeeg_train_end_standard)
    
    noiseEEG_train_end_standard = np.array(noiseEEG_train_end_standard)
    EEG_train_end_standard = np.array(EEG_train_end_standard)
    print('training data prepared', noiseEEG_train_end_standard.shape, EEG_train_end_standard.shape )
    
    #################################  simulate noise signal of test  ##############################
    
    SNR_test_dB = np.linspace(-7.0, 2.0, num=(10))
    SNR_test = 10 ** (0.1 * (SNR_test_dB))
    
    eeg_test = np.array(test_eeg)
    noise_test = np.array(test_eog)
    
    # combin eeg and noise for test set 
    EEG_test = []
    noise_EEG_test = []
    for i in range(10):
        
        noise_eeg_test = []
        for j in range(eeg_test.shape[0]):
            eeg = eeg_test[j]
            noise = noise_test[j]
            
            coe = get_rms(eeg) / (get_rms(noise) * SNR_test[i])
            noise = noise * coe
            neeg = noise + eeg
            
            noise_eeg_test.append(neeg)
        
        EEG_test.extend(eeg_test)
        noise_EEG_test.extend(noise_eeg_test)
    
    
    noise_EEG_test = np.array(noise_EEG_test)
    EEG_test = np.array(EEG_test)
    
    
    # std for noisy EEG
    EEG_test_end_standard = []
    noiseEEG_test_end_standard = []
    std_VALUE = []
    for i in range(noise_EEG_test.shape[0]):
        
        # store std value to restore EEG signal
        std_value = np.std(noise_EEG_test[i])
        std_VALUE.append(std_value)
    
        # Each epochs of eeg and neeg was divide by the standard deviation
        eeg_test_all_std = EEG_test[i] / std_value
        EEG_test_end_standard.append(eeg_test_all_std)
    
        noiseeeg_test_end_standard = noise_EEG_test[i] / std_value
        noiseEEG_test_end_standard.append(noiseeeg_test_end_standard)
    
    std_VALUE = np.array(std_VALUE)
    noiseEEG_test_end_standard = np.array(noiseEEG_test_end_standard)
    EEG_test_end_standard = np.array(EEG_test_end_standard)
    print('test data prepared, test data shape: ', noiseEEG_test_end_standard.shape, EEG_test_end_standard.shape)
    return noiseEEG_train_end_standard, EEG_train_end_standard,  noiseEEG_test_end_standard, EEG_test_end_standard


#noiseEEG_val_end_standard, EEG_val_end_standard,