from scipy.signal import butter, sosfiltfilt
import torch
import re
import sys
sys.path.append('/Users/xz498/Desktop/ultrasound project/data analysis/ultrasonicTesting')
import pickleJar as pj
import os
from src.utils import display_dict_tree
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class morlet_1D_dataset_real(torch.utils.data.Dataset):
    def __init__(self, sq3lite_path, dset_name, image_shape=[1,1], crop=None):
        '''
        path: path to the pickle file
        dset_name: name of the dataset, ['voltage_transmission_forward', 'voltage_echo_forward', 'voltage_transmission_reverse', 'voltage_echo_reverse']
        '''
        self.sq3lite_path = sq3lite_path
        pj.sqliteToPickle(self.sq3lite_path)
        # Load the pickle
        self.dataset_path= os.path.splitext(self.sq3lite_path)[0] + '.pickle'
        self.data = pj.loadPickle(self.dataset_path)
        self.numeric_keys = [k for k in self.data.keys() if not isinstance(k, str)]
        self.dset_name = dset_name     
        self.dt = self.data['parameters']['measureTime']/ (self.data['parameters']['samples'] - 1)
        type_ = self.dset_name.split('_')[-1]
        if type_ == 'forward':
            self.gain_keys = 'gainForward'
            self.gain_offset = 'voltageOffsetForward'#TODO: why don't we take this off too?
        elif type_ == 'reverse':
            self.gain_keys = 'gainReverse'
            self.gain_offset = 'voltageOffsetReverse'
        self.preprocessed = False
        self.crop = [0, self.data[self.numeric_keys[0]][self.dset_name].shape[-1]] if crop is None else crop
        self.preprocess_data()
        
        self.spec_len = self.data['processed_'+self.dset_name].shape[-1]
        self.shape = (len(self.numeric_keys),self.crop[1]-self.crop[0])
        
        
    def preprocess_data(self):
        assert not self.preprocessed, 'Data has already been preprocessed'
        print('preprocessing data...')
        sos = butter(5, 1000000, btype = 'highpass', analog = False, fs = 500000000, output = 'sos')
        self.data['processed_'+self.dset_name] = np.zeros((len(self.numeric_keys), self.crop[1]-self.crop[0]))
        self.coords = np.zeros((len(self.numeric_keys), 2)).astype(int)
        for i in tqdm(self.numeric_keys, leave=True, total=len(self.numeric_keys)):
            # change signal to pre-amplification voltage
            refUngained = pj.correctVoltageByGain(self.data[i][self.dset_name][self.crop[0]:self.crop[1]], 
                                                self.data[i][self.gain_keys] / 10)
            # apply butterworth filter forward and reverse to pre-amplified signal
            refFil = sosfiltfilt(sos, refUngained)             
            self.data['processed_'+self.dset_name][i] = refFil.copy()
            self.coords[i] = np.array([abs(self.data[i]['Z']), abs(self.data[i]['X'])])
            
        self.image_shape = tuple(self.coords.max(axis=0).astype(int) + 1)
        self.data['processed_'+self.dset_name] = self.data['processed_'+self.dset_name]/np.max(np.abs(self.data['processed_'+self.dset_name]))   
        self.preprocessed = True

    def preprocess_data_additional(self, func, additional_process_name, **func_args):
        assert self.preprocessed, 'Data has not been preprocessed'
        self.data['processed_'+self.dset_name+'_'+additional_process_name] = func(self.data['processed_'+self.dset_name], **func_args)
        self.additional_process_name = '_'+additional_process_name

    def __getitem__(self, idx):
        return idx, self.data[f'processed_{self.dset_name}'+self.additional_process_name][idx]
    
    def __len__(self):
        return len(self.numeric_keys)
    
    def display_dict_tree(self):
        display_dict_tree(self.data)