from scipy.signal import butter, sosfiltfilt
import torch
import re
import sys
sys.path.append('/home/xinqiao/new_mount/gaussian_sampler/ultrasonicTesting')
import pickleJar as pj
import os
from Gaussian_Sampler.utils import display_dict_tree
import numpy as np
from sklearn.preprocessing import MinMaxScaler
class morlet_1D_dataset_real(torch.utils.data.Dataset):
    def __init__(self, sq3lite_path, dset_name, image_shape=[1,1], crops=None):
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
        type_ = self.dset_name.split('_')[-1]
        if type_ == 'forward':
            self.gain_keys = 'gainForward'
            self.gain_offset = 'voltageOffsetForward'#TODO: why don't we take this off too?
        elif type_ == 'reverse':
            self.gain_keys = 'gainReverse'
            self.gain_offset = 'voltageOffsetReverse'
        self.preprocessed = False
        self.crops = crops if crops is not None else [(0, self.data[self.numeric_keys[0]][self.dset_name].shape[-1])]
        self.preprocess_data()
        
        self.spec_len = self.data['processed_'+self.dset_name].shape[-1]
        self.image_shape = image_shape
        self.shape = (image_shape[0]*image_shape[1],len(self.crops),self.spec_len)
        
    def preprocess_data(self):
        assert not self.preprocessed, 'Data has already been preprocessed'
        sos = butter(5, 1000000, btype = 'highpass', analog = False, fs = 500000000, output = 'sos')
        self.data['processed_'+self.dset_name] = np.zeros((len(self.numeric_keys), 
                                                                len(self.crops), 
                                                                self.crops[0][1]-self.crops[0][0]))
        for i in self.numeric_keys:
            for c,crop in enumerate(self.crops):  
                # change signal to pre-amplification voltage
                refUngained = pj.correctVoltageByGain(self.data[i][self.dset_name][crop[0]:crop[1]], 
                                                    self.data[i][self.gain_keys] / 10)
                # apply butterworth filter forward and reverse to pre-amplified signal
                refFil = sosfiltfilt(sos, refUngained)             
                self.data['processed_'+self.dset_name][i,c] = refFil.copy()
         
        self.data['processed_'+self.dset_name] =self.data['processed_'+self.dset_name]/np.max(np.abs(self.data['processed_'+self.dset_name]))   
        self.preprocessed = True

    def __getitem__(self, idx):
        return idx, self.data[f'processed_{self.dset_name}'][idx]
    
    def __len__(self):
        return len(self.numeric_keys)
    
    def display_dict_tree(self):
        display_dict_tree(self.data)