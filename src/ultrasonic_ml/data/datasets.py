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

# TODO: just put visualizaiton in here instead of separate file

class morlet_1D_real_dataset(torch.utils.data.Dataset):
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
        self.additional_process_name = ''
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
        self.t = np.arange(0, self.spec_len*self.dt, self.dt)
        self.resonant_frequency = 2.25e-6
        
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
            try: self.coords[i] = np.array([abs(self.data[i]['Z']), abs(self.data[i]['X'])])
            except: pass
            
        self.image_shape = tuple(self.coords.max(axis=0).astype(int) + 1)
        self.processed_max = np.max(np.abs(self.data['processed_'+self.dset_name]))
        self.data['processed_'+self.dset_name] = self.data['processed_'+self.dset_name]/np.max(np.abs(self.data['processed_'+self.dset_name]))   
        self.preprocessed = True

    # TODO: figure out how to make retrieval faster in the future
    def __getitem__(self, idx):
        return idx, self.data[f'processed_{self.dset_name}'+self.additional_process_name][idx]
    
    def __len__(self):
        return len(self.numeric_keys)
    
    def display_dict_tree(self):
        display_dict_tree(self.data)
        
        
class morlet_1D_fft_dataset(morlet_1D_real_dataset):
    def __init__(self, sq3lite_path, dset_name, image_shape=[1,1], crop=None,
                 # ^^super
                 lowpass=200):
        '''
        path: path to the pickle file
        dset_name: name of the dataset, ['voltage_transmission_forward', 'voltage_echo_forward', 'voltage_transmission_reverse', 'voltage_echo_reverse']
        '''
        super().__init__(sq3lite_path, dset_name, image_shape=image_shape, crop=crop)
        self.lowpass = lowpass
        self.lowpass_inds = slice(self.spec_len//2, self.spec_len//2+lowpass)
        self.freq = np.fft.fftshift(np.fft.fftfreq(self.spec_len, self.dt))[self.lowpass_inds]
        self.fft_data()
        self.shape = (len(self.numeric_keys), lowpass)
        self.spec_len = lowpass
        
    def fft_data(self):
        self.data['processed_'+self.dset_name+'_fft'] = np.fft.fftshift(np.fft.fft(self.data['processed_'+self.dset_name]))/max(self.freq)

    def full_data(self,idx):
        return self.data['processed_'+self.dset_name+'_fft'][idx]
    
    def full_freq(self):
        return np.fft.fftshift(np.fft.fftfreq(self.spec_len, self.dt))
    
    def magnitude_spectrum(self, idx):
        return np.abs(self.full_data(idx))[self.lowpass_inds]
    
    def phase_spectrum(self, idx):
        return np.unwrap(np.angle(self.full_data(idx)), discont=np.pi/2)[self.lowpass_inds]
    
    def phase_velocity(self, idx):
        return (self.phase_spectrum(idx) / self.freq)
    
    def group_delay(self, idx):
        return -np.gradient(self.phase_spectrum(idx), self.freq)/(2*np.pi)
    
    # TODO: figure out how to make retrieval faster in the future
    def __getitem__(self, idx):
        return idx, self.data[f'processed_{self.dset_name}'+'_fft'][idx][self.lowpass_inds]
    