import sys
import os 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Disable HDF5 file locking, TODO: check if this is risky

import json
from typing import Any, Iterator    
import h5py
import pprint
from tqdm import tqdm

import math
import numpy as np  
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks, correlate, correlation_lags
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt

from ..utils import NumpyEncoder
from .sqlite_datasets import AcousticSQlite

import pywt

# TODO: add method or attribute for tracking indices when filling raw data or tracking sqlite file length
# TODO: AcousticDatabase class. Base class. This does not have full functionality
class AcousticDatabase:
    '''Base class for acoustic scan databases, providing common functionality for handling HDF5 and SQLite data.
    analytical frequency domain study https://www.sciencedirect.com/science/article/pii/S2352152X22006016
    '''
    
    def __init__(self, parent_folder: str, h5_name: str, sqlite_path: str | None = None):
        '''Initialize the AcousticDatabase.
        
        Parameters
        ----------
        parent_folder : str
            The folder containing the SQlite file.
        h5_name : str
            The path to the HDF5 file.
        sqlite_path : str | None, optional
            The path to the SQLite database file, by default None.
        '''
        self.parent_folder = parent_folder
        self.h5_name = f'{self.parent_folder}/{h5_name}'
        self.sqlite_database = AcousticSQlite(sqlite_path) if sqlite_path else None
        self.shape = None
        self.len = None
        self.parameters = {}
        
        try: 
            self.load_parameters()
        except KeyError as e: 
            print(e)
            print('Run parameter setup functions')
            pass
    
    __len__ = lambda self: self.len if self.len else len(self.sqlite_database) if self.sqlite_database else 0
    
    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
        
    def open_h5(self, mode='a'): 
        '''Open the HDF5 file.
        
        Parameters
        ----------
        mode : str, optional
            The mode to open the file in, by default 'a'.

        Returns
        -------
        h5py.File
            The open h5py file.
        '''
        return h5py.File(self.h5_name, mode)
    
    def get_check_group(self, h5, group_name: str):
        '''Check if the waveform group exists and return the waveforms and hilbert data.

        Parameters
        ----------
        h5 : h5py.File
            The open h5py file.
        group_name : str
            The name of the group to check.

        Returns
        -------
        tuple
            A tuple containing the waveforms and hilbert data.
        '''
        try: return h5[group_name]['waveforms'], h5[group_name]['hilbert']
        except KeyError as e: print(e, f'Run setup for {group_name} first')

    def _print_schema(self, h5, tab: str = ''):
        '''Recursively print the schema of the HDF5 file.
        
        Parameters
        ----------
        h5 : h5py.File
            The open h5py file.
        tab : str, optional
            The tab spacing for the schema print, by default ''.
        '''
        for name, obj in h5.items():
            size_kb = 0.0
            if hasattr(obj, 'id') and hasattr(obj.id, 'get_storage_size'):
                size_kb = obj.id.get_storage_size() / (1024 * 1024)  # Convert bytes to MB
                print(f"{tab}|--{obj} [{size_kb:.2f} MB]")
            else:
                print(f"{tab}|--{obj}")
            if isinstance(obj, h5py.Group):
                self._print_schema(obj, tab + '|  ')
                
    def print_schema(self): 
        '''Print the schema of the HDF5 file.'''
        with self.open_h5() as h5:
            print(f"{h5}")
            self._print_schema(h5)
    
    def print_sqlite_schema(self): 
        '''Print the schema of the SQLite database.'''
        self.sqlite_database.print_schema()
    
    def sqlite_to_h5(self): 
        """Hook to convert the SQLite database to an HDF5 file.
        This method sets up the parameters, raw dataset, and indexing parameters. """
        raise NotImplementedError # specific to the child class

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------
            
    def get_sqlite_parameters(self): 
        # TODO: finish filling these out. Use descriptions from Sam's UltrasonicTesting/runUltrasonicExperiment.py, in the sweep branch.
        # TODO: Make template a part of docs page. 
        '''Get the parameters from the SQLite database and return them as a dictionary with units and descriptions.'''
        # with units and descriptions
        parameters = { # TODO: redo filling multiplexer measurments with scanned/changing offsets and gains.
            'experiment': {
                'value': self.sqlite_database.parameters['experiment'],
                'description': '''options: 
                    'move': move the transducers 
                    'single pulse': perform a single test pulse
                    'repeat pulse': repeat a pulse at a single location for a given time and frequency
                    'single scan': perform a single 2D scan/ 'multi scan': repeat a 2D scan with a set frequency
                    'sweep': repeat pulse changing through parameters given in sweepParameters ''' },
            'collectionMode': {
                'value': self.sqlite_database.parameters['collectionMode'],
                'description': '''options:
                    'echo': collect the echo signal from the transducer
                    'transmission': collect the transmitted signal from the transducer
                    'both': collect both echo and transmitted signals''' },
            'pulserType': {
                'value': self.sqlite_database.parameters['pulserType'],
                'description': '''options:
                    'standard': single wave CompactPulser. 
                    'tone burst': USBUT350 tone burst pulser'''},
            'transducerFrequency': {
                'value': self.sqlite_database.parameters['transducerFrequency'], 
                'unit': 'MHz', 
                'description': 'Transducer center frequency'}, 
            'measureTime': {
                'value': self.sqlite_database.parameters['measureTime'], 
                'unit': 'us', 
                'description': 'Approx measurement time, in us. Note this can be changed by the picoscope time interval based on samples. Changes to the measureTime will be printed in the console when the script is run'},
            'measureDelay': {
                'value': self.sqlite_database.parameters['measureDelay'], 
                'unit': 'us', 
                'description': 'Approx delay after trigger to start measuring, in us'},
            'voltageRange': {
                'value': self.sqlite_database.parameters['voltageRange'], 
                'unit': 'V', 
                'description': 'Picoscope voltage range for transmission transducer in V. Note this is the total range: 1 V = [-0.5 V, 0.5 V]'},
            'waves': {
                'value': self.sqlite_database.parameters['waves'],
                'description': 'Number of waves to collect and average'},
            'samples': {
                'value': self.sqlite_database.parameters['samples'],
                'description': 'Number of data points per wave'},
            'dt': {
                'value': self.sqlite_database.parameters['dt'], 
                'unit': 's', 
                'description': 'Sampling interval'},
            'f_s': {
                'value': self.sqlite_database.parameters['f_s'], 
                'unit': 'Hz', 
                'description': 'Sampling frequency'},
            'pulserPort': {
                'value': self.sqlite_database.parameters['pulserPort'],
                'description': 'COM port for the pulser'},
            'multiplexer': {
                'value': self.sqlite_database.parameters['multiplexer'],
                'description': 'True/False: whether a multiplexer is used'},
            'collectionDirection': {
                'value': self.sqlite_database.parameters['collectionDirection'],
                'description': 'forward/reverse/both'},
            'picoModule': {
                'value': self.sqlite_database.parameters['picoModule'],
                'description': ''},
            'pulseModule': {
                'value': self.sqlite_database.parameters['pulseModule'],
                'description': ''},
        }
        
        ### experiment types ###
        # scan specific parameters
        if 'scan' in parameters['experiment']['value']:
            for k in  ['primaryAxis','secondaryAxis']: 
                parameters[k] = { 'value': self.sqlite_database.parameters[k]}
            for k in ['primaryAxisRange', 'primaryAxisStep', 'secondaryAxisRange', 'secondaryAxisStep']: 
                parameters[k] = {
                    'value': self.sqlite_database.parameters[k], 
                    'unit': 'mm'}  
            parameters['scannerPort'] = {
                'value': self.sqlite_database.parameters['scannerPort'],
                'description': 'Scanner port name' }
            
        # repeat measurement specific parameters
        if 'multi' in parameters['experiment']['value'] or 'repeat' in parameters['experiment']['value']:
            parameters['scanInterval'] =  {
                'value': self.sqlite_database.parameters['scanInterval'], 
                'unit': 's',
                'description': 'For multi scans: Minimum time between start of scans'}
            parameters['numberOfScans'] = {
                'value': self.sqlite_database.parameters['numberOfScans'],
                'description': 'For multi scans: Number of times to run the scan'}
            parameters['pulseInterval'] = {
                'value': self.sqlite_database.parameters['pulseInterval'],
                'unit': 's',
                'description': 'For repeat pulse: Minimum time between pulse collection, in seconds' }
            parameters['experimentTime'] = {
                'value': self.sqlite_database.parameters['experimentTime'],
                'unit': 's',
                'description': 'For repeat pulse: Time to collect data, in seconds' }
        # sweep specific parameters
        if 'sweep' in parameters['experiment']['value']:
            parameters['sweepParameters'] = {
                'value': self.sqlite_database.parameters['sweepParameters'],
                'description': 'For sweep: Parameters to sweep through, in the format of a dictionary. Example: {"transducerFrequency": [1, 2, 3], "measureTime": [10, 20]}'} 
        
        ### collection mode ###
        if parameters['collectionMode']['value'] == 'transmission' or parameters['collectionMode']['value'] == 'both': 
            print('autoRange')
            parameters['autoRange'] = {
                'value': self.sqlite_database.parameters['autoRange'], 
                'description': 'Set the oscilloscope to rerun measurements where the voltage range of the wave has significantly changed. This enables the tightest possible voltageRange to be used, increasing the accuracy of low intensity signals without cutting off high intensity ones. For transmission measurements, this changes the voltage range on the oscilloscope'}
            print('Auto range is', parameters['autoRange']['value'])
            if parameters['autoRange']['value']: # TODO: check
                if parameters['collectionDirection']['value'] == 'forward' or parameters['collectionDirection']['value'] == 'both': 
                    parameters['voltageOffsetForward'] = {
                        'value': self.sqlite_database.parameters['voltageOffsetForward'], 
                        'unit': 'V'}                
                if parameters['collectionDirection']['value'] == 'reverse' or parameters['collectionDirection']['value'] == 'both': 
                    parameters['voltageOffsetReverse'] = {
                        'value': self.sqlite_database.parameters['voltageOffsetReverse'], 
                        'unit': 'V'}
        elif parameters['collectionMode']['value'] == 'echo' or parameters['collectionMode']['value'] == 'both': 
            parameters['autoRangeEcho'] = {
                'value': self.sqlite_database.parameters['autoRangeEcho'], 
                'description': 'Set the oscilloscope to rerun measurements where the voltage range of the wave has significantly changed. This enables the tightest possible voltageRange to be used, increasing the accuracy of low intensity signals without cutting off high intensity ones. For echo measurements, this can add significant overhead (2-3x increase in collection time) for each waveform where the range changes'}
            if parameters['autoRangeEcho']['value']: 
                if parameters['collectionDirection']['value'] == 'forward' or parameters['collectionDirection']['value'] == 'both': 
                    parameters['voltageOffsetForward'] = {
                        'value': self.sqlite_database.parameters['voltageOffsetForward'], 
                        'unit': 'V'}                
                if parameters['collectionDirection']['value'] == 'reverse' or parameters['collectionDirection']['value'] == 'both': 
                    parameters['voltageOffsetReverse'] = {
                        'value': self.sqlite_database.parameters['voltageOffsetReverse'], 
                        'unit': 'V'}
        
        ### pulser type ###
        if parameters['pulserType']['value'] == 'tone burst': 
            parameters['halfCycles'] = {
                'value': self.sqlite_database.parameters['halfCycles'], 
                'description': 'Number of half-cycles in a tone burst pulse. Minimum 1, maximum 32'}

        if parameters['multiplexer']['value']:
            for k in  ['multiplexerPort','rfSwitch', 't0PulseSwitch', 't0ReceiveSwitch', 't1PulseSwitch', 't1ReceiveSwitch']:
                parameters[k] = self.sqlite_database.parameters[k]
        
        ####################################################
        # To format in the parameters table by index
        ####################################################
        # if parameters['autoRange']['value']: pass
        # if parameters['autoRangeEcho']['value']: pass
        
        ####################################################
        # for plotting
        ####################################################        
        parameters['time_array'] = self.sqlite_database.fetch_time()
        parameters['waveform_labels'] = self.sqlite_database.waveform_columns
        
        return parameters
     
    def print_parameters(self):
        '''Pretty-print the parameters dictionary with a threshold for large arrays.'''
        with np.printoptions(threshold=10, edgeitems=2,):
            # Adjust 'indent' to change tab sizing if needed
            pprint.pprint(self.parameters, indent=4, compact=True)
                     
    def save_parameters(self):
        '''Save the parameters dictionary to HDF5 file as a JSON string. Create dataset "config_metadata" in the root of the HDF5 file if it does not exist.'''
        with self.open_h5() as h5:
            json_str = json.dumps(self.parameters, cls=NumpyEncoder) # Convert dictionary to a JSON string
            dt = h5py.special_dtype(vlen=str) # save as special h5 type
            try: del h5['config_metadata']  
            except: pass
            h5.require_dataset('config_metadata', (), data=json_str, dtype=dt)

    def load_parameters(self): 
        '''Load the parameters dictionary from HDF5 file. Read dataset "config_metadata" in the root of the HDF5 file and convert it from a JSON string to a dictionary.'''
        print('Loading h5 parameters...')
        with self.open_h5() as h5:
            loaded_json = h5['config_metadata'][()]
            self.parameters = json.loads(loaded_json)
            self.parameters['time_array'] = np.array(self.parameters['time_array'])
            self.shape = tuple(self.parameters['shape'])
            self.len = self.parameters['len']
        print('\tFinished.')

    def setup_parameters(self): 
        """Hook to setup the parameters for the dataset. Gets the parameters from the SQLite database and calculates the shape and length of the dataset. Saves the parameters to the HDF5 file."""
        raise NotImplementedError # specific to the child class. must also set time_array, shape, and len

    # -------------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------------
    
    def setup_indexed_parameters(self, keys, shapes):
        '''Creates the 'indexed_parameters' group in the HDF5 file and sets up datasets for each parameter with the appropriate shape and data type.
        
        Parameters
        ----------
        keys : list[str]
            List of parameter names to write.
        values : list[np.ndarray]
            List of parameter values to write.
        shapes : list[tuple]
            List of shapes for each parameter.
        '''
        for k, s in zip(keys, shapes):
            with self.open_h5() as h5:
                grp = h5.require_group('indexed_parameters')
                grp.require_dataset(k, shape=s, dtype='f4', track_order=True)
    
    def fill_indexed_parameter(self, keys, values, shapes, idxs=None): # TODO: better and more scalable way to do keys and values?
        '''Fill parameter datasets with values.
        
        Parameters
        ----------
        keys : list[str]
            Indexing parameter datasets to write in.
        values : list[np.ndarray]
            List of parameter values to write.
        shapes : list[tuple]
            List of shapes for each parameter.
        idxs : list[Index] optional
            List of indices to write at.
        '''            
        with self.open_h5() as h5: 
            if not idxs:
                for k, v, s in tqdm(zip(keys, values, shapes)):
                    h5['indexed_parameters'][k][...] = np.array(v).reshape(s)
            else: # TODO: fix in future for more than 1D indexing. This is a temporary fix for 1D indexing.
                for i, idx in enumerate(idxs):
                    h5['indexed_parameters'][k][idx] = v[i]
       
    def write_indexed_parameters(self):
        """Hook to write indexed parameters to the HDF5 file. Calls setup and fill functions with specific indexer. This method should be implemented in the child class."""
        raise NotImplementedError # specific to the child class   
           
    def idx_sqlite_data_generator(self): 
        '''Hook to generate indexed data from SQLite database. To be used for writing raw data. This method is specific to the shape of the experiment and should be implemented in the child class.'''
        raise NotImplementedError # specific to the child class
    
    # fill ___ with some combination of whatever axis are in the dataset (w_x_z, x_z, w_y, etc.)
    def ___generator(self): 
        '''Hook to generate data for filling the raw dataset. This method is specific to the shape of the experiment and should be implemented in the child class.'''
        raise NotImplementedError # specific to the child class
    
    # -------------------------------------------------------------------------
    # Raw data
    # -------------------------------------------------------------------------
        
    def setup_raw_dataset(self):
        '''Creates the 'raw_data' group in the HDF5 file and sets up the 'waveforms' and 'hilbert' datasets with the appropriate shape and data type. 
        It also saves the time array as an attribute of the 'waveforms' dataset.
        '''
        with self.open_h5() as h5:
            grp = h5.require_group('raw_data')
            waveforms_dset = grp.require_dataset('waveforms', shape=self.shape, dtype='f4', track_order=True)
            hilbert_dset = grp.require_dataset('hilbert', shape=self.shape, dtype='c8', track_order=True)
            waveforms_dset.attrs['time_array'] = self.parameters['time_array']
            hilbert_dset.attrs['time_array'] = self.parameters['time_array']
            
    def fill_raw_dataset(self, idx_data : Iterator[tuple[tuple, np.ndarray]]):
        '''Fill a series of specific locations in the raw data dataset.
        
        Parameters
        ----------
        idx_data : Iterator[tuple[tuple, np.ndarray]]
            An iterator of tuples, where each tuple contains an index (tuple) and the corresponding data (np.ndarray) to fill at that index.
            The index should fill axis until the last dimension, and the data should be a 1D array of shape (time_length,).
        '''
        with self.open_h5() as h5: # TODO: calculate the hilbert window and maxes. Look at preprocessed.
            waveform_dset, hilbert_dset = self.get_check_group(h5, 'raw_data')
            
            for idx, data in tqdm(idx_data, desc="Filling raw data and hilbert", total=math.prod(self.shape[:-1])):
                waveform_dset[idx] = data
                hilbert_dset[idx] = self._hilbert(data)
    
    def write_preprocessed_data(self, **kwargs):
        """Hook to write preprocessed data to the HDF5 file. Calls setup and fill functions with specific indexer. This method should be implemented in the child class."""
        raise NotImplementedError # specific to the child class

    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _absolute_max(waveform): 
        '''Calculate the absolute maximum of a waveform.
        
        Parameters
        ----------
        waveform : np.ndarray
            The input waveform. Should be 1D array of shape (time_length,).

        Returns
        -------
        float
            The absolute maximum of the waveform.
        '''
        return float(np.max(np.abs(waveform)))

    @staticmethod
    def _hilbert(waveform): 
        '''Calculate the hilbert envelope of a waveform.
        
        Parameters
        ----------
        waveform : np.ndarray
            The input waveform. Should be array of shape (...,time_length).

        Returns
        -------
        np.ndarray
            The hilbert envelope of the waveform.
        '''
        return hilbert(waveform)

    @staticmethod
    def _undo_gain(waveform, gain, offset): 
        '''Undo the gain of a waveform.
        
        Parameters
        ----------
        waveform : np.ndarray
            The input waveform. Should be 1D array of shape (time_length,).
        gain : float
            The gain to undo.
        offset : float
            The offset to undo.

        Returns
        -------
        np.ndarray
            The waveform with the gain undone.
        '''
        return (waveform - offset) / gain
    
    def _build_filter_sos(self, lower_fs_coeff: float, upper_fs_coeff: float, filter_order: int) -> Any:
        """Build Butterworth bandpass filter coefficients.

        Parameters
        ----------
        lower_fs_coeff : float
            Highpass filter lower limit of lower_fs_coeff*f_s.
        upper_fs_coeff : float
            Lowpass filter upper limit of upper_fs_coeff*f_s.
        filter_order : int
            The order of the filter.

        Returns
        -------
        Any
            The second-order sections of the filter.
        """
        if self.parameters['f_s']['value'] is None: raise ValueError("Sampling frequency is unavailable")
        f_s = self.parameters['f_s']['value']
        return butter(filter_order, [f_s * lower_fs_coeff, f_s * upper_fs_coeff], btype="bandpass", fs=f_s, output="sos")

    def setup_preprocessed_data(self, apply_ungain: bool = False, apply_filter: bool = False, lower_fs_coeff: float = 1/250, upper_fs_coeff: float = 1/5, filter_order: int = 3):
        '''Setup preprocessed group, dataset, and attributes
        
        Parameters
        ----------
        apply_ungain : bool
            Whether to apply ungain to the data.
        apply_filter : bool
            Whether to apply a bandpass filter to the data.
        lower_fs_coeff : float
            Highpass filter lower limit of lower_fs_coeff*f_s.
        upper_fs_coeff : float
            Lowpass filter upper limit of upper_fs_coeff*f_s.
        filter_order : int
            The order of the filter.
        '''
        metadata = {}
        with self.open_h5() as h5:
            grp = h5.require_group('preprocessed_data')
            waveform_dset = grp.require_dataset('waveforms', shape=self.shape, dtype='f4', track_order=True)
            hilbert_dset = grp.require_dataset('hilbert', shape=self.shape, dtype='c8', track_order=True)
            
            if apply_ungain: pass # TODO: later, get gain columns and indices/values from paramters collection mode
            if apply_filter: 
                self.sos = self._build_filter_sos(lower_fs_coeff, upper_fs_coeff, filter_order) if apply_filter else None
                metadata.update({"lower_fs_coeff": lower_fs_coeff, "upper_fs_coeff": upper_fs_coeff, "filter_order": filter_order})
            
            waveform_dset.attrs.update(metadata)
            waveform_dset.attrs['time_array'] = self.parameters['time_array']
            hilbert_dset.attrs['time_array'] = self.parameters['time_array']
    
    def fill_preprocessed_data(self, idxs: Iterator[tuple]) -> None:   
        '''Fill a series of specific locations in the raw data dataset.
                
        Parameters
        ----------
        idxs : Iterator[tuple]
            An iterator of tuples, where each tuple contains an index.
            The index should fill axis until the last dimension. Data will be retrieved from the raw_data group
        '''
        with self.open_h5() as h5:
            waveform_dset, hilbert_dset = self.get_check_group(h5, 'preprocessed_data')
            
            for idx in tqdm(idxs, desc="Writing preprocessed data", total=math.prod(self.shape[:-1])):
                data = h5['raw_data']['waveforms'][*idx]
                # if apply_ungain: waveform = self._undo_gain(waveform, gain, offset)
                try: data = sosfiltfilt(self.sos, data)
                except: continue

                waveform_dset[*idx] = data
                hilbert_dset[*idx] = self._hilbert(data)

    def write_preprocessed_data(self):
        """Write preprocessed data to the HDF5 file. Calls setup and fill functions with specific indexer."""
        raise NotImplementedError
    
    # -------------------------------------------------------------------------
    # Alignment - TODO: is it better to implement as phase shift in fft instead?
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _calculate_rollback(waveform):
        '''
        Calculate the average rollback value for each of the mean waveforms from forward and reverse directions
        
        Parameters:
        -----------
        waveform: np.ndarray
            The waveform data with the collection direction in the first axis and the time in the last axis
            
        Returns:
        --------
        np.ndarray
            The mean rollback values for each waveform profile.
        '''
        mean_profile = waveform.mean(axis=tuple(i for i in range(1,len(waveform.shape)-1)))
        peak_distances = np.array([ find_peaks(profile, height=0.25*profile.max())[0][0] for profile in mean_profile ])
        rollback = peak_distances-min(peak_distances)
        return rollback
    
    def setup_first_break_aligned_data(self, from_group='preprocessed_data'): # TODO: raise error and handle in write function instead if no need to write
        '''
        Set up the group storing waveforms aligned by first breakfor the dataset.
        
        Parameters
        ----------
        from_group : str
            The name of the group from which to read the waveforms to align.
        '''
        if len(self.parameters['waveform_labels'])<2: print('Only one waveform. No need to align peaks'); return 
        if self.parameters['collectionDirection']['value'] != 'both': print('Collection mode is one direction. No need to align peaks'); return
        
        with self.open_h5() as h5:
            self.get_check_group(h5, from_group)
    
            rollbacks = self._calculate_rollback(np.abs(h5[from_group]['hilbert'][:]))
            max_ = rollbacks.max()
            grp = h5.require_group('first_break_aligned')
            
            new_shape = self.shape[:-1] + (self.shape[-1]-max_,)
            dset = grp.require_dataset('waveforms', shape=new_shape, dtype='f4', track_order=True)
            dset.attrs['from_group'] = str(h5[from_group])
            dset.attrs['rollbacks'] = rollbacks
            dset.attrs['time_array'] = self.parameters['time_array'][:-max_]
            
            hilbert = grp.require_dataset('hilbert', shape=new_shape, dtype='c8', track_order=True)
            hilbert.attrs['from_group'] = str(h5[from_group])
            hilbert.attrs['rollbacks'] = rollbacks
            hilbert.attrs['time_array'] = self.parameters['time_array'][:-max_]

    def fill_first_break_aligned_data(self, idxs: Iterator[tuple], from_group='preprocessed_data') -> None:
        """Align the first break of the waveforms in the dataset and write to new dataset

        Parameters
        ----------
        idxs : Iterator[tuple]
            An iterator of tuples containing indices to span all but first and last axis.
        from_group : str
            The name of the group from which to read the waveforms to align.
        """          
        with self.open_h5() as h5:
            waveform_dset_unrolled, hilbert_dset_unrolled = self.get_check_group(h5, from_group)
            waveform_dset, hilbert_dset = self.get_check_group(h5, 'first_break_aligned')
            new_shape = hilbert_dset.shape
            rollbacks = waveform_dset.attrs['rollbacks']

            for idx in tqdm(idxs, desc="Aligning first break", total=math.prod(self.shape[:-1])):
                    waveform_dset[*idx] = waveform_dset_unrolled[*idx,rollbacks[idx[0]]:new_shape[-1]+rollbacks[idx[0]]]
                    hilbert_dset[*idx] = hilbert_dset_unrolled[*idx,rollbacks[idx[0]]:new_shape[-1]+rollbacks[idx[0]]]
    
    # -------------------------------------------------------------------------
    # Basic Analysis
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _tof_idx(hilbert, threshold_fraction=0.1, crop=None):
        '''
        Calculate the time of flight based on the hilbert envelope of the waveform and a threshold fraction of the max.
        
        Parameters:
        -----------
        hilbert: np.ndarray
            The 1D analytic signal (Hilbert transform) of the waveform.
        threshold_fraction: float
            The fraction of the maximum value to use as the threshold for detecting the peak.

        Returns:
        --------
        np.ndarray
            The time of flight for each waveform.
        '''
        if crop is not None: hilbert = hilbert[crop[0]:crop[1]]
        hilbert = np.abs(hilbert)
        threshold_ = hilbert.max()*threshold_fraction
        return int(np.argmax(hilbert > threshold_))
      
    def setup_tof_data(self, from_group='preprocessed_data', threshold_fraction=0.1, crop=None):
        '''
        Set up the group storing waveforms aligned by first breakfor the dataset.
        
        Parameters
        ----------
        from_group : str
            The name of the group from which to read the waveforms to align.
        threshold : float
            The fraction of the maximum value to use as the threshold for detecting the peak.
        crop : tuple
            The start and end indices for cropping the hilbert signal before calculating the time of flight.
        '''
        
        with self.open_h5() as h5:
            _,hilbert = self.get_check_group(h5, from_group)
            dset = h5[from_group].require_dataset('tof', shape=hilbert.shape[:-1], dtype='f4', track_order=True)
            dset.attrs['threshold_fraction'] = threshold_fraction
            dset.attrs['from_group'] = from_group
            if crop: dset.attrs['crop'] = crop
            
    def fill_tof_data(self, idxs: Iterator[tuple], from_group='preprocessed_data', crop=None) -> None:
        """Align the first break of the waveforms in the dataset and write to new dataset

        Parameters
        ----------
        idxs : Iterator[tuple]
            An iterator of tuples containing indices to span all but first and last axis.
        from_group : str
            The name of the group from which to read the waveforms to align.
        """          
        with self.open_h5() as h5:
            threshold_fraction = h5[from_group]['tof'].attrs['threshold_fraction']
            hilbert = h5[from_group]['hilbert']
            time_array = h5[from_group]['waveforms'].attrs['time_array']
            tof_dset = h5[from_group]['tof']
            
            for idx in tqdm(idxs, desc="Calculating ToF", total=math.prod(self.shape[:-1])):
                tof_idx = self._tof_idx(hilbert[*idx], threshold_fraction=threshold_fraction, crop=crop)
                tof_dset[*idx] = time_array[tof_idx]
    
    def write_tof_data(self):
        """Write time of flight data to the HDF5 file. Calls setup and fill functions with specific indexer."""
        raise NotImplementedError
           
    @staticmethod
    def _amplitude(hilbert, crop=None):
            '''
            Calculate the amplitude of the hilbert envelope of the waveform. Optionally crop the hilbert signal before calculating the amplitude.
            
            Parameters:
            -----------
            hilbert: np.ndarray
                The 1D analytic signal (Hilbert transform) of the waveform.
            crop: tuple
                The start and end indices for cropping the hilbert signal.
    
            Returns:
            --------
            np.ndarray
                The amplitude of the hilbert envelope for each waveform.
            '''
            if crop is not None: hilbert = hilbert[crop[0]:crop[1]]
            hilbert = np.abs(hilbert)
            return (hilbert.max())
             
    def setup_amplitude_data(self, from_group='preprocessed_data', crop=None):
        '''
        Set up the group storing waveforms aligned by first breakfor the dataset.
        
        Parameters
        ----------
        from_group : str
            The name of the group from which to read the waveforms to align.
        crop : tuple
            The start and end indices for cropping the hilbert signal before calculating the amplitude.
        '''
        with self.open_h5() as h5:
            _,hilbert = self.get_check_group(h5, from_group)
            h5[from_group].require_dataset('amplitude', shape=hilbert.shape[:-1], dtype='f4', track_order=True)
            if crop is not None: h5[from_group]['amplitude'].attrs['crop'] = crop    
            
    def fill_amplitude_data(self, idxs: Iterator[tuple], from_group='preprocessed_data', crop=None) -> None:
        """Align the first break of the waveforms in the dataset and write to new dataset

        Parameters
        ----------
        idxs : Iterator[tuple]
            An iterator of tuples containing indices to span all but first and last axis.
        from_group : str
            The name of the group from which to read the waveforms to align.
        crop : tuple
            The start and end indices for cropping the hilbert signal before calculating the amplitude.
        """          
        with self.open_h5() as h5:
            hilbert = h5[from_group]['hilbert']
            amplitude_dset = h5[from_group]['amplitude']
            
            for idx in tqdm(idxs, desc="Calculating Amplitude", total=math.prod(self.shape[:-1])):
                amplitude_dset[*idx] = self._amplitude(hilbert[*idx], crop=crop)
    
    def write_amplitude_data(self):
        """Write amplitude data to the HDF5 file. Calls setup and fill functions with specific indexer."""
        raise NotImplementedError         

    # -------------------------------------------------------------------------
    # Correlation Functions
    # -------------------------------------------------------------------------          
    
    @staticmethod
    def _xcorr(waveform, mother_waveform, mode='same'):
            '''
            Calculate the cross-correlation of the waveform with the mother waveform.

            Parameters:
            -----------
            waveform: np.ndarray
                The 1D waveform to calculate the cross-correlation of.
            mother_waveform: np.ndarray
                The 1D mother waveform to cross-correlate with.

            Returns:
            --------
            np.ndarray
                The cross-correlation of the waveform with the mother waveform.
            '''
            return correlate(waveform, mother_waveform, mode=mode)
            
    def _setup_mother_waveform(self, mother_db_path, thresh=0.1, **kwargs):
        """Set up the mother waveform for cross-correlation.
        TODO: for future, if you want multiple input references
        
        Parameters:
        -----------
        mother_db_path: str
            The path to the SQLite database containing the mother waveform.
        thresh: float
            The threshold for cropping the mother waveform based on the hilbert envelope.
        disp: bool
            Whether to display the mother waveform and its hilbert envelope.
            
        Returns:
        --------
        tuple
            A tuple containing the cropped time array, cropped waveform, and transducer frequency (MHz).
        """
        mother_db = AcousticSQlite(mother_db_path)
        mother_waveform = mother_db.fetch_waveform(waveform=mother_db.waveform_columns[0], row=0)
        mother_time_array = mother_db.fetch_time(row=0)
        hilbert_ = np.abs(hilbert(mother_waveform))
        indices = np.where(np.abs(hilbert_) > hilbert_.max()*thresh)[0]
        max_ind = np.argmax(hilbert_)
        crop = (2*indices[0]-max_ind, 2*indices[-1]-max_ind)
        
        return mother_waveform[crop[0]:crop[1]], mother_time_array[crop[0]:crop[1]], mother_db.parameters['transducerFrequency']
            
    def _interp_mother_waveform(self, time_array, mother_waveform, mother_time_array):
        '''
        Interpolate the mother waveform to match the time array of the dataset. The starting times are matched

        Parameters:
        -----------
        time_array: np.ndarray
            ...
        mother_waveform: np.ndarray
            The 1D mother waveform to interpolate.
        mother_time_array: np.ndarray
            ...
            
        Returns:
        --------
        (np.ndarray, np.ndarray)
            The interpolated mother waveform and its corresponding time array.
        '''
        time_array = self.parameters['time_array']
        t_diff = mother_time_array[0]-time_array[0] # so interpolated and true data start at the same time. TODO: phase effect?
        interp_func = CubicSpline(mother_time_array-t_diff, mother_waveform)  
        new_waveform = interp_func(time_array[:len(mother_waveform)])      
        return new_waveform/np.max(np.abs(new_waveform)), time_array[:len(mother_waveform)]
            
    def setup_xcorrelation_data(self, mother_db_path, from_group='preprocessed_data', mode='same', thresh=0.1, **kwargs):
        '''
        Set up the group storing waveforms aligned by first breakfor the dataset.
        
        Parameters
        ----------
        from_group : str
            The name of the group from which to read the waveforms to align.
        mother_waveform : np.ndarray
            The mother waveform to use for cross-correlation.
        mother_time_array : np.ndarray
            The time array of the mother waveform.
        mode : str
            The mode to use for the cross-correlation calculation.
        crop : tuple
            The start and end indices for cropping the hilbert signal before calculating the amplitude.
        '''
        mother_waveform, mother_time_array, mother_frequency = self._setup_mother_waveform(mother_db_path, thresh, **kwargs)
        with self.open_h5() as h5:
            waveform, hilbert = self.get_check_group(h5, from_group)
            x_corr = h5[from_group].require_dataset('x_corr', shape=hilbert.shape, dtype='f4', track_order=True)
            
            lags = correlation_lags(waveform.shape[-1], mother_waveform.shape[-1], mode=mode)
            x_corr.attrs['mode'] = mode
            x_corr.attrs['mother_waveform'], x_corr.attrs['mother_time_array'] = self._interp_mother_waveform(waveform.attrs['time_array'], mother_waveform, mother_time_array)
            x_corr.attrs['mother_frequency'] = mother_frequency
            x_corr.attrs['lags'] = lags

    def fill_xcorrelation_data(self, idxs: Iterator[tuple], from_group='preprocessed_data', mode='same') -> None:
        """Align the first break of the waveforms in the dataset and write to new dataset

        Parameters
        ----------
        idxs : Iterator[tuple]
            An iterator of tuples containing indices to span all but first and last axis.
        from_group : str
            The name of the group from which to read the waveforms to align.
        mode : str
            The mode to use for the cross-correlation calculation.
        crop : tuple
            The start and end indices for cropping the hilbert signal before calculating the amplitude.
        """          
        with self.open_h5() as h5:
            waveforms = h5[from_group]['waveforms']
            mother_waveform = h5[from_group]['x_corr'].attrs['mother_waveform']
            
            for idx in tqdm(idxs, desc= "Calculating Cross-correlation", total=math.prod(self.shape[:-1])):
                h5[from_group]['x_corr'][*idx] = self._xcorr(waveforms[*idx], mother_waveform, mode=mode)
    
    def write_xcorrelation_data(self, mother_db_path: str, mode: str = 'same', thresh: float = 0.1, **kwargs):
        """Write amplitude data to the HDF5 file. Calls setup and fill functions with specific indexer."""
        raise NotImplementedError         

    # -------------------------------------------------------------------------
    # Frequency Domain
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _fft(waveform): 
        '''O(B · N log N)'''
        return np.fft.rfft(waveform) 
    
    @staticmethod
    def _abs(fft_): 
        '''O(B · M)'''
        return np.abs(fft_) # O(B · M)

    @staticmethod
    def _angle(fft_): 
        '''O(B · M)'''
        return np.angle(fft_)

    @staticmethod
    def _phase_delay(phase_, freq): 
        '''O(B · M)'''
        return phase_ / freq

    @staticmethod
    def _group_delay(phase_, freq): 
        '''O(B · M)
        Parameters
        ----------
        phase_ : np.ndarray
            The phase of the FFT in (radians).
        freq : np.ndarray
            The frequencies corresponding to the FFT.
            
        Returns
        -------
        np.ndarray
            The group delay, in units of 1/(frequency units).
        '''
        return -np.gradient(np.unwrap(phase_, axis=-1, discont=0.5*np.pi), 2*np.pi*freq, axis=-1)
    
    @staticmethod
    def _ifft(fft_):
        '''O(B · N log N)'''
        return np.fft.irfft(fft_)
    
    def setup_fft_data(self, from_group='preprocessed_data', window=np.hanning):
        '''
        Set up the group and dataset storing waveforms in the frequency domain for the dataset.

        Parameters
        ----------
        from_group : str
            The name of the group from which to read the waveforms to transform.
        '''
        with self.open_h5() as h5:
            from_waveform, _ = self.get_check_group(h5, from_group)
            dset = h5[from_group].require_dataset('fft_data', shape=from_waveform.shape[:-1] + (from_waveform.shape[-1]//2+1,), dtype='c8', track_order=True)
            
            dset.attrs['freq_array'] = np.fft.rfftfreq(from_waveform.shape[-1], d=self.parameters['dt']['value'])
            dset.attrs['window'] = window.__name__ if window is not None else None
            
    def fill_fft_data(self, idxs: Iterator[tuple], from_group='preprocessed_data', window=np.hanning) -> None:
        '''
        Set up the group storing waveforms in the frequency domain for the dataset.

        Parameters
        ----------
        idxs : Iterator[tuple]
            An iterator of tuples containing indices to span all but first and last axis.
        from_group : str
            The name of the group from which to read the waveforms to transform.
        '''
        with self.open_h5() as h5:
            from_waveform, _ = self.get_check_group(h5, from_group)
            fft_dset = h5[from_group]['fft_data']

            if window is not None:
                window = window(from_waveform.shape[-1])
                for idx in tqdm(idxs, desc="Computing FFT", total=math.prod(self.shape[:-1])):
                    fft_dset[*idx] = self._fft(window * from_waveform[*idx])
            else:
                for idx in tqdm(idxs, desc="Computing FFT", total=math.prod(self.shape[:-1])):
                    fft_dset[*idx] = self._fft(from_waveform[*idx])
         
    def write_fft_data(self, from_group='preprocessed_data', window=np.hanning):
        """Write FFT data to the HDF5 file. Calls setup and fill functions with specific indexer."""
        raise NotImplementedError
                
    def get_fft_data(self, idx, h5, from_group='preprocessed_data'):
        '''Get the values as the full spectrum, not just real portion
        
        Parameters
        ----------
        idx : tuple
            The index of the FFT data to retrieve.

        Returns
        -------
        np.ndarray
            The FFT data for the given index.
        '''
        try: 
            return np.stack( [ h5[from_group]['fft_data'][*idx][...,::-1], h5[from_group]['fft_data']['calculations'][*idx]] )
        except KeyError as e: 
            print(e, 'Run setup_fft_data first'); return None
            
    def get_fft_frequencies(self, idx, h5, from_group='preprocessed_data'):
        '''Get the values as the full spectrum, not just real portion
        
        Parameters
        ----------
        idx : tuple
            The index of the FFT data to retrieve.

        Returns
        -------
        np.ndarray
            The FFT data for the given index.
        '''
        try: 
            return np.stack( [ h5[from_group]['fft_data'].attrs['freq_array'][...,::-1], h5[from_group]['fft_data']['calculations'][*idx].attrs['freq_array'] ] )
        except KeyError as e: 
            print(e, 'Run setup_fft_data first'); return None

    # -------------------------------------------------------------------------
    # Time-frequency
    # -------------------------------------------------------------------------          
    
    # Complex Morlet CWT - Time-dependent amplitude and phase changes
    def morlet_fwhm_info(self, wavelet_name, transducer_frequency):
        """FWHM of a PyWavelets complex Morlet ('cmorB-C') scaled to transducer_frequency.
        Prints and returns widths of the amplitude envelope (not power).
        """
        B, C = map(float, wavelet_name.removeprefix('cmor').split('-'))
        time_fwhm = 2 * np.sqrt(B * np.log(2)) * C / transducer_frequency
        print(wavelet_name, f"at {transducer_frequency:.2f} MHz, width at fwhm: {time_fwhm * 1e6:.2f} us")

    def check_wavelet_parameters(self, wavelet_name, n_scales, sampling_period, samples):
        transducer_frequency = self.parameters['transducerFrequency']['value']*1e6
        freqs_of_interest = np.geomspace(transducer_frequency/5, transducer_frequency*10, num=n_scales) #  in Hz
        scales = pywt.central_frequency( wavelet_name)/ sampling_period/ freqs_of_interest # scale = f_c * fs / f_target
        
        self.morlet_fwhm_info(wavelet_name, freqs_of_interest[0])
        self.morlet_fwhm_info(wavelet_name, transducer_frequency)
        self.morlet_fwhm_info(wavelet_name, freqs_of_interest[-1])
        period_samples = scales / pywt.central_frequency(wavelet_name) # wavelet period in samples
        print('Periods in record (lowest f, highest f):', samples/period_samples.max(), samples/period_samples.min())
        
        prompt = "Are you satisfied with these parameters? (y - proceed with calculation / n - exit): "
        answer = input(prompt).strip().lower()
        if answer == 'y': return scales, freqs_of_interest
        print("Parameters rejected, exiting.")
        sys.exit(0)
    
    def setup_cwt_data(self, from_group='preprocessed_data', wavelet_name='cmor0.5-1.0', n_scales=100, **kwargs):
        '''
        Set up the group storing waveforms aligned by first breakfor the dataset.
        
        Parameters
        ----------
        from_group : str
            The name of the group from which to read the waveforms to align.
        wavelet : str
            The pywt wavelet to use for the continuous wavelet transform.
        '''
        sampling_period = self.parameters['dt']['value']
        with self.open_h5() as h5:
            waveform, _ = self.get_check_group(h5, from_group)
            scales, freqs_of_interest = self.check_wavelet_parameters(wavelet_name, n_scales, sampling_period, waveform.shape[-1])
            cwt_dset = h5[from_group].require_dataset('cwt', shape=waveform.shape[:-1]+(n_scales, waveform.shape[-1]), dtype='c8', track_order=True)

            cwt_dset.attrs['wavelet_name'] = wavelet_name
            cwt_dset.attrs['freqs_of_interest'] = freqs_of_interest
            cwt_dset.attrs['scales'] = scales

    def fill_cwt_data(self, idxs: Iterator[tuple], from_group='preprocessed_data') -> None:
        """Align the first break of the waveforms in the dataset and write to new dataset

        Parameters
        ----------
        idxs : Iterator[tuple]
            An iterator of tuples containing indices to span all but first and last axis.
        from_group : str
            The name of the group from which to read the waveforms to align.
        mode : str
            The mode to use for the cross-correlation calculation.
        crop : tuple
            The start and end indices for cropping the hilbert signal before calculating the amplitude.
        """          
        with self.open_h5() as h5:
            waveforms = h5[from_group]['waveforms']
            cwt_dset =  h5[from_group]['cwt']
            scales = cwt_dset.attrs['scales']
            wavelet = cwt_dset.attrs['wavelet_name']
            sampling_period = self.parameters['dt']['value']

            for idx in tqdm(idxs, desc= "Calculating CWT", total=math.prod(self.shape[:-1])):
               cwt_dset[*idx],_ = pywt.cwt(waveforms[*idx], scales, wavelet, sampling_period=sampling_period, method='fft')
    
    def write_cwt_data(self):
            """Write amplitude data to the HDF5 file. Calls setup and fill functions with specific indexer."""
            raise NotImplementedError         
    
    # Cross-wavelet transform,  Wavelet coherence?
    
    
    # -------------------------------------------------------------------------
    # basic viz
    # -------------------------------------------------------------------------

    def view_fft_plots(self, idx, crop=None, from_group='preprocessed', unwrap=False): #TODO: check
        plt.clf()
        fig, ax = plt.subplots(2, 2, figsize=(8, 6))
        ax = ax.flatten()
        
        with self.open_h5() as h5:
            if crop is None: crop = [0, len(h5[from_group]['waveforms'][0])]
            data = h5[from_group]['waveforms'][:,*idx]
            time_array = h5[from_group]['waveforms'].attrs['time_array']

            fft_data = h5[from_group]['fft_data'][:,*idx]
            freq = h5[from_group]['fft_data'].attrs['freq_array']/1e6
            
            for i, dat in enumerate(fft_data):
                
                magnitude = self._abs(dat)/len(dat)*2
                phase = self._angle(dat)
                if unwrap: phase = np.unwrap(phase)
                group_delay = self._group_delay(phase, freq)
                
                ax[0].plot(time_array/1e3, data[i])
                ax[0].set_title('Original Waveform')
                ax[0].set_xlabel('Time (us)')
                ax[0].set_ylabel('Magnitude (mV)') 
                               
                ax[1].plot(freq[crop[0]:crop[1]], magnitude[crop[0]:crop[1]], label=self.parameters['waveform_labels'][i])
                ax[1].set_title('Magnitude')
                ax[1].set_ylabel('Magnitude (normalized)')
                
                ax[2].plot(freq[crop[0]:crop[1]], phase[crop[0]:crop[1]])
                ax[2].set_title('Phase')
                ax[2].set_ylabel('Phase (rad)')
                
                ax[3].plot(freq[crop[0]:crop[1]], group_delay[crop[0]:crop[1]])
                ax[3].set_title('Group Delay')
                ax[3].set_ylabel('Group Delay (us)')
            
            [a.set_xlabel('Frequency (MHz)') for a in ax[1:]]
            ax[1].legend()
                
        fig.tight_layout()
        plt.show()

    def view_waveform(self, idx, from_group='preprocessed_data', from_dset='waveforms'):    
        fig, ax = plt.subplots()
        with self.open_h5() as h5:
            data = h5[from_group][from_dset]
            ax.plot(data.attrs['time_array'], data[:,*idx])
        fig.show()
       
    def view_mother_waveform(self, from_group='preprocessed_data', from_dset='x_corr'):
        fig, ax = plt.subplots(figsize=(4,3))
        with self.open_h5() as h5:
            mother_waveform = h5[from_group][from_dset].attrs['mother_waveform']
            mother_time_array = h5[from_group][from_dset].attrs['mother_time_array']
            mother_frequency = h5[from_group][from_dset].attrs['mother_frequency']
            ax.plot(mother_time_array, mother_waveform)
            ax.set(xlabel='Time (ns)', ylabel='Voltage (mV)', title=f'Mother Waveform, {mother_frequency} MHz');
        fig.show()

    def view_continuous_wavelet(self, idx, from_group='preprocessed_data', from_dset='cwt'): # TODO: check
        fig, ax = plt.subplots(len(self.waveform_labels), figsize=(4,3))
        with self.open_h5() as h5:
            dset = h5[from_group][from_dset][:,*idx]
            wavelet = pywt.ContinuousWavelet(dset['attrs']['wavelet'])
            psi, x = wavelet.wavefun(precision=10)   # precision sets sample count (2**precision points)

            fig, ax = plt.subplots( figsize=(8, 6))
            for i, dat in dset:
                ax.imshow(dat)
                ax.imshow()
                ax.axvline(wavelet.center_frequency, color='r', ls=':', label='center freq')
                ax.set_xlabel('frequency (cycles per wavelet unit)')
                ax.set_xlim(-3, 3)
                ax.legend()

        plt.tight_layout()
        plt.show()

# Naming convention:  Acoustic<ExperimentType>Database 
# -- Specific to the type of experiment and how the data is collected. They define how to index the data and how to convert from SQLite to HDF5.
# -- Have full functionality and can be used to store analysis results in the HDF5 file.
class AcousticScanDatabase(AcousticDatabase):
    """Acoustics database subclass for 2D scan experiments (collection_index <-> X,Z coordinates).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    # -------------------------------------------------------------------------
    # setup and indexing functions
    # -------------------------------------------------------------------------
    
    def setup_parameters(self):
        """Setup the parameters for the acoustic data. Gets the parameters from the SQLite database and calculates the shape and length of the dataset. Saves the parameters to the HDF5 file."""
        print('Setting up parameters in HDF5 dataset...')
        self.parameters = self.get_sqlite_parameters()
        self.parameters['shape'] = (
            len(self.parameters["waveform_labels"]),
            math.ceil(self.parameters["primaryAxisRange"]['value'] / abs(self.parameters["primaryAxisStep"]['value'])) + 1,
            math.ceil(self.parameters["secondaryAxisRange"]['value'] / abs(self.parameters["secondaryAxisStep"]['value'])) + 1,
            len(self.parameters["time_array"]))
        self.parameters['len'] = self.parameters['shape'][1]*self.parameters['shape'][2]
        
        self.shape = self.parameters['shape']
        self.len = self.parameters['len']
        
        self.save_parameters()

    def sqlite_to_h5(self): 
        """Convert the SQLite database to an HDF5 file.
        This method sets up the parameters, raw dataset, and indexing parameters. 
        """
        self.setup_parameters()

        self.setup_raw_dataset()
        self.fill_raw_dataset(self.idx_sqlite_data_generator())
        
        self.write_indexed_parameters()
        
        print(f'\tFinished')

    def idx_sqlite_data_generator(self):
        '''Generate indexed data from the SQLite database. 
        This method yields tuples of indices and corresponding waveform data for each waveform label, X coordinate, and Z coordinate. 
        The indices are in the format (waveform_index, x_index, z_index), and the waveform data is fetched from the SQLite database using the fetch_waveform method.
        '''
        for w,waveform in enumerate(self.parameters['waveform_labels']):
            for x in range(self.shape[1]):
                for z in range(self.shape[2]):
                    yield (w,x,z), self.sqlite_database.fetch_waveform(waveform, x+z*self.shape[1])

    def w_x_z_generator(self):
        '''Generate indices for waveform, X coordinate, and Z coordinate.
        This method yields tuples of indices for each waveform label, X coordinate, and Z coordinate.
        The indices are in the format (waveform_index, x_index, z_index).
        '''
        for w in range(len(self.parameters['waveform_labels'])):
            for x in range(self.shape[1]):
                for z in range(self.shape[2]):
                    yield (w,x,z)

    # -------------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------------
    
    def write_indexed_parameters(self, keys=['collection_index', 'time_collected', 'X', 'Z']):
        """Write indexed parameters to the HDF5 file. Calls setup and fill functions with specific indexer.

        Parameters
        ----------
        keys : list[str]
            List of parameter names to write.
        values : list[np.ndarray]
            List of parameter values to write.
        shapes : list[tuple]
            List of shapes for each parameter.
        """
        print(f'Writing indexed parameters for: {keys}')
        values = (self.sqlite_database.fetch_column(column) for column in (*keys,))
        shapes=[self.shape[1:3], self.shape[1:3], self.shape[1:3], self.shape[1:3]] # TODO better way to do this? shapes should be based on the length of the values, but we don't know that until we fetch them.
        # TODO: add cases for if there is offset, gain and autorange 
        
        self.setup_indexed_parameters(keys, shapes)
        self.fill_indexed_parameter(keys, values, shapes)
    
    # -------------------------------------------------------------------------
    # preprocessing
    # -------------------------------------------------------------------------
    
    def write_preprocessed_data(self, **kwargs):
        '''Write preprocessed data to the HDF5 file.

        Parameters
        ----------
        **kwargs
            Parameters passed to :meth:`setup_preprocessed_data`. See that
            method for the available preprocessing options.
        '''
        self.setup_preprocessed_data(**kwargs)
        self.fill_preprocessed_data(self.w_x_z_generator() )
        print('\tFinished')
 
    # -------------------------------------------------------------------------
    # alignment
    # -------------------------------------------------------------------------
        
    def write_first_break_aligned_data(self, **kwargs): 
        """Align the first break of the waveforms in the dataset and write to the HDF5 file.

        Parameters
        ----------
        **kwargs
            Parameters passed to :meth:`setup_first_break`. See that
            method for the available alignment options.
        """
        self.setup_first_break_aligned_data(**kwargs)
        self.fill_first_break_aligned_data(self.w_x_z_generator(), **kwargs)
        print('\tFinished')
       
    # -------------------------------------------------------------------------
    # basic analysis
    # -------------------------------------------------------------------------
    
    def write_tof_data(self, from_group='preprocessed_data', threshold_fraction=0.1, crop=None):
        """Write time of flight data to the HDF5 file. Calls setup and fill functions with specific indexer."""
        self.setup_tof_data(from_group=from_group, threshold_fraction=threshold_fraction, crop=crop)
        self.fill_tof_data(self.w_x_z_generator(), from_group=from_group, crop=crop)
        print('\tFinished')
        
    def write_amplitude_data(self, from_group='preprocessed_data', crop=None):
        """Write amplitude data to the HDF5 file. Calls setup and fill functions with specific indexer."""
        self.setup_amplitude_data(from_group, crop)
        self.fill_amplitude_data(self.w_x_z_generator(), from_group, crop)
        print('\tFinished')    
           
    # -------------------------------------------------------------------------
    # FFT
    # -------------------------------------------------------------------------
    
    def write_fft_data(self, from_group, window=np.hanning, **kwargs): 
        '''Write preprocessed data to the HDF5 file.

        Parameters
        ----------
        **kwargs
            Parameters passed to :meth:`setup_preprocessed_data`. See that
            method for the available preprocessing options.
        '''
        self.setup_fft_data(from_group=from_group, **kwargs)
        self.fill_fft_data(self.w_x_z_generator(), from_group = from_group,**kwargs )
        print('\tFinished')
    
    # -------------------------------------------------------------------------
    # correlation functions
    # -------------------------------------------------------------------------
    
    def write_xcorrelation_data(self, mother_db_path: str, from_group='preprocessed_data', mode='same'):
        """Write cross-correlation data to the HDF5 file. Calls setup and fill functions with specific indexer."""
        self.setup_xcorrelation_data(mother_db_path, from_group, mode)
        self.fill_xcorrelation_data(self.w_x_z_generator(), from_group, mode)
        print('\tFinished')  

    # -------------------------------------------------------------------------
    # Time-frequency 
    # -------------------------------------------------------------------------

    def write_cwt_data(self, from_group='preprocessed_data', wavelet_name='cmor0.5-1.0', n_scales=100,):
        """Write cross-correlation data to the HDF5 file. Calls setup and fill functions with specific indexer."""
        self.setup_cwt_data(from_group, wavelet_name, n_scales)
        self.fill_cwt_data(self.w_x_z_generator(), from_group,)
        print('\tFinished')  


    # -------------------------------------------------------------------------
    # basic viz
    # -------------------------------------------------------------------------

    def view_img_plot(self, idx, from_group = 'preprocessed_data', from_dataset = 'waveforms'): 
        '''View the image plot for a specific index.

        Parameters
        ----------
        idx : int
            The index of the data to view. Excludes the waveform axis, so it should be a tuple of (x_index, z_index, y_index).
        from_group : str, optional
            The group in the HDF5 file from which to retrieve the data.
        '''
        fig, ax = plt.subplots(1+self.shape[0], 1, figsize=(6, 2*(1+self.shape[0])))
        ax = ax.flatten()
        with self.open_h5() as h5:
            data = h5[from_group][from_dataset]
            time_array = h5[from_group][from_dataset].attrs['time_array']/1e3
            for i, dat in enumerate(data):
                a = ax[i].imshow(dat[...,idx[-1]].T,)
                plt.colorbar(a, ax=ax[i], label='Ampl. (mV)')
                ax[i].set_title(self.parameters['waveform_labels'][i])
                ax[i].plot(idx[0], idx[1], 'ro', markersize=5)
                
                ax[-1].plot(time_array, dat[*idx[:-1]], label=self.parameters['waveform_labels'][i])
                ax[-1].axvline(x=time_array[idx[-1]], color='r', linestyle='--', linewidth=1)
                ax[-1].set_xlabel('Time (us)')
                ax[-1].set_ylabel('Ampl. (mV)')
            ax[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2, borderaxespad=0.)
        fig.suptitle('t=({} us), x,z=({},{} mm), {} from {}'.format(time_array[idx[-1]], idx[0], idx[1], from_dataset, from_group))
        fig.tight_layout()
        fig.show()

    def view_instantaneous_amplitude_img_plot(self, idx, from_group = 'preprocessed_data'): 
        '''View the image plot for a specific index.

        Parameters
        ----------
        idx : int
            The index of the data to view. Excludes the waveform axis, so it should be a tuple of (x_index, z_index, y_index).
        from_group : str, optional
            The group in the HDF5 file from which to retrieve the data.
        '''
        fig, ax = plt.subplots(1+self.shape[0], 1, figsize=(6, 2*(1+self.shape[0])))
        ax = ax.flatten()
        with self.open_h5() as h5:
            data = h5[from_group]['hilbert']
            time_array = h5[from_group]['hilbert'].attrs['time_array']/1e3
            for i, dat in enumerate(data):
                a = ax[i].imshow(abs(dat[...,idx[-1]]).T,)
                plt.colorbar(a, ax=ax[i], label='Ampl. (au)')
                ax[i].set_title(self.parameters['waveform_labels'][i])
                ax[i].plot(idx[0], idx[1], 'ro', markersize=5)
                
                ax[-1].plot(time_array, abs(dat[*idx[:-1]]), label=self.parameters['waveform_labels'][i])
                ax[-1].axvline(x=time_array[idx[-1]], color='r', linestyle='--', linewidth=1)
                ax[-1].set_xlabel('Time (us)')
                ax[-1].set_ylabel('Ampl. (au)')
            ax[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2, borderaxespad=0.)
        fig.suptitle('t=({} us), x,z=({},{} mm), inst. magnitude from {}'.format(time_array[idx[-1]], idx[0], idx[1], from_group))
        fig.tight_layout()
        fig.show()
    
    def view_instantaneous_phase_img_plot(self, idx, from_group = 'preprocessed_data', unwrap=False):
        '''View the image plot for a specific index.

        Parameters
        ----------
        idx : int
            The index of the data to view. Excludes the waveform axis, so it should be a tuple of (x_index, z_index, y_index).
        from_group : str, optional
            The group in the HDF5 file from which to retrieve the data.
        unwrap : bool, optional
            Whether to unwrap the phase data.
        '''
        cmap = 'viridis' if unwrap else 'twilight'
        fig, ax = plt.subplots(1+self.shape[0], 1, figsize=(6, 2*(1+self.shape[0])))
        ax = ax.flatten()
        with self.open_h5() as h5:
            data = h5[from_group]['hilbert']
            time_array = h5[from_group]['hilbert'].attrs['time_array']/1e3
            for i, dat in enumerate(data):
                if unwrap: phase_data = np.unwrap(np.angle(dat), axis=-1)
                else: phase_data = np.angle(dat)

                a = ax[i].imshow(phase_data[...,idx[-1]].T, cmap=cmap)
                plt.colorbar(a, ax=ax[i], label='Phase (rad)')
                ax[i].set_title(self.parameters['waveform_labels'][i])
                ax[i].plot(idx[0], idx[1], 'ro', markersize=5)
                
                ax[-1].plot(time_array, phase_data[*idx[:-1]], label=self.parameters['waveform_labels'][i])
                ax[-1].axvline(x=time_array[idx[-1]], color='r', linestyle='--', linewidth=1)
                ax[-1].set_xlabel('Time (us)')
                ax[-1].set_ylabel('Phase (rad)')
            ax[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2, borderaxespad=0.)
        fig.suptitle('t=({} us), x,z=({},{} mm), inst. phase from {}'.format(time_array[idx[-1]], idx[0], idx[1], from_group))
        fig.tight_layout()
        fig.show()
    
    def view_instantaneous_frequency_img_plot(self, idx, from_group = 'preprocessed_data'): #TODO: check
        '''View the image plot for instantanoues freuqency of specific index. Works with analytic signal

        Parameters
        ----------
        idx : int
            The index of the data to view. Excludes the waveform axis, so it should be a tuple of (x_index, z_index, y_index).
        from_group : str, optional
            The group in the HDF5 file from which to retrieve the data.
        '''
        fig, ax = plt.subplots(1+self.shape[0], 1, figsize=(6, 2*(1+self.shape[0])))
        ax = ax.flatten()
        with self.open_h5() as h5:
            data = h5[from_group]['hilbert']
            time_array = h5[from_group]['hilbert'].attrs['time_array']/1e3
            for i, dat in enumerate(data):
                phase_data = np.unwrap(np.angle(dat), axis=-1)
                inst_freq_data = np.gradient(phase_data, 2*np.pi*time_array, axis=-1)
                
                a = ax[i].imshow(inst_freq_data[...,idx[-1]].T,)
                plt.colorbar(a, ax=ax[i], label='Inst. Freq. (MHz)')
                ax[i].set_title(self.parameters['waveform_labels'][i])
                ax[i].plot(idx[0], idx[1], 'ro', markersize=5)
                
                ax[-1].plot(time_array, inst_freq_data[*idx[:-1]], label=self.parameters['waveform_labels'][i])
                ax[-1].axvline(x=time_array[idx[-1]], color='r', linestyle='--', linewidth=1)
                ax[-1].set_xlabel('Time (us)')
                ax[-1].set_ylabel('Inst. Freq. (MHz)')
            ax[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2, borderaxespad=0.)
        fig.suptitle('t=({} us), x,z=({},{} mm), inst. freq. from'.format(time_array[idx[-1]], idx[0], idx[1], from_group))
        fig.tight_layout()
        fig.show()

    def img_tof(self, idx=None, from_group='preprocessed_data'): #TODO: check
        '''View the image plot for a specific index.

        Parameters
        ----------
        idx : int
            The index of the data to view. Excludes the waveform axis, so it should be a tuple of (x_index, z_index, y_index).
        from_group : str, optional
            The group in the HDF5 file from which to retrieve the data.
        '''
        fig, ax = plt.subplots(self.shape[0], figsize=(self.shape[0]*2, 3))
        with self.open_h5() as h5:
            for i, tof in enumerate(h5[from_group]['tof']):
                a = ax[i].imshow(1e-3*tof.T,)
                plt.colorbar(a, ax=ax[i], label='ToF (us)')
                if idx is not None:
                    ax[i].plot(idx[0], idx[1], 'ro', markersize=5)
                    ax[i].set_title(self.parameters['waveform_labels'][i]+' {:.2f}'.format(tof[*idx]*1e-3)+' us')
                else: ax[i].set_title(self.parameters['waveform_labels'][i])
            
        fig.suptitle('ToF of {}{}'.format(from_group, f', x,z=({idx[0]},{idx[1]})' if idx is not None else ''))
        fig.tight_layout()
        fig.show()
        
    def img_amplitude(self, idx=None, from_group='preprocessed_data'): #TODO: check
        '''View the image plot for a specific index.

        Parameters
        ----------
        idx : int
            The index of the data to view. Excludes the waveform axis, so it should be a tuple of (x_index, z_index, y_index).
        from_group : str, optional
            The group in the HDF5 file from which to retrieve the data.
        '''
        fig, ax = plt.subplots(self.shape[0], figsize=(self.shape[0]*2, 3))
        with self.open_h5() as h5:
            for i, amplitude in enumerate(h5[from_group]['amplitude']):
                a = ax[i].imshow(1e-3*amplitude.T,)
                plt.colorbar(a, ax=ax[i], label='Amplitude (a.u.)')
                if idx is not None:
                    ax[i].plot(idx[0], idx[1], 'ro', markersize=5)
                    ax[i].set_title(self.parameters['waveform_labels'][i]+' {:.2f}'.format(amplitude[*idx]*1e-3)+' a.u.')
                else: ax[i].set_title(self.parameters['waveform_labels'][i])
            
        fig.suptitle('Amplitude of {}{}'.format(from_group, f', x,z=({idx[0]},{idx[1]})' if idx is not None else ''))
        fig.tight_layout()
        fig.show()

    def view_fft_amplitude_img_plot(self, idx, crop=None, from_group = 'preprocessed_data'): 
        '''View the image plot for a specific index.

        Parameters
        ----------
        idx : int
            The index of the data to view. Excludes the waveform axis, so it should be a tuple of (x_index, z_index, y_index).
        crop : tuple, optional
            The crop bounds for the FFT data.
        from_group : str, optional
            The group in the HDF5 file from which to retrieve the data.
        '''
        fig, ax = plt.subplots(1+self.shape[0], 1, figsize=(6, 2*(1+self.shape[0])))
        ax = ax.flatten()
        
        with self.open_h5() as h5:
            fft_data = h5[from_group]['fft_data'][...,crop[0]:crop[1]] if crop is not None else h5[from_group]['fft_data']
            freq_array = h5[from_group]['fft_data'].attrs['freq_array']/1e6
            
            for i, dat in enumerate(fft_data):
                mag = self._abs(dat)/len(dat)*2
                a = ax[i].imshow(mag[...,idx[-1]].T,)
                plt.colorbar(a, ax=ax[i], label='Ampl. (au)')
                ax[i].set_title(self.parameters['waveform_labels'][i])
                ax[i].plot(idx[0], idx[1], 'ro', markersize=5)

                ax[-1].plot(freq_array[crop[0]:crop[1]], mag[*idx[:-1]], label=self.parameters['waveform_labels'][i])
                ax[-1].axvline(x=freq_array[idx[-1]], color='r', linestyle='--', linewidth=1)
                ax[-1].set_xlabel('Frequency (MHz)')
                ax[-1].set_ylabel('Ampl. (au)')
                
            ax[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2, borderaxespad=0.)
        fig.suptitle('f=({:<.2f} MHz), x,z=({},{} mm), fft magnitude from {}'.format(freq_array[idx[-1]], idx[0], idx[1], from_group))
        fig.tight_layout()
        fig.show()
    
    def view_fft_phase_img_plot(self, idx, from_group = 'preprocessed_data', crop=None, unwrap=False):
        '''View the image plot for a specific index.

        Parameters
        ----------
        idx : int
            The index of the data to view. Excludes the waveform axis, so it should be a tuple of (x_index, z_index, y_index).
        from_group : str, optional
            The group in the HDF5 file from which to retrieve the data.
        unwrap : bool, optional
            Whether to unwrap the phase data.
        '''
        crop = [0, len(self.get_fft_frequencies(idx, self.open_h5(), from_group=from_group)[0])] if crop is None else crop
        cmap = 'viridis' if unwrap else 'twilight'
        fig, ax = plt.subplots(1+self.shape[0], 1, figsize=(6, 2*(1+self.shape[0])))
        ax = ax.flatten()
        with self.open_h5() as h5:
            data = h5[from_group]['fft_data']
            freq_array = h5[from_group]['fft_data'].attrs['freq_array']/1e6
            for i, dat in enumerate(data):
                if unwrap: phase_data = np.unwrap(np.angle(dat), axis=-1)[...,crop[0]:crop[1]]
                else: phase_data = np.angle(dat)[...,crop[0]:crop[1]]
                
                a = ax[i].imshow(phase_data[...,idx[-1]].T, cmap=cmap)
                plt.colorbar(a, ax=ax[i], label='Phase (rad)')
                ax[i].set_title(self.parameters['waveform_labels'][i])
                ax[i].plot(idx[0], idx[1], 'ro', markersize=5)
                
                ax[-1].plot(freq_array[crop[0]:crop[1]], phase_data[*idx[:-1]], label=self.parameters['waveform_labels'][i])
                ax[-1].axvline(x=freq_array[idx[-1]], color='r', linestyle='--', linewidth=1)
                ax[-1].set_xlabel('Frequency (MHz)')
                ax[-1].set_ylabel('Phase (rad)')
            ax[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2, borderaxespad=0.)
        fig.suptitle('f=({:.2f} MHz), x,z=({},{} mm), inst. phase from {}'.format(freq_array[idx[-1]], idx[0], idx[1], from_group))
        fig.tight_layout()
        fig.show()
    
    def view_fft_gdelay_img_plot(self, idx, from_group = 'preprocessed_data', crop=None): #TODO: check
        '''View the image plot for instantanoues freuqency of specific index. Works with analytic signal

        Parameters
        ----------
        idx : int
            The index of the data to view. Excludes the waveform axis, so it should be a tuple of (x_index, z_index, y_index).
        from_group : str, optional
            The group in the HDF5 file from which to retrieve the data.
        '''
        crop = [0, len(self.get_fft_frequencies(idx, self.open_h5(), from_group=from_group)[0])] if crop is None else crop
        fig, ax = plt.subplots(1+self.shape[0], 1, figsize=(6, 2*(1+self.shape[0])))
        ax = ax.flatten()
        with self.open_h5() as h5:
            data = h5[from_group]['fft_data']
            freq_array = h5[from_group]['fft_data'].attrs['freq_array']/1e6
            for i, dat in enumerate(data):
                phase_data = np.unwrap(np.angle(dat), axis=-1)
                gdelay_data = np.gradient(phase_data, 2*np.pi*freq_array, axis=-1)[...,crop[0]:crop[1]]
                
                a = ax[i].imshow(gdelay_data[...,idx[-1]].T,)
                plt.colorbar(a, ax=ax[i], label='Group Delay (us)')
                ax[i].set_title(self.parameters['waveform_labels'][i])
                ax[i].plot(idx[0], idx[1], 'ro', markersize=5)
                
                ax[-1].plot(freq_array[crop[0]:crop[1]], gdelay_data[*idx[:-1]], label=self.parameters['waveform_labels'][i])
                ax[-1].axvline(x=freq_array[idx[-1]], color='r', linestyle='--', linewidth=1)
                ax[-1].set_xlabel('Frequency (MHz)')
                ax[-1].set_ylabel('Group Delay (us)')
            ax[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2, borderaxespad=0.)
        fig.suptitle('f=({:<.2f} MHz), x,z=({},{} mm), group delay from {}'.format(freq_array[idx[-1]], idx[0], idx[1], from_group))
        fig.tight_layout()
        fig.show()

    def view_img_correlation(self, idx=None, from_group='first_break_aligned', from_dataset='x_corr'):
        '''View the image plot for a specific index.
    
        Parameters
        ----------
        idx : int
            The index of the data to view. Excludes the waveform axis, so it should be a tuple of (x_index, z_index, y_index).
        from_group : str, optional
            The group in the HDF5 file from which to retrieve the data.
        '''
        fig, ax = plt.subplots(1+self.shape[0], 1, figsize=(6, 2*(1+self.shape[0])))
        ax = ax.flatten()
        with self.open_h5() as h5:
            data = h5[from_group][from_dataset]
            time_lag = h5[from_group][from_dataset].attrs['lags']/1e3
            for i, dat in enumerate(data):
                a = ax[i].imshow(dat[...,idx[-1]].T,)
                plt.colorbar(a, ax=ax[i], label='Ampl. (mV)')
                ax[i].set_title(self.parameters['waveform_labels'][i])
                ax[i].plot(idx[0], idx[1], 'ro', markersize=5)
                
                ax[-1].plot(time_lag, dat[*idx[:-1]], label=self.parameters['waveform_labels'][i])
                ax[-1].axvline(x=time_lag[idx[-1]], color='r', linestyle='--', linewidth=1)
                ax[-1].set_xlabel('Time lag (us)')
                ax[-1].set_ylabel('Ampl. (mV)')
            ax[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2, borderaxespad=0.)
        fig.suptitle('t=({} us), x,z=({},{} mm), {} from {}'.format(time_lag[idx[-1]], idx[0], idx[1], from_dataset, from_group))
        fig.tight_layout()
        fig.show()
 
      
# # TODO
# class AcousticsSweepDatabase(AcousticDatabase):
#     """Acoustics database subclass for sweep experiments (collection_index <-> single swept parameter)."""

#     # -------------------------------------------------------------------------
#     # collection indexing
#     # -------------------------------------------------------------------------

#     def create_collection_index_mapping(self) -> None:
#         """Map collection_index to the swept parameter's values.
#         """
#         if self.parameters.get("experiment") != "sweep":
#             raise ValueError("Parameters do not describe a sweep experiment")

#         sweep_params = ast.literal_eval(self.parameters["sweepParams"])
#         idx = 0
#         for key, vals in sweep_params.items():
#             mapping = {}
#             self._add_column(key, self.acoustics_table, type=self.types_dict.get(type(vals[0]).__name__, "REAL"))
#             for val in vals:
#                 mapping[idx] = val
#                 idx += 1
#             self.store_acoustics_value(collection_index=list(mapping.keys()), column=key, value=list(mapping.values()))

#     def get_sweep_column(self) -> str:
#         """Get the name of the swept parameter column.

#         Returns
#         -------
#         str
#             The name of the swept parameter.
#         """
#         sweep_params = ast.literal_eval(self.parameters["sweepParams"])
#         return next(iter(sweep_params))

#     def get_sweep_values(self) -> list[Any]:
#         """Get the swept parameter's values, ordered by collection_index.

#         Returns
#         -------
#         list[Any]
#             The swept parameter values.
#         """
#         column = self.get_sweep_column()
#         query = f"SELECT {self._quote(column)} FROM {self._quote(self.acoustics_table)} ORDER BY collection_index"
#         return [r[0] for r in self.connection.execute(query)]

#     def fetch_collection_index_by_value(self, value: Any) -> int:
#         """Fetch the collection_index matching a swept parameter value.

#         Parameters
#         ----------
#         value : Any
#             The swept parameter value to look up.

#         Returns
#         -------
#         int
#             The matching collection_index.
#         """
#         column = self.get_sweep_column()
#         query = f"SELECT collection_index FROM {self._quote(self.acoustics_table)} WHERE {self._quote(column)}=?"
#         row = self.connection.execute(query, (value,)).fetchone()
#         if row is None:
#             raise ValueError(f"No collection_index found for {column}={value}")
#         return int(row[0])


# # TODO: build out and integrate with visualization classes    
# class AcousticScanFrequencyDomainDatabase(AcousticDatabase):
#     """A class for handling frequency domain data in an acoustics database."""
     
# class AcousticDatabaseCWT(AcousticDatabaseFrequencyDomain):
#     """A class for handling continuous wavelet transform data in an acoustics database.

#     This class extends the AcousticsDatabaseFrequencyDomain class to provide additional functionality
#     for working with continuous wavelet transform data.
#     """

# class AcousticDatabaseTMM(AcousticDatabaseFrequencyDomain):
#     """A class for handling time-frequency map data in an acoustics database.

#     This class extends the AcousticsDatabase class to provide additional functionality
#     for working with time-frequency map data.
#     """
