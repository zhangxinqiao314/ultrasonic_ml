import sys
# sys.path.append('..') # path to the src directory
# sys.path.append('/Users/xz498/Library/CloudStorage/OneDrive-DrexelUniversity/Chang Lab - Documents/General/Individual/Xinqiao Zhang/data_analysis/ultrasonicTesting/')

import os
import os.path

import itertools
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import pickleJar as pj

import pickle
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt

import pandas as pd

from  sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ipywidgets import interact
import ipywidgets as widgets
import plotly.graph_objects as go



## TODO: is it better to have h5 files instead?


### data

def delete_old_pickles(sqliteFiles):
    '''Delete old pickle files corresponding to the given sqlite files.'''
    for sqliteFile in sqliteFiles:
        pickleFile = sqliteFile.replace('.sqlite3', '.pickle')
        if os.path.exists(pickleFile):
            os.remove(pickleFile)
            print(f"Deleted old pickle file: {pickleFile}")
        else:
            print(f"No pickle file found for: {sqliteFile}")
            
            
def stitch_interp_normalize_pickles(pickleFiles, interp=True, saveName='combined.pickle', save_directory=None, num_points=2500, 
                                    overwrite_pickles=False, overwrite_combined=False):
    """
    Stitch together multiple pickle files into one.
    Crops to the latest start and earliest end time of all scans,
    interpolates to ``num_points``, and normalizes the voltage to [-1, 1].
    The combined result is written to a pickle file.

    Parameters
    ----------
    pickleFiles : list of str
        Paths to the pickle files to be combined.
    interp : bool, default=True
        Whether to interpolate the data.
    saveName : str, default='combined.pickle'
        Name of the output combined pickle file.
    save_directory : str or None, default=None
        Path to the directory where the output file will be saved.
    num_points : int, default=2500
        Number of points to interpolate the data to.
    overwrite : bool, default=False
        Whether to overwrite an existing pickle file.

    Returns
    -------
    combined_data : dict
        The combined data from all pickle files.
    """
    if save_directory: saveName = f'{save_directory}/{saveName}'
    
    if not overwrite_combined and os.path.isfile(saveName):
        print(f"Combined pickle file already exists. Conversion aborted for {saveName}")
        return -1
    
    combined_data = {'fileName': saveName, 
                     'parameters': {i: {} for i in range(len(pickleFiles))}}
    
    # check if the pickle file already exists. If it does, print an error message and exit early
    if os.path.isfile(combined_data['fileName']):
        if overwrite_pickles:  os.remove(combined_data['fileName'])
        else:
            print("sqliteToPickle Warning: pickle file " + combined_data['fileName'] + " already exists. Conversion aborted.")
            # return pj.loadPickle(combined_data['fileName'])

    start, end = 0, 100000
    for i, pickleFile in enumerate(pickleFiles):
        data = pj.loadPickle(pickleFile)
        combined_data['parameters'][i]['original_params'] = data['parameters']
        if data[0]['time'][0]>=start: start = data[0]['time'][0]
        if data[0]['time'][-1]<=end: end = data[0]['time'][-1]
        
    combined_data['parameters']['dt'] = (end-start)/num_points
    idx=0
    
    for i, pickleFile in enumerate(pickleFiles):
        data = pj.loadPickle(pickleFile)
        combined_data['parameters'][i]['file_start_idx'] = int(idx)
        for k,v in data.items():
            if str(k).isdigit():
                combined_data[idx] = v.copy()
                combined_data[idx]['exp_idx'] = i
                for key in v.keys():
                    if 'voltage' in key:
                        if interp: # TODO: need to recheck
                            combined_data[idx]['time'] = np.linspace(start, end, num_points)
                            combined_data[idx][key] = np.interp(combined_data[idx]['time'], v['time'], v[key])                
                        else: 
                            combined_data[idx][key] = v[key]
                idx+=1
        combined_data['parameters'][i]['file_end_idx'] = int(idx)
    
    # save the dataDict as a pickle. We checked if the file exists earlier, so this operation is safe
    with open(combined_data['fileName'], 'wb') as f:
        pickle.dump(combined_data, f)
    
    # return combined_data


def load_merge_to_controller_df(monitor_file_path_list, settings_file_path_list, monitor_cols = None, settings_cols = None):
    """Load and merge controller data from multiple files into a single DataFrame.

    Parameters
    ----------
    monitor_file_path_list : list of str or pathlib.Path
        Paths to controller monitor log files.
    settings_file_path_list : list of str or pathlib.Path
        Paths to controller settings log files.
    monitor_cols : list of str, optional
        Columns to read from monitor logs. If None, all columns are read.
    settings_cols : list of str, optional
        Columns to read from settings logs. If None, all columns are read.

    Returns
    -------
    pandas.DataFrame
        Merged controller monitor and settings data with a unified ``Time``
        column in seconds since the Unix epoch and a ``Time_str`` column
        preserving the original timestamp strings.

    Notes
    -----
    The monitor and settings logs are concatenated, sorted by time, and
    duplicate entries are dropped while keeping the last occurrence.
    Milliseconds are added to the timestamp to produce a floating-point
    seconds value.
    """
    print(f"Loading and merging controller logs")
    controller_monitor_dfs = [pd.read_csv(file, delimiter=';', usecols=monitor_cols) for file in monitor_file_path_list]

    controller_monitor_df = (
        pd.concat(controller_monitor_dfs, ignore_index=True)
        .sort_values(['Time', 'Milliseconds'])
        .drop_duplicates(keep='last')
        .reset_index(drop=True)  )

    controller_monitor_df['Time_str'] = controller_monitor_df['Time']
    controller_monitor_df['Time'] = controller_monitor_df['Time'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p').timestamp())
    controller_monitor_df['Time'] += controller_monitor_df['Milliseconds']/1000
    controller_monitor_df.drop(columns=['Milliseconds'], inplace=True)  

    ## Read in settings data, add ms to time
    controller_settings_dfs = [pd.read_csv(file, delimiter=';', usecols=settings_cols) for file in settings_file_path_list]

    controller_settings_df = (
        pd.concat(controller_settings_dfs, ignore_index=True)
        .sort_values(['Time', 'Milliseconds'])
        .drop_duplicates(keep='last')
        .reset_index(drop=True) )

    controller_settings_df['Time_str'] = controller_settings_df['Time']
    controller_settings_df['Time'] = controller_settings_df['Time'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p').timestamp())
    controller_settings_df['Time'] += controller_settings_df['Milliseconds']/1000
    controller_settings_df.drop(columns=['Milliseconds'], inplace=True)
    
    controller_monitor_df.sort_values('Time', inplace=True)
    controller_settings_df.sort_values('Time', inplace=True)
    
    common_cols = set(controller_monitor_df.columns) & set(controller_settings_df.columns) - {'Time'}

    ## Merge monitor and settings data
    controller_df = pd.merge_asof(
        left=controller_monitor_df,
        right=controller_settings_df.drop(columns=common_cols),
        left_on='Time',
        right_on='Time',
        direction='nearest' 
    )

    # Convert time to float
    controller_monitor_df['Time'] = controller_monitor_df['Time'].astype(float)
    controller_settings_df['Time'] = controller_settings_df['Time'].astype(float)
    controller_settings_df['Time'] = controller_settings_df['Time'].astype(float)

    # unify Time_str columns and remove the originals
    if 'Time_str_x' in controller_df.columns or 'Time_str_y' in controller_df.columns:
        controller_df['Time_str'] = controller_df.get('Time_str_x').fillna(controller_df.get('Time_str_y'))
        controller_df.drop(columns=[c for c in ['Time_str_x', 'Time_str_y'] if c in controller_df.columns], inplace=True)

    return controller_df


def load_and_merge_to_data_df(data_file_path, monitor_file_path_list, settings_file_path_list, 
                              save_name='merged_df.pickle', save_directory='.', overwrite=False, keep_every = 1):
    """
    Load and merge monitor and settings data from CSV files, and combine with waveform data. 
    Drops NaN values and saves the merged DataFrame to a pickle file.
    
    Parameters
    ----------
    monitor_file_path_list : list of str
        Paths to the monitor CSV files
    settings_file_path_list : list of str
        Paths to the settings CSV files
    data_file_path : str
        Path to the data pickle file
    save_name : str, optional
        Name of the pickle file to save (default: 'merged_df.pickle')
    save_directory : str, optional
        Directory to save the pickle file (default: '.')
    overwrite : bool, optional
        Whether to overwrite existing pickle file (default: False)
    keep_every : int, optional
        Keep every nth row from waveform data (default: 1)
    
    Returns
    -------
    merged_df : pd.DataFrame
        The merged DataFrame containing waveform and controller data.
    """
    # check if the merged pickle file already exists. If it does, print an error message and exit early
    if os.path.isfile(f'{save_directory}/{save_name}'):
        if not overwrite:
            print(f"File already exists. Load the existing pickle file instead: {save_directory}/{save_name}")
            return pd.read_pickle(f'{save_directory}/{save_name}')
        
    print('loading waveform...')
    data = pj.loadPickle(data_file_path)
    measurements = {k:v for k,v in data.items() if isinstance(k, int)}
    waveform_df = pd.DataFrame(measurements).transpose() 
    waveform_df.drop(columns=['time'], inplace=True)

    # Convert time to float 
    waveform_df['time_collected'] = waveform_df['time_collected'].astype(float)
    waveform_df = waveform_df.sort_values('time_collected').reset_index(drop=True)
    waveform_df = waveform_df.iloc[::keep_every]

    controller_df = load_merge_to_controller_df(monitor_file_path_list, settings_file_path_list,
                                               monitor_cols = [ 'Time', 'Milliseconds',
                                                            '1000.1: CH1 Object', '1000.2: CH2 Object', 
                                                            '1001.1: CH1 Sink', '1001.2: CH2 Sink',
                                                            '1045.1: HR1 Temp', '1045.2: HR2 Temp',
                                                            '1044.1: LR1 Temp', '1044.2: LR2 Temp', '1044.3: LR3 Temp',
                                                            '3000.1: CH1 Target', '3000.2: CH2 Target' ],
                                               settings_cols = [ 'Time', 'Milliseconds',
                                                            '2010.1: Output Enable', '2010.2: Output Enable', 
                                                            '3034.1: Peltier Polarity', '3034.2: Peltier Polarity',
                                                            '1200.1: Temperature is Stable','1200.2: Temperature is Stable', ], 
                                               )

    print('merging waveform and controller data...')
    merged_df = pd.merge_asof(
                            left=waveform_df,
                            right=controller_df,
                            left_on='time_collected',
                            right_on='Time',
                            direction='nearest' 
                            )

    ## reformat time Time_str to '%m/%d/%y %H:%M:%S'
    merged_df['Time_str'] = pd.to_datetime(merged_df['Time_str']).dt.strftime('%m/%d/%y %H:%M:%S')
    
    ##get rid of nans
    merged_df = merged_df.dropna()
    
    ## write metadata
    merged_df.attrs['time'] = data[0]['time']
    merged_df.attrs['freq'] = np.fft.fftshift( np.fft.fftfreq( len(merged_df.attrs['time']), d=merged_df.attrs['time'][1] - merged_df.attrs['time'][0]))

    merged_df.attrs['save_directory'], merged_df.attrs['save_name'] = save_directory, save_name
    
    print('saving merged dataframe to pickle...')
    merged_df.to_pickle(f'{save_directory}/{save_name}')

    print('complete')
    return merged_df


def load_and_merge_all_controller_columns(merged_filtered_df, monitor_file_path_list, settings_file_path_list):
    mask = merged_filtered_df.map(lambda x: isinstance(x, (list, np.ndarray, tuple))).any(axis=0) # arrays
    monitor_df = merged_filtered_df.drop(mask.index[mask], axis=1).copy() # drop arrays
    controller_df = load_merge_to_controller_df(monitor_file_path_list, settings_file_path_list)

    common_cols = set(monitor_df.columns) & set(controller_df.columns) - {'Time', 'time_collected'}
    
    full_monitor_df = pd.merge_asof(
        left=monitor_df,
        right=controller_df.drop(columns=common_cols),
        left_on='time_collected',
        right_on='Time',
        direction='nearest',
        )

    # non unique columns aren't informative
    non_unique_cols = [c for c in full_monitor_df.columns if len(full_monitor_df[c].unique()) == 1] 
    full_monitor_df.drop(columns=non_unique_cols, inplace=True)

    time_str_cols = [c for c in ['Time_str', 'Time_str_x', 'Time_str_y'] if c in full_monitor_df.columns]
    if time_str_cols:
        full_monitor_df.drop(columns=time_str_cols, inplace=True)  # drop non-numeric time

    # drop columns that contain any NaN values
    full_monitor_df.dropna(axis=1, inplace=True)

    # mask = full_monitor_df.map(lambda x: isinstance(x, (str))).any(axis=0)
    # str_df = full_monitor_df.loc[:, mask]
    
    
    
    return full_monitor_df



def preprocess_worker(x, offset, sos, correct_gain=True, butterworth_filter=True, n1_1_scaling=False, bkg_subtraction=False):
    if correct_gain: x = pj.correctVoltageByGain(x, offset)
    if butterworth_filter: x = sosfiltfilt(sos, x)
    if n1_1_scaling: x = x / np.max(np.abs(x))
    if bkg_subtraction: x = x - np.mean(x[:50])
    return x

def preprocess_merged_df(merged_df, correct_gain=True, butterworth_filter=True, n1_1_scaling=False, bkg_subtraction=False, save_to_pickle=True):
    '''
    Preprocess the merged dataframe in place.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        The merged DataFrame containing waveform and controller data
    correct_gain : bool, optional
        Whether to apply gain correction (default is True)
    butterworth_filter : bool, optional
        Whether to apply a Butterworth high-pass filter (default is True)
    n1_1_scaling : bool, optional
        Whether to normalize the voltage to [-1, 1] (default is False)
    bkg_subtraction : bool, optional
        Whether to subtract the background noise (default is False)
        
    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame
    '''
    if merged_df.attrs.get('preprocessed', False):
        print("Warning: The merged dataframe has already been preprocessed. Skipping preprocessing.")
        return merged_df
    
    keys = merged_df.columns[np.where(merged_df.columns.str.startswith('voltage_'))]
    offset_keys = ['voltageOffsetForward', 'voltageOffsetForward', 'voltageOffsetReverse', 'voltageOffsetReverse']
    sos = butter(5, 1000000, btype = 'highpass', analog = False, fs = 500000000, output = 'sos')
    
    for i, k in tqdm(enumerate(keys), total=len(keys), desc='Preprocessing waveforms'):
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:        
            result = list( executor.map(preprocess_worker, merged_df[k], merged_df[offset_keys[i]]/10, itertools.repeat(sos), itertools.repeat(correct_gain), itertools.repeat(butterworth_filter), itertools.repeat(n1_1_scaling), itertools.repeat(bkg_subtraction)) )
            merged_df[k] = result
            
    merged_df.attrs['preprocessed'] = True
    
    if save_to_pickle:
        print('Saving preprocessed dataframe to pickle...')
        merged_df.to_pickle(f"{merged_df.attrs['save_directory']}/{merged_df.attrs['save_name']}")

    return merged_df


### calculations
## TODO: find a way to speed up and parallelize

def calculate_mean_diff(merged_df, overwrite=False, save_to_pickle=False):
    '''Calculate the mean and difference of temperatures from the merged dataframe.'''
    
    if merged_df.attrs.get('mean_diff_calculations', False) and not overwrite:
        print("Warning: The merged dataframe has already had mean and difference of temperatures calculated. Skipping calculation.")
        return merged_df
    
    merged_df['index'] = merged_df.index
    merged_df['mean_T (°C)'] = merged_df[['1000.1: CH1 Object', '1000.2: CH2 Object']].mean(axis=1)
    merged_df['diff_T (°C)'] = merged_df['1000.1: CH1 Object'] - merged_df['1000.2: CH2 Object']
    
    merged_df.attrs['mean_diff_calculations'] = True
    
    if save_to_pickle:
        print('Saving mean and difference of temperatures to pickle...')
        merged_df.to_pickle(f"{merged_df.attrs['save_directory']}/{merged_df.attrs['save_name']}")
    
    return merged_df


def feature_worker(x, t_):
    return {
        "amplitude": pj.maxMinusMin(x),
        "Hilbert_ToF": pj.envelopeThresholdTOF(x, t_, 0.15),
        "Hilbert_noise_ToF": pj.firstIndexAboveNoise(x),
        "max": max(abs(x)),
        "hilbert": pj.hilbertEnvelope(x),
    }
        

def calculate_waveform_features(merged_df, overwrite=False, save_to_pickle=False, pooled=False):
    '''Calculate waveform features from the merged dataframe.'''
    
    if merged_df.attrs.get('waveform_features_calculated', False) and not overwrite:
        print("Warning: The merged dataframe has already had waveform features calculated. Skipping calculation.")
        return merged_df
    
    keys = merged_df.columns[np.where(merged_df.columns.str.startswith('voltage_'))]
    t_ = merged_df.attrs['time']

    for k in tqdm(keys, total=len(keys), desc='Calculating waveform features'):
        if not pooled:
            results = list(map(feature_worker, merged_df[k], itertools.repeat(t_)))
        else:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(executor.map(feature_worker, merged_df[k], itertools.repeat(t_)))
            
        merged_df['amplitude_'+k+' (mV)'] = [r['amplitude'] for r in results]
        merged_df['Hilbert_ToF_'+k+' (ns)'] = [r['Hilbert_ToF'] for r in results]
        merged_df['Hilbert_noise_Tof_'+k+' (ns)'] = [r['Hilbert_noise_ToF'] for r in results]
        merged_df['max_'+k+' (mV)'] = [r['max'] for r in results]
        merged_df['Hilbert_window_'+k] = [r['hilbert'] for r in results]

    merged_df.attrs['waveform_features_calculated'] = True

    if save_to_pickle:
        print('Saving waveform features to pickle...')
        merged_df.to_pickle(f"{merged_df.attrs['save_directory']}/{merged_df.attrs['save_name']}")

    return merged_df


def fft_magnitude_worker(x, f_len):
    '''worker for calculating FFT and magnitude
    Parameters
    ----------
    x : array-like
        The input signal
    f_len : int
        The length of the frequency domain
    Returns
    -------
    dict
        A dictionary containing the FFT and magnitude
    '''
    fft_val = np.fft.fftshift(np.fft.fft(x))/ f_len
    magnitude = np.abs(fft_val) 
    return {
        "fft": fft_val,
        "magnitude": magnitude,
    }

def calculate_fft_magnitude_features(merged_df, overwrite=False, save_to_pickle=False, pooled=False):
    '''Calculate fft and fft magnitude features from the merged dataframe.'''

    if merged_df.attrs.get('fft_magnitude_features_calculated', False) and not overwrite:
        print("Warning: The merged dataframe has already had fft magnitude features calculated. Skipping calculation.")
        return merged_df

    keys = merged_df.columns[np.where(merged_df.columns.str.startswith('voltage_'))]
    f_len = len(merged_df.attrs['time'])

    for k in tqdm(keys, desc='Calculating FFT and magnitude'):
        if not pooled: results = list(map(fft_magnitude_worker, merged_df[k], itertools.repeat(f_len)))
        else:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(executor.map(fft_magnitude_worker, merged_df[k], itertools.repeat(f_len)))

        merged_df['fft_'+k] = [r['fft'] for r in results]
        merged_df['fft_magnitude_'+k] = [r['magnitude'] for r in results]

    merged_df.attrs['fft_magnitude_features_calculated'] = True

    if save_to_pickle:
        print('Saving waveform features to pickle...')
        merged_df.to_pickle(f"{merged_df.attrs['save_directory']}/{merged_df.attrs['save_name']}")

    return merged_df


def phase_group_delay_worker(x, f_):
    '''worker for calculating phase and group delay
    Parameters
    ----------
    x : array-like
        The input signal
    f_ : array-like
        The frequency domain
    Returns
    -------
    dict
        A dictionary containing the phase and group delay
    '''
    phase = np.unwrap(np.angle(x)) #TODO: why unwrap with period of pi an not 2pi?
    group_delay = -np.gradient(phase, f_) / 2/np.pi
    return {
        "phase": phase,
        "group_delay": group_delay,
    }

def calculate_phase_group_delay_features(merged_df, overwrite=False, save_to_pickle=False, pooled=False):
    '''Calculate fft phase and group delay features from the merged dataframe.'''

    if merged_df.attrs.get('phase_group_delay_features_calculated', False) and not overwrite:
        print("Warning: The merged dataframe has already had frequency phase and group delay features calculated. Skipping calculation.")
        return merged_df

    keys = merged_df.columns[np.where(merged_df.columns.str.startswith('voltage_'))]
    f_ = merged_df.attrs['freq']

    for k in tqdm(keys, desc='Calculating phase and group delay'):
        if not pooled: 
            results = list(map(phase_group_delay_worker, merged_df['fft_'+k], itertools.repeat(f_)))
        else:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(executor.map(phase_group_delay_worker, merged_df['fft_'+k], itertools.repeat(f_)))
                
        merged_df['fft_phase_'+k] = [r['phase'] for r in results]
        merged_df['fft_group_delay_'+k] = [r['group_delay'] for r in results]

    merged_df.attrs['phase_group_delay_features_calculated'] = True

    if save_to_pickle:
        print('Saving waveform features to pickle...')
        merged_df.to_pickle(f"{merged_df.attrs['save_directory']}/{merged_df.attrs['save_name']}")
    
    return merged_df


    
# TODO: test
def detrend_linear(df, keys=None):
    """
    Detrend amplitude and ToF data using polynomial and logarithmic fits.
    
    Parameters:
    - df: pandas.DataFrame, the filtered dataframe to detrend
    - amplitude_keys: list of str, column names for amplitude data to detrend (default: forward and reverse)    
    - tof_keys: list of str, column names for ToF data to detrend (default: Hilbert and Simple ToF)
    Returns:
    - df: pandas.DataFrame, the detrended dataframe (modifies in place)
    """
    if amplitude_keys is None:  amplitude_keys = ['amplitude_forward (mV)', 'amplitude_reverse (mV)']
    
    # Detrend amplitude with linear fit
    for k in amplitude_keys:
        a, b = np.polyfit(df['time_collected'], df[k], 1)
        fit = a * df['time_collected'] + b
        df[k] -= fit

# TODO: test
def detrend_log(df, amplitude_keys=None, tof_keys=None):
    """
    Detrend amplitude and ToF data using polynomial and logarithmic fits.
    
    Parameters:
    - df: pandas.DataFrame, the filtered dataframe to detrend
    - tof_keys: list of str, column names for ToF data to detrend (default: Hilbert and Simple ToF)
    
    Returns:
    - df: pandas.DataFrame, the detrended dataframe (modifies in place)
    """
    if tof_keys is None: tof_keys = ['Hilbert_ToF_forward (ns)', 'Hilbert_ToF_reverse (ns)']
    # Detrend ToF with logarithmic fit
    def log_func(t_rel, a, b): 
        return a + b * np.log(t_rel)
    
    t = df['time_collected'].to_numpy(dtype=float)
    t0 = t.min()
    t_rel = t - t0 + 1.0  # strictly > 0 for log
    
    for k in tof_keys:
        y = df[k].to_numpy(dtype=float)
        popt, _ = curve_fit(log_func, t_rel, y, maxfev=20000)
        fit = log_func(t_rel, *popt)
        df[k] -= fit
    
    return df


def calculate_avgs_by_T(merged_filtered_df, temps, crop=None, polyfit_degree=1,
                        keys_=['amplitude_forward (mV)', 
                                'amplitude_reverse (mV)', 
                                'Hilbert_ToF_forward (ns)', 
                                'Hilbert_ToF_reverse (ns)'], ):
    avgs = {k: [] for k in keys_}
    errs = {k: [] for k in keys_}
    params, fits, residuals = {}, {}, {}
    
    T_fits_df = pd.DataFrame({'calculated': keys_,
                              'avg': [[] for _ in keys_],
                              'err': [[] for _ in keys_],
                              'params': [[] for _ in keys_],
                              'fits': [[] for _ in keys_],
                              'residuals': [[] for _ in keys_]}).set_index('calculated')
                             
    T_fits_df.attrs = {'dT': [abs(t[0] - t[1]) for t in temps],
                       'T_avg': [np.mean(t) for t in temps]}
    
    for k in keys_:
        for t in temps:
            mask = ( (merged_filtered_df['3000.1: CH1 Target'] == t[0])
                    & (merged_filtered_df['3000.2: CH2 Target'] == t[1])
                    & (merged_filtered_df['2010.1: Output Enable'] == 'ON')
                    & (merged_filtered_df['2010.2: Output Enable'] == 'ON') )    
            if crop: mask+=( (merged_filtered_df.index >= crop[0]) & (merged_filtered_df.index < crop[1]) )
                
            if mask.sum() == 0: 
                print(f"No data found for temperature {t}. Skipping this temperature.")
                continue
            
            filtered_df = merged_filtered_df[mask]
            T_fits_df.loc[k,'avg'].append(filtered_df[k].mean())
            T_fits_df.loc[k,'err'].append(filtered_df[k].std())
            
        T_fits_df.loc[k,'params'] = np.polyfit(T_fits_df.attrs['dT'], T_fits_df.loc[k,'avg'], polyfit_degree)
        T_fits_df.loc[k,'fits'] = np.polyval(T_fits_df.loc[k,'params'], T_fits_df.attrs['dT'])
        T_fits_df.loc[k,'residuals'] = T_fits_df.loc[k,'avg'] - T_fits_df.loc[k,'fits']
        
    return T_fits_df



### plotting functions
# TODO
def plot_detrended_amplitude(df,):
    """
    Plot the detrended amplitude of the merged dataframe.
    
    Parameters:
    - merged_df: pd.DataFrame, the merged DataFrame containing waveform and controller data
    """
    # before/ater detrending amplitude
    k = 'amplitude_forward_normalized'
    a,b = np.polyfit(df['time_collected'], df[k], 1)
    fit = a * df['time_collected'] + b
    t_rel = df['time_collected'] - df['time_collected'].iloc[0]

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
        
    ax1.plot(t_rel, df[k], label='data', alpha=0.8, color='blue', lw=1)
    ax1.plot(t_rel, fit, label=f'{a:.3f} + {b:.3f} * t', lw=1, linestyle='--', color='orange')
    ax1.set_xlabel('Time since start (s)')
    ax1.set_ylabel('Amplitude (ns)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2.plot(t_rel, df[k] - fit, label='detrended', alpha=0.8, color='green')
    ax2.set_ylabel('Detrended Amplitude (ns)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(k)
    plt.tight_layout()
    plt.show()

# TODO
def plot_detrended_tof(merged_df):
    """
    Plot the detrended ToF of the merged dataframe.
    
    Parameters:
    - merged_df: pd.DataFrame, the merged DataFrame containing waveform and controller data
    """
    df = merged_df.iloc[1500:].copy()
    k = 'Hilbert_ToF_forward (ns)'

    # use relative time to keep log-fit numerically stable
    t = df['time_collected'].to_numpy(dtype=float)
    y = df[k].to_numpy(dtype=float)
    t0 = t.min()
    t_rel = t - t0 + 1.0  # strictly > 0 for log

    def log_func(t_rel, a, b): return a + b * np.log(t_rel)

    popt, _ = curve_fit(log_func, t_rel, y, maxfev=20000)
    fit = log_func(t_rel, *popt)

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(t_rel, y, label='data', alpha=0.8, color='blue', lw=1)
    ax1.plot(t_rel, fit, label=f'{popt[0]:.3f} + {popt[1]:.3f} * ln(t)', lw=1, linestyle='--', color='orange')
    ax1.set_xlabel('Time since start (s)')
    ax1.set_ylabel('ToF (ns)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2.plot(t_rel, y - fit, label='detrended', alpha=0.8, color='green')
    ax2.set_ylabel('Detrended ToF (ns)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(k)
    plt.tight_layout()
    plt.show()


def waveform_interact_decorator(func):
    def wrapper(*args, **kwargs):
        merged_df = kwargs.pop('merged_df', None)
        if merged_df is None: raise ValueError("merged_df is required")
        max_ = kwargs.pop('max_', None)
        
        slider = widgets.IntSlider(value=0,
                                    min=0, max=len(merged_df)-1,
                                    step=1,
                                    description='measurement:',
                                    continuous_update=False
                                )
        function = lambda i: func(i, merged_df, max_, **kwargs)
        interact(function, i=slider)
    return wrapper

@waveform_interact_decorator
def plot_waveform(i, merged_df=None, max_=None, keys=None):
    if not keys:
        keys = merged_df.columns[np.where(merged_df.columns.str.startswith('voltage_'))]
    
    fig,ax = plt.subplots(figsize=(8, 4))
            
    for key_ in keys:
        ax.plot(merged_df.attrs['time'], merged_df[key_][i], lw=1, label=key_)
    
    ax.set_title(f'{i}: ({merged_df["Time_str"][i]}), (F, R)=({merged_df["1000.1: CH1 Object"][i]:.2f}, {merged_df["1000.2: CH2 Object"][i]:.2f})ºC ')
    ax.set_ylim(-max_, max_)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Voltage (mV)')
    
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
    fig.tight_layout()
    
@waveform_interact_decorator
def plot_hilbert(i, merged_df=None, max_=None):
     keys = merged_df.columns[np.where(merged_df.columns.str.startswith('voltage_'))]
     
     fig,ax = plt.subplots(figsize=(8, 4))
             
     for key_ in keys:
         ax.plot(merged_df.attrs['time'], merged_df[key_][i], lw=1, label=key_)
         ax.plot(merged_df.attrs['time'], merged_df['Hilbert_window_'+key_][i], lw=0.5, linestyle='--', color=ax.lines[-1].get_color())
     
     ax.set_title(f'{i}: ({merged_df["Time_str"][i]}), (F, R)=({merged_df["1000.1: CH1 Object"][i]:.2f}, {merged_df["1000.2: CH2 Object"][i]:.2f})ºC ')
     ax.set_ylim(-max_, max_*1.1)
     ax.set_xlabel('Time (ns)')
     ax.set_ylabel('Voltage (mV)')
     
     fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
     fig.tight_layout()
 
    
@waveform_interact_decorator
def plot_re_im_fft(i, merged_df=None, max_=None, freq_crop=None):
    keys = merged_df.columns[np.where(merged_df.columns.str.startswith('voltage_'))]
    # keys = ['voltage_echo_reverse']

    fig,ax = plt.subplots(4, figsize=(6,10))
    fig.suptitle(f'{i}: ({merged_df["Time_str"][i]}), (F, R)=({merged_df["1000.1: CH1 Object"][i]:.2f}, {merged_df["1000.2: CH2 Object"][i]:.2f})ºC ')
    if freq_crop is None: freq_crop = [0, len(merged_df.attrs['freq'])]

    for ind,k in enumerate(keys):
        fft_dat = merged_df['fft_'+k][i][freq_crop[0]:freq_crop[1]]
        ax[ind].plot(merged_df.attrs['freq'][freq_crop[0]:freq_crop[1]], fft_dat.real, lw=1)
        ax[ind].plot(merged_df.attrs['freq'][freq_crop[0]:freq_crop[1]], fft_dat.imag, lw=0.5, linestyle='-.', color=ax[ind].lines[-1].get_color())
    
        ax[ind].set_title('fft_'+k)
        
    ax[ind].set_xlabel('frequency (1/ns)')
    
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
    fig.tight_layout()
     
@waveform_interact_decorator
def plot_waveform_and_fft(i, merged_df=None, max_=None, freq_crop=None, return_figure=False):
    keys = merged_df.columns[np.where(merged_df.columns.str.startswith('voltage_'))]
    
    fig,ax = plt.subplots(2,2,figsize=(12, 8))
    ax = ax.flatten()
    fig.suptitle(f'{i}: ({merged_df["Time_str"][i]}), (F, R)=({merged_df["1000.1: CH1 Object"][i]:.2f}, {merged_df["1000.2: CH2 Object"][i]:.2f})ºC ')
    if freq_crop is None: freq_crop = [0, len(merged_df.attrs['freq'])]
    
    for k in keys:
        ax[0].plot(merged_df.attrs['time'], merged_df[k][i], label=k, lw=1)
        ax[0].plot(merged_df.attrs['time'], merged_df['Hilbert_window_'+k][i], lw=0.5, linestyle='--', color=ax[0].lines[-1].get_color())
        ax[1].plot(merged_df.attrs['freq'][freq_crop[0]:freq_crop[1]], merged_df['fft_magnitude_'+k][i][freq_crop[0]:freq_crop[1]], lw=1)
        ax[2].plot(merged_df.attrs['freq'][freq_crop[0]:freq_crop[1]], merged_df['fft_phase_'+k][i][freq_crop[0]:freq_crop[1]], lw=1)
        ax[3].plot(merged_df.attrs['freq'][freq_crop[0]:freq_crop[1]], merged_df['fft_group_delay_'+k][i][freq_crop[0]:freq_crop[1]], lw=1)
        
    ax[0].set_title('waveform, Hilbert win')
    ax[0].set_xlabel('time (ns)')
    ax[0].set_ylabel('amplitude')
    ax[0].set_ylim(-max_, max_)

    ax[1].set_title('fft')
    ax[1].set_xlabel('frequency (1/ns)')
    ax[1].set_ylabel('magnitude')
    
    ax[2].set_title('phase')
    ax[2].set_xlabel('frequency (1/ns)')
    ax[2].set_ylabel('phase (rad)')

    ax[3].set_title('group delay')
    ax[3].set_xlabel('frequency (1/ns)')
    ax[3].set_ylabel('group delay (ns)')

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.03), ncol=2)
    fig.tight_layout()
    
    if return_figure: return fig, ax
     
 
def experimental_data_interact_decorator(func):
    def wrapper(*args, **kwargs):
        df = kwargs.get('df', None)
        if df is None:  raise ValueError("merged_df is required")
        idx_label_type = kwargs.get('idx_label_type', '')
        x_scaling = kwargs.get('x_scaling', 'collection_time')
        temp_values = df[['1000.1: CH1 Object', '1000.2: CH2 Object', '1044.3: LR3 Temp']].values
        ylims_ = (temp_values.min() - 5, temp_values.max() + 5)
        slider = widgets.IntSlider(
            value=0,
            min=0,
            max=len(df)-1,
            step=1,
            description='measurement:',
            continuous_update=False,
            layout=widgets.Layout(width='800px')  # or '100%'
        )
        selected_keys = [col for col in df.columns if col.startswith(('Hilbert', 'amplitude'))]
        selected_keys.sort()
        keys = widgets.SelectMultiple(
            options=selected_keys,
            value=tuple([]),
            description='Variable:',
            disabled=False,
        )
        function = lambda i, keys_: func(i=i, keys_=list(keys_), df=df, ylims_=ylims_, idx_label_type=idx_label_type, x_scaling=x_scaling)
        interact(function, i=slider, keys_=keys)
    return wrapper

@experimental_data_interact_decorator
def plot_experimental_data(i=0, keys_=(), df=None, idx_label_type='', x_scaling='collection_time',ylims_=(0, 60),
                           return_figure=False,
                           ):
    """Plot temperature data with optional secondary variables.

    Parameters
    ----------
    i : int, optional
        Index of the measurement to highlight.
    df : pd.DataFrame, optional
        The DataFrame containing the temperature data.
    keys_ : tuple or list of str, optional
        Additional columns to plot on a secondary y-axis.
    idx_label_type : str, optional
        Type of x-axis labels to use ('time', 'dT', 'T').
    x_scaling : str, optional
        Type of x-axis scaling to use ('collection_time', 'index').
    """
    xticks_labels = df['index'].astype(str) + ': ' + df['Time_str']
    if x_scaling=='collection_time':
        xticks = df['time_collected'].values
    else:
        xticks = df['index'].values

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # plot temperature data vs ΔT
    ax1.plot(xticks, df['mean_T (°C)'], label='mean', marker='o', linestyle='-', linewidth=0.01, markersize=2)
    lower_bound = df['1000.1: CH1 Object']
    ax1.plot(xticks, lower_bound, label='transmitter side', linestyle='--', linewidth=0.01, marker='o', markersize=2)
    upper_bound = df['1000.2: CH2 Object']
    ax1.plot(xticks, df['1000.2: CH2 Object'], label='receiver side', linestyle='--', linewidth=0.01, marker='o', markersize=2)
    ax1.fill_between(xticks, df['1000.1: CH1 Object'], df['1000.2: CH2 Object'], alpha=0.25)
    ax1.axvline(xticks[i], color='red', linestyle='--', linewidth=0.8, alpha=0.4)
    ax1.plot(xticks, df['1044.3: LR3 Temp'], label='chamber', marker='o', linestyle='-', linewidth=0.01, markersize=1)
    ax1.plot(xticks, df['1001.1: CH1 Sink'], label='trans sink', marker='o', linestyle='-', linewidth=0.01, markersize=1)
    ax1.plot(xticks, df['1001.2: CH2 Sink'], label='rec. sink', marker='o', linestyle='-', linewidth=0.01, markersize=1)

    # plot secondary keys
    if len(keys_) > 0:
        ax2 = ax1.twinx()  # second y-axis sharing x-axis
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        start_idx = 6
        for idx, key_ in enumerate(keys_):
            ax2.scatter(xticks, df[key_], label=key_, s=10, color=colors[start_idx + idx])
        ax2.set_ylabel('\n'.join([f'{key_}' for key_ in keys_]))
        
    # formatting
    ax1.set_title('Temperature gradient vs collection time')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_ylim(*ylims_)
    if idx_label_type == 'time':
        ax1.set_xticks(ticks=xticks[::max(1, len(xticks_labels)//10)] , 
                       labels=xticks_labels[::max(1, len(xticks_labels)//10)], 
                       rotation=45, ha='right')
        ax1.set_xlabel('Collection idx: Time')
        
    elif idx_label_type == 'dT':
        x = (df['3000.1: CH1 Target'] - df['3000.2: CH2 Target']).values
        t_change_idx = np.flatnonzero(np.diff(x, prepend=x[0]) != 0).tolist()
        diffs = np.diff(t_change_idx, prepend=t_change_idx[0])
        for j in range(len(t_change_idx) - 1, 0, -1):
            if diffs[j] < 20: t_change_idx.pop(j)
        t_change_idx = np.asarray(t_change_idx, dtype=int)
        
        # for idx in t_change_idx: # plot temp changes
        ax1.set_xticks(ticks=xticks[t_change_idx][::2], 
                       labels=x[t_change_idx][::2], rotation=45, ha='right')
        ax1.set_xlabel('Target ΔT (°C)')
        for idx in t_change_idx: ax1.axvline(xticks[idx], color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
        
    elif idx_label_type == 'T':
        x = df['3000.1: CH1 Target'].values
        t_change_idx = np.flatnonzero(np.diff(x, prepend=x[0]) != 0).tolist()
        diffs = np.diff(t_change_idx, prepend=t_change_idx[0])
        for j in range(len(t_change_idx) - 1, 0, -1):
            if diffs[j] < 30: t_change_idx.pop(j)
        t_change_idx = np.asarray(t_change_idx, dtype=int)
        ax1.set_xticks(ticks=xticks[t_change_idx], 
                       labels=x[t_change_idx], rotation=45, ha='right')
        ax1.set_xlabel('Target Temperature (°C)')
        for idx in t_change_idx: ax1.axvline(xticks[idx], color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
        
    fig.legend(bbox_to_anchor=(0.5, 0.03), loc='upper center', ncols=min(max(len(keys_)//2+2,1), 3))
    fig.tight_layout()

    if return_figure: return fig, ax1


def plot_wave_fft_experimental_data(i=0, keys_=(), df=None, idx_label_type='', x_scaling='collection_time',ylims_=(0, 60),
                           return_figure=False,
                           ): ## 
    pass

def format_polyfit_params(T_fits_df, k, t_type):
    if T_fits_df.loc[k,"params"].shape[0] == 1:
        return f'{T_fits_df.loc[k,"params"][-1]:.2f}'
    if T_fits_df.loc[k,"params"].shape[0] > 1:
        fstring_list = [ f'{T_fits_df.loc[k,"params"][-1]:.2f}', f'{T_fits_df.loc[k,"params"][-2]:.2f}{t_type}' ]
    for i,p in enumerate(T_fits_df.loc[k,"params"][:-2]):
        fstring_list.append(f'{p:.2f}{t_type}^{len(T_fits_df.loc[k,"params"])-i-1}')
    
    direction =  k.split(' ')[0].split('_')[-1]
    
    return direction + ': '+ ' + '.join(fstring_list)

def plot_avg_fits(T_fits_df, suptitle, t_type, keys, title, y_label):
    s = T_fits_df.shape
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    fig.suptitle(suptitle)

    for k in keys:
        ax[0].plot(T_fits_df.attrs[t_type], T_fits_df.loc[k,'fits'], 
                linewidth=1, linestyle='--', label=format_polyfit_params(T_fits_df, k, t_type))
        ax[0].errorbar(T_fits_df.attrs[t_type], T_fits_df.loc[k,'avg'], yerr=T_fits_df.loc[k,'err'], 
                    fmt='o', markersize=5, capsize=3, label=f'1 StDv', color=ax[0].lines[-1].get_color(), ecolor=ax[0].lines[-1].get_color())


        ax[1].plot(T_fits_df.attrs[t_type], T_fits_df.loc[k, 'residuals'], linewidth=1, linestyle='--')
        ax[1].errorbar(T_fits_df.attrs[t_type], T_fits_df.loc[k, 'residuals'], yerr=T_fits_df.loc[k, 'err'], 
                       fmt='o', markersize=5, color=ax[0].lines[-1].get_color(), ecolor=ax[0].lines[-1].get_color(), capsize=3)

        # ax[1].plot(T_fits_df.attrs[t_type], T_fits_df.loc[k_forward.replace('forward', 'reverse'), 'residuals'], color='green', linewidth=1, linestyle='--',)
        # ax[1].errorbar(T_fits_df.attrs[t_type], T_fits_df.loc[k_forward.replace('forward', 'reverse'), 'residuals'], yerr=T_fits_df.loc[k_forward.replace('forward', 'reverse'), 'err'], fmt='s', markersize=5, color='green', ecolor='green', capsize=3,)

    ax[0].set_title(title)
    ax[0].set_xlabel(t_type+' (ºC)')
    ax[0].set_ylabel(y_label)
    
    ax[1].set_title('Residuals')
    ax[1].set_xlabel(t_type+' (ºC)')
    ax[1].set_ylabel('Residuals')

    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.14), ncol=2)

    fig.tight_layout()    
    

def plot_overall_waveforms(merged_filtered_df, temps, key, crop=None, t_type='dT'):
    '''
    Plot overall waveforms for a given key and temperature range.

    Parameters:
    merged_filtered_df (pd.DataFrame): The filtered DataFrame.
    temps (list): A list of temperature tuples.
    key (str): The key to plot.
    crop (tuple, optional): A tuple of start and end indices to crop the data.
    t_type (str, optional): The type of temperature to display. Defaults to 'dT'. options: ['T', 'dT', 'Both']

    Returns:
    None
    
    '''
    
    if not crop or len(crop) != 2:
        crop = (0, len(merged_filtered_df.attrs['time']))
        
    fig, ax = plt.subplots(figsize=(7.5, 4))
    
    for i, t in enumerate(temps):
        mask = ( (merged_filtered_df['3000.1: CH1 Target'] == t[0])
                & (merged_filtered_df['3000.2: CH2 Target'] == t[1])
                & (merged_filtered_df['2010.1: Output Enable'] == 'ON')
                & (merged_filtered_df['2010.2: Output Enable'] == 'ON') )
            
        if mask.sum() == 0: 
            print(f"No data found for temperature {t}. Skipping this temperature.")
            continue
        
        temp_df = merged_filtered_df.loc[mask].iloc[::50]
        
        for idx, row in temp_df.iterrows():
            ax.plot(
                merged_filtered_df.attrs['time'][crop[0]:crop[1]],
                row[key][crop[0]:crop[1]],
                color=cm.viridis(i / (len(temps)-1)),
                linewidth=0.5,
                alpha=0.7,
                label=f'20 +/-ΔT' if idx == temp_df.index[-1] else None
            )
            
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Voltage (mV)')
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    
    if t_type == 'dT':
        ax.set_title(f'{key} 20 +/-ΔT')
        cax1 = fig.add_axes([0.81, 0.15, 0.02, 0.7])
        sm1 = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=merged_filtered_df['diff_T (°C)'].min(), 
                                                                        vmax=merged_filtered_df['diff_T (°C)'].max()))
        cbar1 = fig.colorbar(sm1, cax=cax1)
        cbar1.ax.set_title('ΔT', pad=10)
    elif t_type == 'T':
        ax.set_title(f'{key}')
        cax1 = fig.add_axes([0.81, 0.15, 0.02, 0.7])
        sm1 = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=merged_filtered_df['mean_T (°C)'].min(), 
                                                                        vmax=merged_filtered_df['mean_T (°C)'].max()))
        cbar1 = fig.colorbar(sm1, cax=cax1)
        cbar1.ax.set_title('T', pad=10)
    elif t_type == 'Both':
        cax1 = fig.add_axes([0.81, 0.15, 0.02, 0.7])
        sm1 = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=merged_filtered_df['diff_T (°C)'].min(), 
                                                                        vmax=merged_filtered_df['diff_T (°C)'].max()))
        cbar1 = fig.colorbar(sm1, cax=cax1)
        cbar1.ax.set_title('ΔT', pad=10)

        cax2 = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        sm2 = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=merged_filtered_df['mean_T (°C)'].min(), 
                                                                        vmax=merged_filtered_df['mean_T (°C)'].max()))
        cbar2 = fig.colorbar(sm2, cax=cax2)
        cbar2.ax.set_title('T', pad=10)    
    
    
## PCA and stat analysis
    
def get_pca_loadings(pca):
    '''get loadings from PCA object and return as a dataframe with feature names as index and PC numbers as columns.'''
    return pd.DataFrame(
                pca.components_.T, # eigenvectors
                index=pca.feature_names_in_,
                columns=[f"PC: {i+1}" for i, val in enumerate(pca.explained_variance_)] # enum eigenvalues
            ) 

def get_top_loadings(loadings_df, top_n=10):
    '''Get the top n loadings for each principal component.'''
    top_loadings = {}
    for pc in loadings_df.columns:
        top_loadings[pc] = loadings_df[pc].abs().nlargest(top_n).index.tolist()
        
    return pd.DataFrame(top_loadings, 
                        index=[f"Rank {i+1}" for i in range(10)])


def plot_pca_scree(X_scaled, components=20):
    '''Plot the explained variance and cumulative explained variance of PCA components.'''
    pca = PCA(n_components=components)
    transformed = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(4,4))
    x = np.arange(1, components + 1)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    ax.plot(x, explained, marker='o', label='explained variance of component')
    for xi, ex in zip(x, explained):  ax.text(xi, ex + 0.02, f'{ex:.2%}', ha='center', va='bottom', rotation=90, fontsize=8)

    ax.plot(x, cumulative, marker='s', linestyle='--', label='cumulative explained variance')
    for xi, cum in zip(x, cumulative):  ax.text(xi, cum + 0.02, f'{cum:.2%}', ha='center', va='bottom', rotation=90, fontsize=8)

    ax.set_title('Explained Variance of Principal Components')
    ax.set_xlabel('Principal Component Number')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_ylim(0, 1.25)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.legend(loc='lower center', ncol=1, bbox_to_anchor=(0.5, -0.2), frameon=False)   
        
def plotly_loadings_heatmap(loadings_df):
    fig = go.Figure(
        data=go.Heatmap(
            z=loadings_df.to_numpy(),
            x=loadings_df.columns,
            y=loadings_df.index,
            colorscale="RdBu_r",
            zmid=0,
            hovertemplate=(
                "Feature: %{y}<br>"
                "PC: %{x}<br>"
                "Loading: %{z:.3f}<extra></extra>"
            ),
        )
    );

    fig.update_layout(
        title="PCA Loadings Heatmap",
        xaxis_title="Principal component",
        yaxis_title="Feature",
        template="plotly_white",
        height=1200,
        width=800,
    );

    fig.update_xaxes(side="top");
    fig.update_yaxes(autorange="reversed");
    return fig    


def plot_pc_scores_by_rank(top_10_df, transformed, full_monitor_df, pc_x=1, pc_y=2, color_pc=None, rank=1, cmap='viridis', s=12, alpha=0.75):
    """
    Scatter plot of PCA-transformed samples.

    Parameters
    ----------
    pc_x : int
        Principal component to use on the x-axis (1-based index).
    pc_y : int
        Principal component to use on the y-axis (1-based index).
    color_pc : int or None
        PCA column used to select the feature whose values color the points.
        If None, uses pc_x.
    rank : int
        Rank within the selected PC column (1..10) to choose which feature to color by.
    """
    if pc_x == pc_y: raise ValueError("pc_x and pc_y must be different.")
    if not (1 <= pc_x <= transformed.shape[1]) or not (1 <= pc_y <= transformed.shape[1]): raise ValueError(f"pc_x and pc_y must be in [1, {transformed.shape[1]}].")
    if color_pc is None:  color_pc = pc_x
    if not (1 <= color_pc <= transformed.shape[1]): raise ValueError(f"color_pc must be in [1, {transformed.shape[1]}].")
    if not (1 <= rank <= 10): raise ValueError("rank must be between 1 and 10.")

    pc_x_name = f"PC: {pc_x}"
    pc_y_name = f"PC: {pc_y}"
    color_col = f"PC: {color_pc}"
    

    feature = top_10_df.loc[f"Rank {rank}", color_col]
    if feature not in transformed.columns: raise ValueError(f"Feature '{feature}' not found in transformed.columns.")

    x_vals = transformed[:, pc_x - 1]
    y_vals = transformed[:, pc_y - 1]
    color_vals = full_monitor_df[feature].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        x_vals,
        y_vals,
        c=color_vals,
        cmap=cmap,
        s=s,
        alpha=alpha,
        edgecolors='none'
    )

    ax.set_xlabel(pc_x_name)
    ax.set_ylabel(pc_y_name)
    ax.set_title(f"Colors: '{feature}' ({color_col}, rank {rank})")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(feature)

    plt.tight_layout()
    return fig, ax


