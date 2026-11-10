import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np
from czt import czt, iczt
from scipy import signal
from ..utils import profile
import time
from tqdm import tqdm

import re

### Parameter_setup ###

def compute_acoustic_properties(param_df, f0=2.25e6):
    rho, E = np.array(param_df["density"]), np.array(param_df["E"])
    c = np.sqrt(E / rho)
    param_df["sound_speed"] = c.tolist()
    param_df["Z"] = (rho * c).tolist()
    param_df['wavelength'] = [c / f0 for c in param_df['sound_speed']]
    return param_df

def format_chem_formula(formula, mode="mpl"):
    """
    Convert 'Li_0.9 Ni_0.6 Mn_0.2 Co_0.2 O_2' style strings into
    properly subscripted chemical formulas.

    mode='mpl'     -> matplotlib mathtext, e.g. for axis labels/legends
    mode='unicode' -> unicode subscript chars, for plain-text dataframe display
    mode='html'    -> <sub> tags, for df.style / Jupyter HTML rendering
    """
    # split into (Element, subscript) tokens, ignoring spaces
    tokens = re.findall(r'([A-Za-z]+)_?([\d.]*)', formula.replace(" ", ""))
    tokens = [(el, num) for el, num in tokens if el]  # drop empty matches

    if mode == "mpl":
        parts = [f"{el}_{{{num}}}" if num else el for el, num in tokens]
        return r"$\mathrm{" + "".join(parts) + "}$"

    elif mode == "unicode":
        sub_map = str.maketrans("0123456789.", "₀₁₂₃₄₅₆₇₈₉.")
        parts = [f"{el}{num.translate(sub_map)}" if num else el for el, num in tokens]
        return "".join(parts)

    elif mode == "html":
        parts = [f"{el}<sub>{num}</sub>" if num else el for el, num in tokens]
        return "".join(parts)

    else:
        raise ValueError("mode must be 'mpl', 'unicode', or 'html'")
    
Si_param_dict = {
    "name": ["Si", 
             "Li Si", 
             "Li_12 Si_7", 
             "Li_13 Si_4", 
             "Li_15 Si_4", 
             "Li_22 Si_5 / Li_4.4 Si"],
    "density": np.array([2.3, 1.93, 1.55, 1.28, 1.22, 1.18]) * 1e3,
    "E": np.array([151.1, 92.68, 84.59, 82.92, 48.59, 75.69]) * 1e9,
    "thickness": [0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
    "attenuation": np.zeros(6), }
Si_param_dict["name_mpl"] = [format_chem_formula(n, "mpl") for n in Si_param_dict["name"]]

NMC622_param_dict = {
    "name": ["Li Ni_0.6 Mn_0.2 Co_0.2 O_2", 
             "Li_0.9 Ni_0.6 Mn_0.2 Co_0.2 O_2", 
             "Li_0.8 Ni_0.6 Mn_0.2 Co_0.2 O_2",
             "Li_0.7 Ni_0.6 Mn_0.2 Co_0.2 O_2", 
             "Li_0.6 Ni_0.6 Mn_0.2 Co_0.2 O_2", 
             "Li_0.4 Ni_0.6 Mn_0.2 Co_0.2 O_2",
             "Li_0.3 Ni_0.6 Mn_0.2 Co_0.2 O_2"],
    "density": np.array([4.76, 4.72, 4.71, 4.69, 4.63, 4.67, 4.67]) * 1e3,
    "E": np.array([170, 156, 145, 137, 116, 76, 96]) * 1e9,
    "thickness": [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
    "attenuation": np.zeros(7),
}
NMC622_param_dict["name_mpl"] = [format_chem_formula(n, "mpl") for n in NMC622_param_dict["name"]]

LSPCl_param_dict = {
    "name": ["Li_6 P S_5 Cl"],
    "density": np.array([1.64]) * 1e3,
    "E": np.array([27.4]) * 1e9,
    "thickness": [0.01],
    "attenuation": [0.0],
}
LSPCl_param_dict["name_mpl"] = [format_chem_formula(n, "mpl") for n in LSPCl_param_dict["name"]]

metals_param_dict = {
    "name": ["Steel_top", "Steel_bottom", "Cu"],
    "density": np.array([7.85, 7.85, 8.96]) * 1e3,
    "E": np.array([195, 195, 110]) * 1e9,
    "thickness": np.array([.04041, .02479, 0.01]),
    "attenuation": [0., 0., 0],
}
metals_param_dict["name_mpl"] = metals_param_dict["name"]

couplant_param_dict = {
    "name": ["Buna-N 50A"],
    "density": np.array([1.171]) * 1e3,
    "E": np.array([1.348]) * 1e9,
    "thickness": np.array([0.00238125]),
    "attenuation": np.zeros(1),
}
couplant_param_dict["name_mpl"] = couplant_param_dict["name"]

param_dicts = [Si_param_dict, NMC622_param_dict, LSPCl_param_dict, metals_param_dict, couplant_param_dict]
params = pd.concat([pd.DataFrame(dict_) for dict_ in param_dicts], ignore_index=True)
params = compute_acoustic_properties(params)


def construct_stack_df(stack):
    '''
    Construct a DataFrame from a list of materials and their properties.
    Parameters:
        stack (list): A list of material names.
        params (DataFrame): A DataFrame containing material properties.
    Returns:
        DataFrame: A DataFrame containing the properties of the materials in the stack.
    '''
    stack_rows = []
    for material in stack:
        row = params[params['name'] == material].iloc[0]
        stack_rows.append(row)

    stack_df = pd.DataFrame(stack_rows)
    stack_df = stack_df.reset_index(drop=True)
    name_mask = stack_df.columns.str.startswith("name_")
    ordered_columns = stack_df.columns[~name_mask].append(stack_df.columns[name_mask])
    stack_df = stack_df.loc[:, ordered_columns]
    
    return stack_df

def style_stack_df(stack_df):
    """
    Style a stack DataFrame with formatted numeric columns.

    Parameters
    ----------
    stack_df : pandas.DataFrame
        DataFrame containing the properties of the materials in the stack.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styled DataFrame.
    """
    numeric_columns = stack_df.select_dtypes(include=np.number).columns
    return stack_df.style.format('{:.2e}', subset=numeric_columns)

### calculations ###

def layers_from_df(df):
    """
    Convert a layer DataFrame to dictionaries.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the layer properties.

    Returns
    -------
    list of dict
        Layer properties in DataFrame row order.
    """
    return df.to_dict("records")

def impedance(layer):
    """
    Calculate acoustic impedance.

    Parameters
    ----------
    layer : dict
        Layer properties containing ``density`` and ``sound_speed``.

    Returns
    -------
    float
        Acoustic impedance.
    """
    return layer.get('impedance', layer["density"] * layer["sound_speed"])

def reflection_coefficient(z1, z2):
    """
    Calculate the pressure reflection coefficient.

    Parameters
    ----------
    z1 : float
        Acoustic impedance of the incident medium.
    z2 : float
        Acoustic impedance of the transmitted medium.

    Returns
    -------
    float
        Pressure reflection coefficient.
    """
    return (z2 - z1) / (z1 + z2)

def transmission_coefficient(z1, z2):
    """
    Calculate the pressure transmission coefficient.

    Parameters
    ----------
    z1 : float
        Acoustic impedance of the incident medium.
    z2 : float
        Acoustic impedance of the transmitted medium.

    Returns
    -------
    float
        Pressure transmission coefficient.
    """
    return 2 * z2 / (z1 + z2)


def propagation_matrix(f, layer):
    """
    Construct the propagation matrix for a layer.

    Parameters
    ----------
    f : float
        Frequency in Hz.
    layer : dict
        Layer properties containing ``sound_speed``, ``thickness``,
        and optionally ``attenuation``.

    Returns
    -------
    numpy.ndarray
        2 x 2 complex propagation matrix.
    """
    k = 2 * np.pi * f / layer["sound_speed"]
    d = layer["thickness"]
    a = layer.get("attenuation", 0.0)
    return np.diag([ np.exp((1j * k - a) * d), np.exp(-(1j * k - a) * d), ])

def interface_matrix(layer1, layer2):
    """
    Construct the interface matrix between two layers.

    Parameters
    ----------
    layer1 : dict
        Properties of the incident layer.
    layer2 : dict
        Properties of the transmitted layer.

    Returns
    -------
    numpy.ndarray
        2 x 2 complex interface matrix.
    """
    z1, z2 = impedance(layer1), impedance(layer2)
    r = reflection_coefficient(z1, z2)
    t = transmission_coefficient(z1, z2)
    return np.array([[1, r], [r, 1]], dtype=complex) / t


def transfer_matrix(f, layers):
    """
    Construct the transfer matrix of the complete stack.

    Parameters
    ----------
    f : float
        Frequency in Hz.
    layers : list of dict
        Layer properties ordered from input to output.

    Returns
    -------
    numpy.ndarray
        2 x 2 complex transfer matrix.
    """
    T = propagation_matrix(f, layers[0])

    for l1, l2 in zip(layers[:-1], layers[1:]):
        T = T @ interface_matrix(l1, l2)
        T = T @ propagation_matrix(f, l2)

    return T


# TODO: when you put in class, rephrase so you only calculate transfer matrix once during initialization
# TODO: or store layer by layer transfer matrix in a list
def transmission_response(frequencies, layers):
    """
    Calculate the frequency-domain transmission response.

    Parameters
    ----------
    frequencies : array_like
        Frequencies in Hz.
    layers : list of dict
        Layer properties ordered from input to output.

    Returns
    -------
    numpy.ndarray
        Complex transmission response for each frequency.
    """
    return np.array([
        1 / transfer_matrix(f, layers)[0, 0]
        for f in frequencies
    ])

def reflection_response(frequencies, layers):
    """
    Calculate the frequency-domain reflection response.

    Parameters
    ----------
    frequencies : array_like
        Frequencies in Hz.
    layers : list of dict
        Layer properties ordered from input to output.

    Returns
    -------
    numpy.ndarray
        Complex reflection response for each frequency.
    """
    reflections = []
    for f in frequencies:
        T = transfer_matrix(f, layers)
        reflections.append(T[1, 0] / T[0, 0])

    return np.array(reflections)


def pad_t_signal(t, signal, dt, beginning_pad_zeros=0, end_pad_zeros=0):
    """
    Pad a signal with zeros at the beginning and end.

    Parameters
    ----------
    signal : array_like
        Input signal.
    beginning_pad_zeros : int, optional
        Number of zeros to pad at the beginning. Default is 0.
    end_pad_zeros : int, optional
        Number of zeros to pad at the end. Default is 0.

    Returns
    -------
    numpy.ndarray
        Zero-padded signal.
    """
    new_signal = np.pad(signal, (beginning_pad_zeros, end_pad_zeros), mode='constant', constant_values=0)
    new_t = np.pad(t, (beginning_pad_zeros, end_pad_zeros), mode='linear_ramp', end_values=(t[0]-beginning_pad_zeros*dt, t[-1]+end_pad_zeros*dt))
    return new_t, new_signal


def unpad_t_signal(t, signal, beginning_pad_zeros=0, end_pad_zeros=0):
    """
    Remove zero-padding from a signal.

    Parameters
    ----------
    signal : array_like
        Input signal.
    beginning_pad_zeros : int, optional
        Number of zeros padded at the beginning. Default is 0.
    end_pad_zeros : int, optional
        Number of zeros padded at the end. Default is 0.

    Returns
    -------
    numpy.ndarray
        Unpadded signal.
    """
    new_signal = signal[beginning_pad_zeros:len(signal)-end_pad_zeros]
    new_t = t[beginning_pad_zeros:len(t)-end_pad_zeros]
    return new_t, new_signal

def fft_phase_shifted(t, signal, dt, 
                      beginning_pad_zeros=0, end_pad_zeros=0):
    '''Apply phase correction to a signal in the frequency domain.'''
    t0 = t[0] # this is how much time has passed before the first sample, so we need to correct for it in the frequency domain
    
    t, signal = pad_t_signal(t, signal, dt, beginning_pad_zeros, end_pad_zeros)
    freqs = np.fft.fftfreq(len(t), d=dt)
    phase_correction = np.exp(-1j * 2 * np.pi * freqs * t0)
    
    fft_raw = np.fft.fft(signal, norm="ortho")
    fft_shifted = fft_raw * phase_correction

    return freqs, fft_shifted


# def apply_response(waveform, dt, response, transform_method="fft"):
#     """
#     Apply a frequency response to a real-valued waveform.

#     Parameters
#     ----------
#     waveform : array_like
#         Real-valued input waveform.
#     dt : float
#         Sampling interval in seconds.
#     response : array_like
#         Complex frequency response corresponding to the positive
#         frequencies of the waveform.
#     transform_method : str, optional
#         Method to use for the transform. Options are "fft" or "czt".

#     Returns
#     -------
#     numpy.ndarray
#         Real-valued output waveform.
#     """
#     if transform_method == "fft":
#         spectrum = np.fft.rfft(waveform)
#         return np.fft.irfft(spectrum * response, n=len(waveform))
#     elif transform_method == "czt":
#         # 1. Create a sample time-domain signal (e.g., a sine wave)
#         N, M = len(waveform), len(waveform)
#         czt_transform = czt(m=M, n=N)
#         iczt_transform = iczt(m=M, n=N)
#         spectrum = czt_transform(waveform)
#         return iczt_transform(spectrum * response)
#     else:
#         raise ValueError("Unsupported transform method")

def apply_response(waveform, response, transform_method="fft"):
    """
    Apply a frequency response to a real-valued waveform.

    Parameters
    ----------
    waveform : array_like
        Real-valued input waveform.
    dt : float
        Sampling interval in seconds.
    response : array_like
        Complex frequency response, evaluated at the *signed* frequencies
        from np.fft.fftfreq(len(waveform), dt) -- i.e. full spectrum,
        not just non-negative frequencies.
    transform_method : str, optional
        Method to use for the transform. Options are "fft" or "czt".

    Returns
    -------
    numpy.ndarray
        Real-valued output waveform.
    """
    if transform_method == "fft":
        spectrum = np.fft.fft(waveform)
        out = np.fft.ifft(spectrum * response)
        # # imaginary part should be ~numerical noise if response satisfies H(-f) = H(f)*; report it instead of hiding it
        # max_imag = np.max(np.abs(out.imag))
        # if max_imag > 1e-6 * np.max(np.abs(out.real)):
        #     warnings.warn(f"apply_response: non-negligible imaginary part "
        #                    f"({max_imag:.3e}), check H(-f)=H(f)* symmetry")
        return out
    
    elif transform_method == "czt":
        N, M = len(waveform), len(waveform)
        czt_transform = czt(m=M, n=N)
        iczt_transform = iczt(m=M, n=N)
        spectrum = czt_transform(waveform)
        return iczt_transform(spectrum * response).real
    else:
        raise ValueError("Unsupported transform method")
    
def propagate_waveform(waveform, dt, layers, transform_method='fft'):
    """
    Propagate a waveform through the layered stack.

    Parameters
    ----------
    waveform : array_like
        Real-valued input waveform.
    dt : float
        Sampling interval in seconds.
    layers : list of dict
        Layer properties ordered from input to output.

    Returns
    -------
    numpy.ndarray
        Transmitted waveform.
    """
    if transform_method == 'fft':
        f = np.fft.fftfreq(len(waveform), dt)
    else:
        raise ValueError("Unsupported transform method")
    H = transmission_response(f, layers)
    return apply_response(waveform, H, transform_method=transform_method)

def reflect_waveform(waveform, dt, layers):
    """
    Calculate the waveform reflected from the layered stack.

    Parameters
    ----------
    waveform : array_like
        Real-valued input waveform.
    dt : float
        Sampling interval in seconds.
    layers : list of dict
        Layer properties ordered from input to output.

    Returns
    -------
    numpy.ndarray
        Reflected waveform.
    """
    f = np.fft.fftfreq(len(waveform), dt)
    H = reflection_response(f, layers)
    return apply_response(waveform, H)


def total_waveform(waveform, dt, layers):
    """
    Calculate the total waveform (transmitted + reflected) from the layered stack.

    Parameters
    ----------
    waveform : array_like
        Real-valued input waveform.
    dt : float
        Sampling interval in seconds.
    layers : list of dict
        Layer properties ordered from input to output.

    Returns
    -------
    numpy.ndarray
        Total waveform (transmitted + reflected).
    """
    transmitted = propagate_waveform(waveform, dt, layers)
    reflected = reflect_waveform(waveform, dt, layers)
    return transmitted + reflected


### Utilities ###
def get_unique(arr, tolerance):
    arr.sort()
    arr = np.array(arr) 
    diffs  = np.diff(arr)  # Round to avoid floating point issues
    mask = np.concatenate(([True], diffs > tolerance))
    return arr[mask]

@profile
def generate_acoustic_paths(num_layers=2, max_order=1):
    """
    Generates all valid reflection and transmission paths up to a specified order. With gemini
    
    Layers: 1 to num_layers
    Incident Medium: 0
    Substrate: num_layers + 1
    Order: Number of reflection events
    """
    # Layers are numbered 1 through ``num_layers``; the substrate is the
    # medium immediately beyond the final layer.
    substrate = num_layers
    
    # Store complete paths
    # Format: (Type ['Reflection' or 'Transmission'], Path List, Order)
    completed_paths = []
    
    # DFS Stack elements: (current_layer, direction, current_order, path_history)
    # direction: +1 for forward/right, -1 for backward/left
    # Initial state: ray is in layer 0, moving forward (+1)
    stack = [(0, 1, 0, [0])]
    
    while stack:
        layer, direction, order, path = stack.pop()
        
        if direction == 1:
            # Moving forward: approaching the interface between 'layer' and 'layer + 1'
            next_layer = layer + 1
            
            if next_layer == substrate:
                # Entering the substrate completes a transmission path.
                completed_paths.append(('Transmission', path+['exit'], order))
                continue
            
            # Option A: Transmission (Ray enters next_layer, direction & order stay same)
            stack.append((next_layer, 1, order, path + [next_layer]))
            
            # Option B: Reflection (Ray bounces off interface, stays in current layer, reverses direction)
            # The incident-medium boundary is not an internal reflection path.
            if layer >= 0 and order + 1 <= max_order:
                stack.append((layer, -1, order + 1, path + [f"R({layer}|{next_layer})", layer]))
                
        elif direction == -1:
            # Moving backward: approaching the interface between 'layer' and 'layer - 1'
            next_layer = layer-1
            # print(f"layer: {layer}, direction: {direction}, order: {order}, path: {path}, next_layer: {next_layer}")
            if next_layer == -1:
                # Ray has completely bounced back into the incident medium (Layer 0)
                completed_paths.append(('Reflection', path+['exit'], order))
                continue
                
            # Option A: Transmission (Ray enters next_layer going backward)
            stack.append((next_layer, -1, order, path + [next_layer]))
            
            # Option B: Reflection (Ray bounces off interface, stays in current layer, moves forward)
            # Do not reflect at the incident-medium boundary.
            if next_layer >= 0 and order + 1 <= max_order:
                stack.append((layer, 1, order + 1, path + [f"R({next_layer}|{layer})", layer]))

    return completed_paths

def print_paths(paths, return_paths=False):
    transmission_paths_str = [p for p in paths if p[0] == 'Transmission']; transmission_paths_str.sort(key=lambda x:len(x[1])) 
    reflection_paths_str = [p for p in paths if p[0] == 'Reflection']; reflection_paths_str.sort(key=lambda x:len(x[1]))
    
    for p_type, path, order in transmission_paths_str: print(f"Transmission: {order} | Path: {' -> '.join(map(str, path))}")
    for p_type, path, order in reflection_paths_str: print(f"Reflection: {order} | Path: {' -> '.join(map(str, path))}")
        
    if return_paths: return transmission_paths_str, reflection_paths_str


def theoretical_path_time(acoustic_path, layers_df):
    """
    Calculate the theoretical time delay for a given acoustic path.

    Parameters
    ----------
    acoustic_path : list
        List of layer indices representing the acoustic path.
    layers_df : pandas.DataFrame
        DataFrame containing the layer properties.

    Returns
    -------
    float
        Theoretical time delay for the acoustic path.
    """
    path = [p for p in acoustic_path[1] if isinstance(p, int)] 
    return np.sum(layers_df.iloc[path]["thickness"] / layers_df.iloc[path]["sound_speed"])
    
def theoretical_path_times(paths, layers_df, tolerance=1e-9, verbose=True):
    transmission_paths_str = [p for p in paths if p[0] == 'Transmission']; transmission_paths_str.sort(key=lambda x:len(x[1])) 
    transmission_paths = [[p for p in path[1] if isinstance(p, int)] for path in transmission_paths_str]
    theoretical_transmission_delays = []
    for p in range(len(transmission_paths)):
        theoretical_transmission_delays.append( np.sum(layers_df.iloc[transmission_paths[p]]["thickness"] / layers_df.iloc[transmission_paths[p]]["sound_speed"]) )

    reflection_paths_str = [p for p in paths if p[0] == 'Reflection']; reflection_paths_str.sort(key=lambda x:len(x[1]))
    reflection_paths = [[p for p in path[1] if isinstance(p, int)] for path in reflection_paths_str]
    theoretical_reflection_delays = []
    for p in range(len(reflection_paths)):
        theoretical_reflection_delays.append( np.sum(layers_df.iloc[reflection_paths[p]]["thickness"] / layers_df.iloc[reflection_paths[p]]["sound_speed"]) )

    if verbose:
        print(f"Theoretical transmission delays: {theoretical_transmission_delays}")
        print(f"Theoretical reflection delays: {theoretical_reflection_delays}")

    return theoretical_transmission_delays, theoretical_reflection_delays

def unique_theoretical_path_times(paths, layers_df, tolerance=1e-9, verbose=False):
    theoretical_transmission_delays, theoretical_reflection_delays = theoretical_path_times(paths, layers_df, tolerance, verbose)
    
    try: theoretical_transmission_delays = get_unique(theoretical_transmission_delays, tolerance)
    except: pass
    
    try: theoretical_reflection_delays = get_unique(theoretical_reflection_delays, tolerance)
    except: pass
    
    return theoretical_transmission_delays, theoretical_reflection_delays


def find_peak_inds(t, envelope, verbose=True):
    """
    Find the peak times of the transmitted waveform.

    Parameters
    ----------
    t : array_like
        Time array corresponding to the waveforms.
    transmitted_envelope : array_like
        Envelope of the transmitted waveform.

    Returns
    -------
    numpy.ndarray
        Times at which peaks occur in the transmitted envelope.
    """
    peaks, _ = signal.find_peaks(envelope, prominence=0.01)
    if verbose:
        print(f"TMM time for transmitted waveform:")
        for pt in t[peaks]: print(f"\tPeak at {pt*1e6:.2f} μs")

    return peaks 


def compare_peak_times(t, layers_df, 
                       transmitted_envelope, reflected_envelope, 
                       paths, 
                       tolerance=1e-9):
    """
    Compare the peak times of the transmitted waveform to the theoretical delays.

    Parameters
    ----------
    t : array_like
        Time array corresponding to the waveforms.
    transmitted_envelope : array_like
        Envelope of the transmitted waveform.
    reflected_envelope : array_like
        Envelope of the reflected waveform.
    theoretical_delays : array_like
        Theoretical delays for each path.

    Returns
    -------
    list of bool
        List indicating whether each theoretical delay has a corresponding peak in the transmitted envelope.
    """
    
    theoretical_transmission_delays, theoretical_reflection_delays = unique_theoretical_path_times(paths, layers_df, tolerance)

    for envelope, theoretical_delays, label in zip([transmitted_envelope, reflected_envelope], 
                               [theoretical_transmission_delays, theoretical_reflection_delays],
                               ["Transmitted", "Reflected"]):
        peaks = find_peak_inds(t, envelope, verbose=False)
        peak_times = t[peaks]
        
        print(f'{label} tof: (TMM peaks,\ttheoretical)')
        for i in range(min(len(peak_times), len(theoretical_delays))):
            print(f"\t{i}:\t({peak_times[i]:.4e},\t{theoretical_delays[i]:.4e})")
     
def compare_tmm_waveform_coefficients(input_waveform, transmitted_waveform, reflected_waveform, layers, dt): 
    """
    Compare the total coefficients calculated from TMM and waveforms.

    Parameters
    ----------
    input_waveform : array_like
        The input waveform.
    transmitted_waveform : array_like
        The transmitted waveform.
    reflected_waveform : array_like
        The reflected waveform.
    layers : list of dict
        List of layer properties.
    dt : float
        Time step.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the total coefficients calculated from TMM and waveforms.
    """
    # TMM calculation of transmission and reflection coefficients
    f = np.fft.rfftfreq(len(input_waveform), dt)
    H_t = transmission_response(f, layers)
    H_r = reflection_response(f, layers)
    transmission_coefficient_sq = np.mean(np.abs(H_t)**2)
    reflection_coefficient_sq = np.mean(np.abs(H_r)**2)
    total_coefficient_sq = transmission_coefficient_sq + reflection_coefficient_sq

    # Calculate power ratios from waveforms
    input_power = np.mean(np.abs(input_waveform)**2)
    transmission_power_ratio = np.mean(np.abs(transmitted_waveform)**2)/input_power
    reflection_power_ratio = np.mean(np.abs(reflected_waveform)**2)/input_power
    total_power_ratio = transmission_power_ratio + reflection_power_ratio
    
    df = pd.DataFrame({
        "calculate from": ["TMM", "Waveforms"],
        "transmission coefficient": [transmission_coefficient_sq, transmission_power_ratio],
        "reflection coefficient": [reflection_coefficient_sq, reflection_power_ratio],
        'total': [total_coefficient_sq, total_power_ratio]
        })
    
    return df.style.format({
        "transmission coefficient": "{:.4f}",
        "reflection coefficient": "{:.4f}",
        "total": "{:.4f}"
        })
    
    
    
#########################################################
##################### visualization #####################
#########################################################

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 16,
        'figure.dpi': 100,
        'savefig.dpi': 100, })

def get_max_fig_dims(margin=0.9, fallback=14):
    """
    Determine a sensible max figure width (in inches) based on the 
    current screen's resolution, so tk windows never exceed screen bounds.

    Parameters
    ----------
    margin : float
        Fraction of screen width to use, leaving room for window chrome/taskbar.
    fallback : float
        Width (in inches) to use if screen detection fails (e.g. no display).
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # don't actually show a blank window
        screen_width_px = root.winfo_screenwidth()
        screen_height_px = root.winfo_screenheight()
        dpi = plt.rcParams.get('figure.dpi')  # default to 100 if not set
        root.destroy()
        return (screen_width_px / dpi) * margin, (screen_height_px / dpi) * margin
    except Exception:
        return fallback

max_figwidth, max_figheight = get_max_fig_dims()

print(f'tk max_dimensions: {max_figwidth:.2f} x {max_figheight:.2f}')



def plot_layers(layers_df, ax=None, show_layers='all', aspect_scale=12, title='Layers_stack'):
    """
    Plot a horizontal bar chart of the layered stack.

    Parameters
    ----------
    layers_df : pandas.DataFrame
        DataFrame containing the layer properties.
    ax : matplotlib.axes.Axes, optional
        The axes object to plot on. If None, a new figure and axes will be created.
    show_layers : str, optional
        A string indicating which layers to show. Default is 'all'. other option could be a list of indices.
    max_figwidth : float, optional
        Cap on figure width in inches. If None, auto-detected from the current screen.
    aspect_scale : float
        Scales figure width relative to total thickness, before capping. Can scale w max number of digits in layer name
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers_df)))
    if ax is None:
        total_thickness = layers_df["thickness"].sum()
        figwidth = min(total_thickness * aspect_scale * 100, max_figwidth)
        fig, ax = plt.subplots(figsize=(figwidth, 3), dpi=100)
    else: fig = ax.figure

    x = 0
    
    if show_layers == 'all':
        show_indices = list(range(len(layers_df)))
    elif isinstance(show_layers, list):
        show_indices = show_layers
    else:
        raise ValueError("show_layers must be 'all' or a list of indices")

    for _, l in layers_df.iterrows():
        if _ in show_indices:
            ax.barh(0.5, l["thickness"], left=x, height=1, alpha=0.5, color = colors[_], edgecolor="black")
            ax.text(x + l["thickness"]/2, 0.5, l["name_mpl"], ha="center", va="center", rotation=90)
        else:
            ax.barh(0, l["thickness"], left=x, height=1, alpha=0.,)
        x += l["thickness"]

    ax.set(xlim=(0, x), ylim=(0, 1), yticks=[],
        xlabel="Depth (m)")
    ax.set_title(title)
    
    return fig, ax

def plot_waveforms(t, waveforms, envelopes=None, labels=None, ax=None, 
                   title="Waveforms", x_label="Time (μs)", y_label="Amplitude (mV)"):

    if ax is None: _, ax = plt.subplots(figsize=(max_figwidth, 3), dpi=100)

    for i in range(len(waveforms)):
        try: ax.plot(t*1e6, waveforms[i], label=labels[i])
        except: ax.plot(t*1e6, waveforms[i])
        if envelopes is not None:
            ax.plot(t*1e6, envelopes[i], linestyle="--", color=ax.get_lines()[-1].get_color())
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if labels: ax.legend()

    return ax


def plot_tmm_results(layers_df, t, waveforms, envelopes, labels=None, ax=None):
    """
    Plot the layer stack and the corresponding waveforms.

    Parameters
    ----------
    layers_df : pandas.DataFrame
        DataFrame containing the layer properties.
    t : array_like
        Time array corresponding to the waveforms.
    input_waveform : array_like
        Input waveform.
    transmitted_waveform : array_like
        Transmitted waveform.
    reflected_waveform : array_like
        Reflected waveform.
    input_envelope : array_like
        Envelope of the input waveform.
    transmitted_envelope : array_like
        Envelope of the transmitted waveform.
    reflected_envelope : array_like
        Envelope of the reflected waveform.
    """
    fig, ax = plt.subplots(2, 1, figsize=(max_figwidth, 0.4*max_figwidth), dpi=100)
    ax=ax.flatten()

    _, ax[0] = plot_layers(layers_df, ax[0], show_layers=[0])
    ax[1] = plot_waveforms(t, waveforms, envelopes,
                                labels=["Input", "Transmitted", "Reflected"], ax=ax[1])

    fig.tight_layout()
    
    return fig, ax


def plot_acoustic_path(layers_df, acoustic_path, ax,
                        y_start=0.1, y_step=0.1,
                        title=None):  
    """
    Overlay the acoustic ray path on an existing layer-stack plot.

    Parameters
    ----------
    layers_df : pandas.DataFrame
        Same DataFrame used to build the stack (needs "thickness").
    acoustic_path : list
        Alternating list of layer indices (int) and interface labels 
        (str, e.g. 'R(1|2)'). Direction starts rightward and flips at 
        each reflection string.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Passed to plot_layers().
    show_layers : str or list, optional
        Passed through to plot_layers().
    y_start : float
        Starting height above the stack bar for the first segment.
    y_step : float
        Vertical increment per reflection (controls climb rate / spacing).
    dx : float
        Horizontal offset applied to reflection arrows so they slope 
        instead of being purely vertical. In the same units as "thickness" 
        (e.g. meters) — scale to your stack's total width.
    arrow_color : str
        Color of the path arrows.
    """
    type_, path_, order_ = acoustic_path
    y_step = min(y_step, 1/2**(order_))
    if type_ == 'Transmission': arrow_color="orange"
    if type_ == 'Reflection': arrow_color="green"
    
    ylim = ax.get_ylim()
    edges = np.concatenate(([0], np.cumsum(layers_df["thickness"].values)))
    left  = {i: edges[i]   for i in range(len(layers_df))}
    right = {i: edges[i+1] for i in range(len(layers_df))}

    direction = 1      # +1 = rightward, -1 = leftward
    y = y_start

    for step in path_[:-1]:
        if isinstance(step, (int, np.integer)):
            x_l, x_r = left[step], right[step]
            x_start, x_end = (x_l, x_r) if direction == 1 else (x_r, x_l)
            y_new = y if direction == 1 else y + y_step

            ax.annotate(
                "", xy=(x_end, y_new), xytext=(x_start, y),
                arrowprops=dict(arrowstyle="-|>", 
                                 color=arrow_color,
                                 lw=2, mutation_scale=12,
                                 shrinkA=0, shrinkB=0)
            )
            y = y_new

        elif isinstance(step, str) and step.startswith("R"):
            direction *= -1
        else:
            raise ValueError(f"Unrecognized path element: {step!r}")

    ax.set_ylim(ylim[0], ylim[1])
    if title is not None: ax.set_title(title)

    return ax

def clear_acoustic_path(ax):
    """
    Clear the acoustic path from an existing layer-stack plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to clear the path from.
    """
    for child in list(ax.get_children()):
        if isinstance(child, matplotlib.text.Annotation):
            child.remove()
    return ax


def plot_tmm_peaks(t, envelope, peak_inds, ax, crop_peaks=20, show_legend=False, label='TMM Peaks'):
    """
    Plot the peaks in the TMM waveform.
    """
    ax.scatter(t[peak_inds[:crop_peaks]]*1e6, envelope[peak_inds[:crop_peaks]], color='red', marker='x', label=label)
    if show_legend: ax.legend()
    
    return ax


def plot_theoretical_path_times(theoretical_transmission_delays, theoretical_reflection_delays, ax, show_legend=True):
    ax.vlines([t*1e6 for t in theoretical_transmission_delays], -5,5, linewidth=1, color='orange', linestyle=':', label='Theor. Trans. times')
    ax.vlines([t*1e6 for t in theoretical_reflection_delays], -5,5, linewidth=1, color='green', linestyle=':', label='Theor. Refl. times')
    if show_legend: ax.legend()
    
    return ax
    
    
# TODO: eliminated redundant calculations
def plot_path_waveforms_theoretical_v_tmm_time_delays(layers_df, acoustic_paths, t,
                                          input_waveform, transmitted_waveform, reflected_waveform, 
                                          input_envelope, transmitted_envelope, reflected_envelope,
                                          layers='all', return_fig=False
                                          ):
    """
    Plot (0): reflected path through stack, (1): waveform with diaplayed peaks, (2): slider
    """
    fig, ax = plt.subplots(3, 1, figsize=(max_figwidth, max_figheight), 
                           gridspec_kw={"height_ratios": [6, 10, 1]},
                           constrained_layout=True,)
    ax = ax.flatten()

    fig, ax[0] = plot_layers(layers_df, ax=ax[0], show_layers=layers)
    ax[0] = plot_acoustic_path(layers_df, acoustic_paths[0], ax=ax[0],
                                y_start=0.1, y_step=0.1,
                                title="Acoustic Path")
    
    ax[1] = plot_waveforms(t, 
                              [input_waveform, transmitted_waveform, reflected_waveform], 
                              [input_envelope, transmitted_envelope, reflected_envelope],
                              labels=["Input", "Transmitted", "Reflected"], 
                              ax=ax[1])
    waveform_ylim = ax[1].get_ylim()
    ax[1].set_ylim(waveform_ylim)
    ax[1].set_autoscale_on(False)
    
    theoretical_transmission_delays, theoretical_reflection_delays = unique_theoretical_path_times(acoustic_paths, layers_df, tolerance=1e-9, verbose=False)
    ax[1] = plot_theoretical_path_times(theoretical_transmission_delays, theoretical_reflection_delays, ax[1], show_legend=False)
    
    peak_inds = find_peak_inds(t, transmitted_envelope, verbose=False)
    ax[1] = plot_tmm_peaks(t, transmitted_envelope, peak_inds, ax[1])
    
    peak_inds = find_peak_inds(t, reflected_envelope, verbose=False)
    ax[1] = plot_tmm_peaks(t, reflected_envelope, peak_inds, ax[1], label=None)
    
    current_path_line = ax[1].axvline(
        theoretical_path_time(acoustic_paths[0], layers_df) * 1e6,
        -1.5, 2, linewidth=1.5, color='red', linestyle='-',
        label='Current path')
    ax[1].set_title(f"Waveforms")
    
    legend_y_pad = plt.rcParams.get('font.size')/plt.rcParams.get('figure.dpi')
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -legend_y_pad), ncol=3)    
    
    def update(val): # avoid replotting when possible
        path_idx = int(slider.val)
        ax[0] = clear_acoustic_path(ax[0])
        ax[0] = plot_acoustic_path(layers_df, acoustic_paths[path_idx], ax=ax[0], 
                                    y_start=0.1, y_step=0.1)
        ax[0].set_title(f"{acoustic_paths[path_idx][0]} | Order: {acoustic_paths[path_idx][2]}")
        current_path_line.set_xdata([ theoretical_path_time(acoustic_paths[path_idx], layers_df) * 1e6] )
        fig.canvas.draw_idle()
    
    # slider at bottom to select which path to display
    slider = Slider(ax[2], '', 0, len(acoustic_paths)-1, valinit=0, valstep=1, valfmt='%03d')
    ax[2].set_title("Select Acoustic Path")
    slider.on_changed(update)
    if return_fig: return fig, ax, slider


def save_plot_path_waveforms_gif(layers_df, acoustic_paths, t,
                                 input_waveform, transmitted_waveform, reflected_waveform,
                                 input_envelope, transmitted_envelope, reflected_envelope,
                                 filepath, layers='all', fps=2, dpi=None):
    """Save a GIF containing one frame for every acoustic-path slider value."""
    fig, _, slider = plot_path_waveforms_theoretical_v_tmm_time_delays(
        layers_df, acoustic_paths, t,
        input_waveform, transmitted_waveform, reflected_waveform,
        input_envelope, transmitted_envelope, reflected_envelope,
        layers=layers, return_fig=True,
    )

    writer = PillowWriter(fps=fps)
    try:
        with writer.saving(fig, filepath, dpi or fig.dpi):
            for path_idx in tqdm(range(len(acoustic_paths)), desc="Saving GIF frames"):
                slider.set_val(path_idx)
                fig.canvas.draw()
                writer.grab_frame()
    finally:
        plt.close(fig)

    return filepath


    