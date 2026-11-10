import numpy as np
import time
from functools import wraps
import sys
import json

from moviepy import VideoFileClip
from moviepy.video.fx import Resize, MultiplySpeed


# Profiling decorator
def profile(func):
    '''Decorator to measure the execution time of a function.
    
    Parameters
    ----------
    func : function
        The function to be profiled.
        
    Returns
    -------
    function
        A wrapper function that measures the execution time of the original function and prints it.
        
    Usage
    -----
    ```
    @profile
    def my_function():
        # Function implementation
    
    >>> my_function()
    my_function took 0.1234 seconds
    ```
    '''
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def tensor_to_numpy(tensor):
    '''Convert a PyTorch tensor to a NumPy array and move to cpu. For visualization and plotting.'''
    return tensor.detach().cpu().numpy()

def display_dict_tree(data, indent=0):
    """Display the tree structure of the pickle file with indentation for nested items."""
    tab = "--" * indent
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{tab}{key}:")
            display_dict_tree(value, indent=indent+1)
        else:
            print(f"{tab}{key}")
            
def get_deep_size(obj):
    """Recursively calculates the true memory footprint of a nested object. 
    
    Parameters
    ----------
    obj : object
        The object for which to calculate the memory footprint.
        
    Returns
    -------
    int
        The total memory footprint of the object and its nested elements.
    """
    size = sys.getsizeof(obj)
    memory_footprint = size
    if isinstance(obj, dict):
        size += sum(get_deep_size(k) + get_deep_size(v) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)): #hashable
        size += sum(get_deep_size(i) for i in obj)
    return size, memory_footprint

def convert_mov_to_gif(input_path, output_path, target_fps=12, scale_factor=0.5):
    """
    Converts a .mov to a silent, fast GIF using correct MoviePy v2.0+ class names.
    
    Parameters:
    -----------
    input_path : str
        Path to the input .mov file.
    output_path : str
        Path to save the output .gif file.
    target_fps : int, optional
        Frames per second for the output GIF. Default is 12.
    scale_factor : float, optional
        Factor to scale down the resolution of the GIF. Default is 0.5 (50% of original size).
        
    Returns:
    --------
    None
    
    Usage:
    ------
    ```
    >>> input_video = "/Users/xz498/Library/Mobile Documents/com~apple~QuickTimePlayerX/Documents/trimmed scan hilber.mov"    # Replace with your MP4 filename
    >>> output_gif = "./UT_scan_viewer.gif"    # Replace with your desired GIF filename
    
    >>> convert_mov_to_silent_fast_gif_v2(input_video, output_gif, target_fps=12, scale_factor=0.5) # Run the function
    ```
    input_video = "/Users/xz498/Library/Mobile Documents/com~apple~QuickTimePlayerX/Documents/trimmed scan hilber.mov"    # Replace with your MP4 filename
    output_gif = "./UT_scan_viewer.gif"    # Replace with your desired GIF filename

    # Run the function
    convert_mov_to_silent_fast_gif_v2(input_video, output_gif, target_fps=12, scale_factor=0.5)

    """
    print(f"Loading {input_path}...")
    
    # Strip audio directly upon loading the clip
    clip = VideoFileClip(input_path, audio=False)
    
    # 1. Speed up the video by 2x (samples every other frame)
    print("Doubling the playback speed...")
    fast_clip = clip.with_effects([MultiplySpeed(2.0)])
    
    # 2. Decrease quality/file size by downscaling resolution
    if scale_factor != 1.0:
        print(f"Rescaling video resolution by {scale_factor}x...")
        # In v2, Resize takes a scale_factor keyword argument or a new size tuple
        fast_clip = fast_clip.with_effects([Resize(new_size=scale_factor)])
        
    print(f"Writing fast, silent GIF to {output_path} at {target_fps} FPS...")
    
    # 3. Write out the GIF
    fast_clip.write_gif(output_path, fps=target_fps)
    
    # Close resources
    fast_clip.close()
    clip.close()
    print("Conversion complete!")
    
class NumpyEncoder(json.JSONEncoder):
    '''A helper tool to handle numpy types when converting to JSON.'''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
