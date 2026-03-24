import numpy as np
import time

# Profiling decorator
def profile(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def tensor_to_numpy(tensor):
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
            
