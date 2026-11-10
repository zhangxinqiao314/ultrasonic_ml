# Ultrasonic ML

gathering code and notebooks for ultrasonic analysis

## Installation
In command line:
```
git clone -b hdf5_format https://github.com/changbatterygroup/ultrasonic_ml.git
```

Install requirements:
```
pip install -r requirements.txt
```

Before attempting to import ultrasonic_ml packages in python, add to paths:
```python
import sys
sys.path.append('<path_to_repo>/ultrasonic_ml/src/')

from ultrasonic_ml.viz import ...
```

You can follow along with examples in the Notebooks folder

## Repository Structure:

```
ultrasonic_ml/
|
|-- README.md
|-- .gitignore
|
|-- Notebooks/
|   |-- fft/
|   |   |-- index.rst (outline what you want to display for sphinx documentation)
|   |   |-- notebooks and affiliated docs
|   |   ...
|   |
|   |-- gradient/
|   |   |-- index.rst (outline what you want to display for sphinx documentation)
|   |   |-- notebooks and affiliated docs
|   |   ...
|   |
|   |-- toy_tmm_modeling/
|   |   |-- index.rst (outline what you want to display for sphinx documentation)
|   |   |-- notebooks and affiliated docs
|   |   ...
|
|-- src/ (add to path and import pkgs as shown below)
|   |-- __init__.py
|   |-- ultrasonic_ml/
|   |   |-- __init__.py
|   |   |-- utils.py
|   |   |   ...
|   |   |-- data/
|   |   |-- models/
|   |   |-- viz/
|
|-- docs/ (sphinx documentation- updating instructions on bottom)
|   |-- build/ (Don't edit)
|   |   |-- html/
|   |   |   |-- index.html (open this to look at documentation)
|   |   |   |   ...
|   |   |   ...
|   |
|   |-- source/
|   |   |-- conf.py (configuration file for building and loading modules)
|   |   |-- index.rst (structure of the documentation page)
|   |   |-- examples/ (symlink to Notebooks folder)
|   |   |   ...
|
```

```python
import sys
sys.path.append('~/.../ultrasonic_ml/src/')

from ultrasonic_ml.viz import ...
```

## Restructured to HDF5 to faster analysis and visualization

#### Maintainance:
* periodically update readme as the functions change
* find and write TODO for tests
* test heritability of base classes as more functions added on

#### Naming Conventions:
* These are used throughout databases. There are no hard rules, but provide structure for understanding the code
* Hook: Subclass hooks are meant to be overridden by child classes and will be marked
* Within class methods:
    * `__<...>__()` are magic mathods that exibit special behaviors in python
    * `_<...>()` are helper functions, meant to be called within the class only. But they can be accessed elsewhere
    * `@staticmethod` above a method means it can be accessed without initializing a class object first
    * `get_<...>` prefix usually means getting headers, metadata or small slices
    * `fetch_<...>` prefix means retrieving large chunks of actual data. 
    * `check_<...>` means checking the correct structure or shape of object
    * `setup_<...>` includes checking and setting up correct dataset properties and metadata
    * `fill_<...>` usually takes an index iterator, then calculates and fills the appropriate dataset at the indices
* Class Names:
    * `AcousticDatabase<...>(AcousticDatabase)` inherits from the base class and adds general methods for `<...>` analysis. For example, CWT.
    * `Acoustic<...>Database(AcousticDatabase)` inherits from the base class and adds specific methods for experiment type `<...>`. For example, a 2D scan


### SQlite Database structure:
1. 1st level indents for tables
2. 2nd level indent for columns

* original waveform is referenced with collection_index
  
```text
DATABASE.sqlite3
|
|-- parameters (originally here)
|   |-- gui TEXT
|   |-- experiment TEXT
|   ...
|   |-- 'omega_scaled' REAL (*method* for calculating last 3)
|   |-- 'dt' REAL
|   |-- 'f_s' REAL
|
|-- acoustics (originally here)
|   |-- voltage BLOB (*method* for getting key names)
|   ... multiple scans from multiplexer
|   |-- time REAL
|   |-- frequency REAL (*method* for calculating)
|   |-- time_collected REAL (*method* need to retireve and translate to datetime format)
|   |-- collection_index INTEGER 
|   |-- X REAL
|   |-- Z REAL
|   PRIMARY KEY collection_index
|   
```

### AcousticSQlite python class and methods

```
DATABASE.sqlite3
    |
    V
AcousticSQlite
|
|-- magic
|   |-- __init__()
|   |-- __enter__()
|   |-- __exit__()
|   |-- __len__()
|   
|-- database
|   |-- connect()
|   |-- close()
|   |-- print_schema()
|   |-- get_tables()
|   |-- get_columns()
|   |-- fetch_value()
|   |-- fetch_column()
|
|-- parameters
|   |-- get_sqlite_parameters()
|   |-- initialize_frequency_parameters()
|
|-- Raw waveform handling
|   |-- store_acoustics_values()
|   |-- store_acoustics_values_batch()
|   |-- fetch_waveform()
|   |-- fetch_waveform_batch()
|   |-- get_waveform_columns()
|   |-- fetch_time()
|   |-- get_acquisition_count()
|
|-- serialization
|   |-- serialize_array()
|   |-- deserialize_array()
```

### HDF5 Database Structure example. 
- More groups can be added or nested as needed

```
<HDF5 file "091626_singleroll_scan_1mm_multiplexer.h5" (mode r+)>
|--<HDF5 dataset "config_metadata": shape (), type "|O">
|
|--<HDF5 group "/first_break_aligned" (2 members)>
|  |--<HDF5 dataset "hilbert": shape (2, 151, 31, 17111), type "<f4">
|  |--<HDF5 dataset "waveforms": shape (2, 151, 31, 17111), type "<f4">
|
|--<HDF5 group "/preprocessed_data" (2 members)>
|  |--<HDF5 dataset "hilbert": shape (2, 151, 31, 20000), type "<f4">
|  |--<HDF5 dataset "waveforms": shape (2, 151, 31, 20000), type "<f4">
|
|--<HDF5 group "/raw_data" (2 members)>
|  |--<HDF5 dataset "hilbert": shape (2, 151, 31, 20000), type "<f4">
|  |--<HDF5 dataset "waveforms": shape (2, 151, 31, 20000), type "<f4">
|
```

### AcousticDatabase python base class and methods
```
DATABASE.sqlite3
    |
    V
AcousticSQlite object
    |
    V
AcousticDatabase object
|
|-- magic
|   |-- __init__()
|   |-- __len__()
|   
|-- database
|   |-- open_h5()
|   |-- get_check_group()
|   |-- _print_schema()
|   |-- print_schema()
|   |-- print_sqlite_schema()
|   |-- sqlite_to_h5() - Hook meant to call setup_parameters and setup/fill_raw_dataset
|
|-- parameters
|   |-- print_parameters()
|   |-- get_sqlite_parameters()
|   |-- save_parameters()
|   |-- load_parameters()
|   |-- setup_parameters() - Hook mean to retrieve parameters from sqlite file and save as H5 dataset in root
|
|-- Raw data
|   |-- setup_raw_dataset()
|   |-- fill_raw_dataset()
|
|-- Indexing - subclass hooks which will be reviesed for specific chaped datasets
|   |-- write_indexed_parameter() - Hook
|   |-- idx_sqlite_data_generator() - Hook
|   |-- <...>_generator() - Hook. multiple
|
|-- Preprocessing
|   |-- _absolute_max() - @staticmethod
|   |-- _hilbert_window() - @staticmethod
|   |-- _undo_gain() - @staticmethod
|   |-- _build_filter_sos()
|   |-- setup_preprocessing_data()
|   |-- fill_preprocessing_data()
|
|-- Alignment 
|   |-- _calculate_rollback() - @staticmethod
|   |-- setup_first_break_aligned_data()
|   |-- fill_first_break_aligned_data()
|
|-- Basic viz
|   |-- view_waveform() - noninteractive view at a single index
|
```

### AcousticViewer python base class and methods
- Inherits from param.Parameterized base class in the HoloViz Param library. Marked attributes will be tracked for interactive viewing

```
DATABASE.sqlite3
    |
    V
AcousticSQlite object
    |
    V
AcousticDatabase
    |
    V
AcousticViewer(param.Parameterized)
|
|-- Magic
|   |-- __init__()
|   |-- __enter__()
|   |-- __exit__()
|   
|-- Database
|   |-- open()
|   |-- close() 
|   |-- clear_cache() - Hook to call .cache_clear() on all methods with header @lrucache 
|   |-- _open_datasets() - Hook to open h5 datasets to render
|   |-- _close_datasets() - Hook to close h5 file and datasets
| 
|-- Data Fetchers
| 
|-- Interactive grid components
| 
|-- Dashboard
|   |-- create_dashboard() - Hook meant to assemble grid components, to be called by launch
|   |-- launch() - @classmethod to create a context manager so all files are correctly opened and closed
|
```

## Example of a child class and viewer 

### AcousticScanDatabase python base class and methods
- Inherits all methods of `AcousticDatabase`. Subclass hooks are defined. Additional methods added if needed.

```
DATABASE.sqlite3
    |
    V
AcousticSQlite object
    |
    V
AcousticScanDatabase(AcousticDatabase)
|
|-- magic
|   |-- __init__()
|   
|-- setup and indexing functions - subclass hooks filled here
|   |-- sqlite_to_h5() 
|   |-- setup_parameters()
|   |-- idx_sqlite_data_generator()
|   |-- w_x_z_generator() 
|
|-- Preprocessing
|   |-- write_preprocessed_data() - calls setup_preprocessing_data, fill_preprocessing_data
|
|-- Alignment 
|   |-- write_first_break_aligned_dataset() - calls up_first_break_aligned_data, fill_first_break_aligned_data
|
|-- Basic viz
|   |-- view_mean_img_plot() - noninteractive view of mean scan and waveform
|
```

### AcousticScanViewer python base class and methods
- Inherits all methods of `AcousticViewer`. Subclass hooks are defined. Additional methods added to fetch data and create interactive grids

```
DATABASE.sqlite3
    |
    V
AcousticSQlite object
    |
    V
AcousticScanDatabase
    |
    V
AcousticScanViewer(AcousticViewer)
|
|-- Magic
|   |-- __init__()
|   
|-- Database
|   |-- _open_datasets() 
|   |-- _close_datasets() 
| 
|-- Data Fetchers
|   |-- get_2d_slice() - reads directly from h5 file.
|   |-- _fetch_1d_waveform() - @lru_cache t0 save the hilbert and waveforms. Cached bc last dimension is the longest.
|   |-- get_1d_waveform() - calls _fetch_1d_waveform with corrected coordinates
| 
|-- Interactive grid components
|   |-- build_spatial_grid()
|   |-- build_overlay_waveform_plot()
| 
|-- Dashboard
|   |-- create_dashboard()
|
```


## Sphinx documentation
- When getting started, run `sphinx-build -M html docs/source/ docs/build/`. This shouldn't overwrite existing documentation formatting files.
- To autogenerate documentation html, from inside docs folder run `sphinx-autobuild source build/html`
