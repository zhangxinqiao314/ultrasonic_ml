#!/usr/bin/env python3
"""
Visualization script for acoustic scan data pickle files using Plotly.
Designed for use in Jupyter notebooks with interactive sliders.
"""

import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import argparse
import sys
import pprint
import matplotlib

def load_scan_data(pickle_path):
    """Load scan data from pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract numeric keys (scan indices)
    numeric_keys = sorted([k for k in data.keys() if isinstance(k, (int, np.integer))])
    
    print(f"Loaded {len(numeric_keys)} scans from {pickle_path}")
    print(f"File name: {data.get('fileName', 'N/A')}")
    
    return data, numeric_keys

try:
    import ipywidgets as widgets
    from IPython.display import display
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False

def plotly_viewer(dataset):
    """Interactive plot viewer that updates automatically as the slider changes.
    Creates vertical subplots for each crop region with correct time labels."""
    if not IPYWIDGETS_AVAILABLE:
        raise ImportError("ipywidgets is required for interactive plotting. Please install it: pip install ipywidgets")
    
    # # Create slider
    # i_slider = widgets.IntSlider(
    #     description='Scan #', 
    #     min=0, 
    #     max=len(dataset.numeric_keys)-1, 
    #     step=1, 
    #     value=0
    # )
    z_slider = widgets.IntSlider(
        description='Z #', 
        min=0, 
        max=dataset.image_shape[0]-1, 
        step=1, 
        value=0
    )
    x_slider = widgets.IntSlider(
        description='X #', 
        min=0, 
        max=dataset.image_shape[1]-1, 
        step=1, 
        value=0
    )
        
    full_time = dataset.data[0]['time']

    s_slider = widgets.IntSlider(
        description='Slice #', 
        min=0, 
        max=len(full_time)-1, 
        step=1, 
        value=0
    )
    
    # Get initial data
    # idx, processed_data = dataset[i_slider.value]
    idx, processed_data = dataset[z_slider.value*dataset.image_shape[1]+x_slider.value]
    # processed_data shape: (len(numeric_keys), len(crops), crop_length)
    # idx is the actual key value, need to find its position in numeric_keys
    
    # Create subplots - one per crop, stacked vertically
    fig = make_subplots(
        rows=2, 
        cols=1,
        # subplot_titles=[f'Crop [{dataset.crop[0]}:{dataset.crop[1]}]'],
        vertical_spacing=0.15,  # Increased spacing to prevent overcrowding
        shared_xaxes=False,
        horizontal_spacing=0.15,
        subplot_titles=['z-x plane', 'y plane'],
        specs=[
            [{'type': 'heatmap'}],
            [{'type': 'scatter'}]
        ],
        # row_heights=[1.0, 0.5]
    )
    
    # Convert to FigureWidget for interactive updates
    fig = go.FigureWidget(fig)
    
    # Get time data for the current scan
    # print(dataset[:,s_slider.value][1].shape)
    fig.add_trace(go.Heatmap(z=dataset[:,s_slider.value][1].reshape(dataset.image_shape),
                             name='Voltage',
                             zmin=-1,
                             zmax=1,
                             colorscale='Viridis',
                             colorbar=dict(x=1.02, len=0.5, yanchor='top', y=1.0)
                             ),
                    row=1, col=1, )
    
    # print(x_slider.value, z_slider.value)
    fig.add_trace(go.Scatter(x=[x_slider.value], y=[z_slider.value], 
                             mode='markers', name='Coords'), 
                  row=1, col=1)
    fig.add_scatter(x=full_time, y=processed_data,
                  row=2, col=1, name=dataset.dset_name)
    fig.update_yaxes(range=[-1, 1], row=2, col=1)
    fig.update_xaxes(title_text="Time (ns)", row=2, col=1)
    fig.update_yaxes(title_text="Voltage (normalized)", row=2, col=1)
    fig.add_vline(x=float(full_time[s_slider.value]), line_width=1.5, line_dash='dash',
                  line_color='red', row=2, col=1)
    
    # Set up layout
    fig.update_layout(
        title=dict(
            text=f'Z: {z_slider.value} X: {x_slider.value} Slice: {s_slider.value}',
            x=0.5,
            xanchor='center'
        ),
        height=600,
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=80, r=150, t=80, b=60),
        legend=dict(
            x=1.02,
            y=0.45,
            xanchor='left',
            yanchor='top'
        )
    )
    
    # Update function that gets called when slider changes
    def update_plot(change): 
        # Get new data based on slider value
        idx, processed_data = dataset[z_slider.value*dataset.image_shape[1]+x_slider.value]
        
        # Get time data for the current scan
        full_time = dataset.data[idx]['time']
        
        # Convert to lists if needed (FigureWidget sometimes needs lists)
        def to_list(arr):
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return list(arr) if not isinstance(arr, list) else arr
        
        # Update all subplots
        with fig.batch_update():
            fig.data[0].z = to_list(dataset[:,s_slider.value][1].reshape(dataset.image_shape))
            fig.data[1].x = to_list([x_slider.value])
            fig.data[1].y = to_list([z_slider.value])
            fig.data[2].x = to_list(full_time[dataset.crop[0]:dataset.crop[1]])
            fig.data[2].y = to_list(processed_data)
            fig.layout.shapes[0].x0 = float(full_time[s_slider.value])
            fig.layout.shapes[0].x1 = float(full_time[s_slider.value])

            # Update title
            fig.layout.title.text = f'Z: {z_slider.value} X: {x_slider.value} Slice: {s_slider.value}'
    
    # Connect the sliders to the update function
    z_slider.observe(update_plot, names='value')
    x_slider.observe(update_plot, names='value')
    s_slider.observe(update_plot, names='value')
    
    # Create container with slider and figure
    container = widgets.VBox([z_slider, x_slider, s_slider, fig])
    return container


def training_viewer(dset, fits, params, idx=0, crop_idx=0):
    """
    Visualization for comparing original data with fitted Morlet packet components.
    
    Args:
        dset: Dataset containing original data
        fits: Fitted data with shape [num_samples, num_channels, time_points]
        params: Parameters array with shape [num_samples, num_channels, 4] 
                where 4 represents [a, mu, sigma, omega]
        x: X coordinate (default: 0)
        y: Y coordinate (default: 0)
        crop_idx: Index of the crop to visualize (default: 0)
    
    Returns:
        plotly.graph_objects.Figure: Figure with scatter plot and parameter table
    """
    viridis = matplotlib.colormaps.get_cmap('viridis').resampled(fits.shape[1])
    
    # idx = y * dset.shape[0] + x
    
    # Get original data - dset[idx] returns (idx, processed_data)
    # processed_data has shape (len(numeric_keys), len(crops), crop_length)
    # We need to extract the data for the specific spatial location and crop
    _, processed_data = dset[idx]
    
    # Find the position of idx in numeric_keys to access the correct row
    scan_idx_in_array = dset.numeric_keys.index(idx) if hasattr(dset, 'numeric_keys') else idx
    
    # Extract the data for the specific crop
    # Handle different possible shapes of processed_data
    if len(processed_data.shape) == 3:
        # Shape: (len(numeric_keys), len(crops), crop_length)
        original_signal = processed_data[scan_idx_in_array, crop_idx, :]
    elif len(processed_data.shape) == 2:
        # Shape: (len(crops), crop_length) - already indexed by scan
        original_signal = processed_data[crop_idx, :]
    else:
        # Fallback: try to flatten or take first crop
        original_signal = processed_data.flatten() if processed_data.ndim > 1 else processed_data
    
    # Build parameter table data
    param_table = {'channel': [], 'a': [], 'mu': [], 'sigma': [], 'omega': []}
    for i in range(fits.shape[1]):  # Iterate over all channels
        param_table['channel'].append(i)
        param_table['a'].append(f'{params[idx][i][0]:.2f}')
        param_table['mu'].append(f'{params[idx][i][1]:.0f}')
        param_table['sigma'].append(f'{params[idx][i][2]:.0f}')
        param_table['omega'].append(f'{params[idx][i][3]:.2e}')
    
    # Highlight rows where a != 0
    cell_colors = ['#ffffb3' if float(a) != 0 else 'white' for a in param_table['a']]
    # Build table_fill_colors per Plotly spec: a list of lists of shape [num_columns][num_rows].
    table_fill_colors = [cell_colors]*5
    
    # --- Font colors section (must be shape [num_rows][num_columns] for Plotly) ---
    font_colors = [matplotlib.colors.rgb2hex(viridis(i)) for i in param_table['channel']]
    table_font_colors = [font_colors]*5
    
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        specs=[[{"type": "scatter"}, {"type": "table"}]]
    )
    
    # --- Plot traces ---
    # Original data
    fig.add_trace(go.Scatter(
        x=list(range(len(original_signal))), 
        y=original_signal,
        mode='lines', 
        name='original',
    ), row=1, col=1)
    
    # Sum of all fitted channels
    if fits.shape[1] > 0:
        fig.add_trace(go.Scatter(
            x=list(range(fits.shape[2])), 
            y=fits[idx].sum(axis=0),
            mode='lines', 
            name='sum',
        ), row=1, col=1)
    
    # Individual channel fits
    for i in range(fits.shape[1]):  # Iterate over all channels
        color = matplotlib.colors.rgb2hex(viridis(i))
        fig.add_trace(go.Scatter(
            x=list(range(fits.shape[2])), 
            y=fits[idx][i],
            mode='lines', 
            line=dict(dash='dot', width=1, color=color),
            name=f'Channel {i}',
            showlegend=False
        ), row=1, col=1)
    
    # --- Table ---
    fig.add_trace(go.Table(
        header=dict(values=['channel','a', 'mu', 'sigma', 'omega'],
                    fill_color='#4a4a4a',
                    font=dict(color='white'),
                    align='center'),
        cells=dict(values=[param_table['channel'],
                          param_table['a'], 
                          param_table['mu'],
                          param_table['sigma'], 
                          param_table['omega']],
                   fill_color=table_fill_colors,
                   font=dict(color=table_font_colors),  # shape: [rows][columns]
                   align='center')
    ), row=1, col=2)
    
    fig.update_layout(title='Comparing original and fitted Morlet packet (a, mu, sigma, omega)')
    return fig


def training_viewer_(dset, fits, params, idx=0, crop_idx=0):
    """
    Visualization for comparing original data with fitted Morlet packet components.
    
    Args:
        dset: Dataset containing original data
        fits: Fitted data with shape [num_samples, num_channels, time_points]
        params: Parameters array with shape [num_samples, num_channels, 4] 
                where 4 represents [a, mu, sigma, omega]
        x: X coordinate (default: 0)
        y: Y coordinate (default: 0)
        crop_idx: Index of the crop to visualize (default: 0)
    
    Returns:
        plotly.graph_objects.Figure: Figure with scatter plot and parameter table
    """
    viridis = matplotlib.colormaps.get_cmap('viridis').resampled(fits.shape[1])
    z_slider = widgets.IntSlider(
        description='Z #', 
        min=0, 
        max=dset.image_shape[0]-1, 
        step=1, 
        value=0
    )
    x_slider = widgets.IntSlider(
        description='X #', 
        min=0, 
        max=dset.image_shape[1]-1, 
        step=1, 
        value=0
    )
        
    full_time = dset.data[0]['time']

    s_slider = widgets.IntSlider(
        description='Slice #', 
        min=0, 
        max=len(full_time)-1, 
        step=1, 
        value=0
    )
    
    fit_slider = widgets.IntSlider(
        description='Fit #', 
        min=0, 
        max=fits.shape[1]-1, 
        step=1, 
        value=0
    )
    
    # Get initial data
    # idx, processed_data = dataset[i_slider.value]
    idx, processed_data = dset[z_slider.value*dset.image_shape[1]+x_slider.value]
    
    # Build parameter table data
    param_table = {'channel': [], 'a': [], 'mu': [], 'sigma': [], 'omega': []}
    for i in range(fits.shape[1]):  # Iterate over all channels
        param_table['channel'].append(i)
        param_table['a'].append(f'{params[idx][i][0]:.2f}')
        param_table['mu'].append(f'{params[idx][i][1]:.0f}')
        param_table['sigma'].append(f'{params[idx][i][2]:.0f}')
        param_table['omega'].append(f'{params[idx][i][3]:.2e}')
    
    # Highlight rows where a != 0
    cell_colors = ['#ffffb3' if float(a) != 0 else 'white' for a in param_table['a']]
    # Build table_fill_colors per Plotly spec: a list of lists of shape [num_columns][num_rows].
    table_fill_colors = [cell_colors]*5
    
    # --- Font colors section (must be shape [num_rows][num_columns] for Plotly) ---
    font_colors = [matplotlib.colors.rgb2hex(viridis(i)) for i in param_table['channel']]
    table_font_colors = [font_colors]*5
    
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.6, 0.4],
        specs=[[{"type": "scatter"}, {"type": "table"}]]
    )
    
    # --- Plot traces ---
    # fitted image
    fig.add_trace(go.Heatmap(z=fits[idx].reshape(dset.image_shape),
                             name='fitted',
                             zmin=-1,
                             zmax=1,
                             colorscale='Viridis',
                             colorbar=dict(x=1.02, len=0.5, yanchor='top', y=1.0)
                             ), row=1, col=2)
    
    # Original data
    fig.add_trace(go.Scatter(
        x=list(range(len(original_signal))), 
        y=original_signal,
        mode='lines', 
        name='original',
    ), row=1, col=1)
    
    # Sum of all fitted channels
    if fits.shape[1] > 0:
        fig.add_trace(go.Scatter(
            x=list(range(fits.shape[2])), 
            y=fits[idx].sum(axis=0),
            mode='lines', 
            name='sum',
        ), row=1, col=1)
    
    # Individual channel fits
    for i in range(fits.shape[1]):  # Iterate over all channels
        color = matplotlib.colors.rgb2hex(viridis(i))
        fig.add_trace(go.Scatter(
            x=list(range(fits.shape[2])), 
            y=fits[idx][i],
            mode='lines', 
            line=dict(dash='dot', width=1, color=color),
            name=f'Channel {i}',
            showlegend=False
        ), row=1, col=1)
    
    # --- Table ---
    fig.add_trace(go.Table(
        header=dict(values=['channel','a', 'mu', 'sigma', 'omega'],
                    fill_color='#4a4a4a',
                    font=dict(color='white'),
                    align='center'),
        cells=dict(values=[param_table['channel'],
                          param_table['a'], 
                          param_table['mu'],
                          param_table['sigma'], 
                          param_table['omega']],
                   fill_color=table_fill_colors,
                   font=dict(color=table_font_colors),  # shape: [rows][columns]
                   align='center')
    ), row=1, col=2)
    
    fig.update_layout(title='Comparing original and fitted Morlet packet (a, mu, sigma, omega)')
    return fig

