import h5py
import numpy as np
import holoviews as hv
import panel as pn

import param
from holoviews import streams
from functools import lru_cache

from bokeh.models import CustomJSTickFormatter

# Initialize HoloViews with Bokeh backend
hv.extension('bokeh')


def launch_visualizer(class_, **kwargs):
    """
    Launches the interactive dashboard for visualizing acoustic scan data.
    
    Parameters:
    -----------
    class_ : AcousticViewer class
        The class to instantiate for the visualizer.
    **kwargs : dict
        Additional keyword arguments to pass to the visualizer. May include:
        - database : AcousticDatabase class
        - from_group : str, optional
    """
    with class_(**kwargs) as visualizer:
        app = visualizer.create_dashboard()
        server = pn.serve(app, show=True, threaded=True)
        input("Use the viewer, then press Enter here to close it...")
        server.stop()
        
        
class AcousticScanViewer(param.Parameterized):
    current_x = param.Integer(default=0)
    current_z = param.Integer(default=0)
    current_y = param.Integer(default=0)

    # -------------------------------------------------------------------------
    # Magic Methods
    # -------------------------------------------------------------------------

    def __init__(self, db, from_group='preprocessed_data', **params):
        """
        Parameters:
        -----------
        db : AcousticDatabase class
        """
        super().__init__()
        self.db = db
        self.labels = db.parameters['waveform_labels']
        self.from_group = from_group
        self.aspect_ratio = 10#
        self._spatial_tap_streams = {}#
        self._waveform_tap_stream = None#
        
        self._is_open = False
        
        # Reactive global states for tracking crosshairs/slices without static sliders
        self.current_x = db.shape[1] // 2
        self.current_z = db.shape[2] // 2
        self.current_y = db.shape[3] // 2
        
        self.open_reader()

    def __enter__(self) -> AcousticScanViewer:
        self.open_reader()
        return self

    def __exit__(self):
        self.close()

    # -------------------------------------------------------------------------
    # Database management
    # -------------------------------------------------------------------------

    def open_reader(self):
        """Opens H5 file read-only and automatically builds dynamic indices."""
        if not self._is_open:
            self.h5 = h5py.File(self.db.h5_name, 'r')
            self.acoustic_ds = self.h5[f"/{self.from_group}/waveforms"] # open datasets
            self.hilbert_ds = self.h5[f"/{self.from_group}/hilbert"]
            self.time = self.acoustic_ds.attrs['time_array'] # get time
            self.shape = self.acoustic_ds.shape # get shape
            self._is_open = True
            
    def close(self):
        if self._is_open and self.h5 is not None:
            self.h5.close() # close dataset
            self.clear_cache() # clear cache
            self.h5 = None
            self._is_open = False

    # -------------------------------------------------------------------------
    # Data Fetchers
    # -------------------------------------------------------------------------

    def get_2d_slice(self, w: int, y: int, dataset: str = 'hilbert') -> np.ndarray:
        if dataset == 'hilbert': source = self.hilbert_ds
        elif dataset == 'acoustic': source = self.acoustic_ds
        else: raise ValueError("dataset must be 'hilbert' or 'acoustic'")
        return source[int(w), :, :, int(y)]

    @lru_cache(maxsize=128)
    def _fetch_1d_waveform(self, w: int, x: int, z: int): return self.acoustic_ds[w, x, z, :], self.hilbert_ds[w, x, z, :]

    def get_1d_waveform(self, w: int, x: int, z: int) -> tuple: return self._fetch_1d_waveform(int(w), int(x), -int(z))

    def clear_cache(self): self._fetch_1d_waveform.cache_clear()

    # -------------------------------------------------------------------------
    # 2D SPATIAL SCAN GRID COMPONENT
    # -------------------------------------------------------------------------
    
    #TODO: get rid of prints
    def build_spatial_grid(self, direction_idx: int, dataset: str = 'hilbert'):
        """
        Generates a 2D spatial scan grid for a specific waveform layout plane.
        Includes an active crosshair tracking where you click.
        """
        tap_stream = streams.Tap(x=self.current_x, y=self.current_z)

        def spatial_renderer(current_y, current_x, current_z, x, y):
            # Fetch slice matrix
            grid_data = self.get_2d_slice(direction_idx, current_y, dataset)
            
            # Create a HoloViews Image object for the 2D grid
            img = hv.Image(
                grid_data.T, 
                kdims=['x', 'z'], 
                vdims=['Intensity'],
                bounds=(0, 0, self.shape[1], self.shape[2]) #(x0, y0, x1, y1) # changing has no effect
            ).opts(
                title=f"{self.labels[direction_idx]} {dataset.title()} Grid (Time Y={current_y})",
                cmap='Viridis', colorbar=True, width=self.shape[1]*self.aspect_ratio, height=self.shape[2]*self.aspect_ratio,
                tools=['tap', 'hover'],
            )
            
            # Crosshair Overlay lines mapping active interaction coordinates
            crosshair = hv.VLine(current_x).opts(color='white', line_dash='dashed', line_width=1.5) * \
                        hv.HLine(current_z).opts(color='white', line_dash='dashed', line_width=1.5)
            
            return img * crosshair
            
        state_stream = streams.Params(
            parameterized=self,
            parameters=['current_y', 'current_x', 'current_z'],
        )
        dynamic_grid = hv.DynamicMap(
            spatial_renderer,
            streams=[state_stream, tap_stream],
        )

        def update_coordinates(x, y):
            if x is not None and y is not None:
                self.current_x = int(np.clip(x, 0, self.shape[1] - 1))
                self.current_z = int(np.clip(y, 0, self.shape[2] - 1))

        tap_stream.add_subscriber(update_coordinates)
        self._spatial_tap_streams[direction_idx] = tap_stream
        return dynamic_grid

    # -------------------------------------------------------------------------
    # 1D WAVEFORM OVERLAY VIEW
    # -------------------------------------------------------------------------
    
    def build_overlay_waveform_plot(self):
        """
        Plots waveforms for ALL labels overlayed in a single multi-colored layout graph.
        Includes a dynamic vertical indicator line showing the selected time step.
        """
        # Distinguishable color palette for tracking multiple waveform variations
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        time_formatter = CustomJSTickFormatter( 
            args={'times': self.time.tolist()},
            code="""
                const i = Math.round(tick);
                return i >= 0 && i < times.length
                    ? `${times[i]} ns`
                    : "";
                """ )
        line_tap = streams.Tap(x=self.current_y)

        def waveform_renderer(current_x, current_z, current_y, x, y):
            time_axis = np.arange(self.shape[3])
            overlay_elements = []
            
            for w_idx in range(self.shape[0]):
                acoustic_wave, hilbert_win = self.get_1d_waveform(w_idx, current_x, current_z)
                color = palette[w_idx % len(palette)]
                label_str = self.labels[w_idx]
                
                # Raw trace
                curve = hv.Curve(
                    (time_axis, acoustic_wave), kdims=['Time (Y)'], vdims=['Amplitude'], 
                    label=f"{label_str} Acoustic"
                ).opts(color=color, tools=['tap'], alpha=0.8)
                
                # Matching Hilbert trace (uses same color with a distinct dash pattern)
                envelope = hv.Curve(
                    (time_axis, hilbert_win), kdims=['Time (Y)'], vdims=['Amplitude'], 
                    label=f"{label_str} Hilbert"
                ).opts(color=color, line_dash='dotdash', alpha=0.5)
                
                overlay_elements.extend([curve, envelope])
                
            # Vertical time step tracking line
            time_marker = hv.VLine(current_y).opts(color='red', line_width=2)
            overlay_elements.append(time_marker)
            
            return hv.Overlay(overlay_elements).opts(
                title=f"Waveform Comparison at Node (X: {current_x}, Z: {current_z})",
                width=self.shape[1]*self.aspect_ratio, height=self.shape[2]*self.aspect_ratio, 
                legend_position='right', show_legend=True, xformatter=time_formatter,

            )
            
        state_stream = streams.Params(
            parameterized=self,
            parameters=['current_x', 'current_z', 'current_y'],
        )
        waveform_map = hv.DynamicMap(
            waveform_renderer,
            streams=[state_stream, line_tap],
        )

        def update_time(x, y):
            if x is not None: self.current_y = int(np.clip(x, 0, self.shape[3] - 1))

        line_tap.add_subscriber(update_time)
        self._waveform_tap_stream = line_tap
        return waveform_map

    # -------------------------------------------------------------------------
    # MAIN LAYOUT COMPOSER
    # -------------------------------------------------------------------------
    
    def create_dashboard(self):
        """Assembles dynamically computed items into an integrated control dashboard."""
        # Create an individual 2D plot panel row matching each waveform array slice index
        grid_plots = [self.build_spatial_grid(w) for w in range(self.shape[0])]

        # Generate the shared, synchronized 1D timeline trace
        waveform_plot = self.build_overlay_waveform_plot()

        # Format elements cleanly into columns & rows
        dashboard_layout = pn.Column(
            pn.pane.Markdown(f"## 📊 Scanned waveform visualization, ({self.shape[0]} channels)"),
            pn.pane.Markdown("*Click an image to center the spatial crosshair. Click the line plot to select a specific time slice.*"),
            pn.Column(
                pn.Column(*grid_plots), # Stacks image dimensions vertically based on array count
                pn.Card(waveform_plot, title="Synchronized Waveform Array View", margin=(0,10))
            )
        )
        return dashboard_layout





