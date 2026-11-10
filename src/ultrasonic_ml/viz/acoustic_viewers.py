import h5py
import numpy as np
from functools import lru_cache

import param
import panel as pn
import holoviews as hv
from holoviews import streams

from bokeh.models import CustomJSTickFormatter



class AcousticViewer(param.Parameterized):
    # Put tracked parameters here

    # -------------------------------------------------------------------------
    # Magic Methods
    # -------------------------------------------------------------------------
    
    def __init__(self, db, from_group='preprocessed_data', **params):
        super().__init__()
        self.db = db
        self.from_group = from_group
        self.labels = db.parameters['waveform_labels']
        # TODO: open time here?
        self.h5 = None
        self._is_open = False
    
        self.open()
                
        # Distinguishable color palette for tracking multiple waveform variations       
        self.palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        self.time_formatter = CustomJSTickFormatter( 
            args={'times': self.time.tolist()},
            code="""
                const i = Math.round(tick);
                return i >= 0 && i < times.length
                    ? `${times[i]/1e3}`
                    : "";
                """ )
        
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -------------------------------------------------------------------------
    # Database management
    # -------------------------------------------------------------------------
    
    def open(self):
        if self._is_open: return

        self.h5 = h5py.File(self.db.h5_name, 'r')
        self._open_datasets()
        self._is_open = True

    def close(self):
        if not self._is_open: return
        self._close_datasets()
        self.clear_cache()
        if self.h5 is not None: self.h5.close()
        self.h5 = None
        self._is_open = False
        print("AcousticViewer closed and cache cleared.")

    def clear_cache(self): pass # call .cache_clear() on all methods with header @lrucache

    def _open_datasets(self): raise NotImplementedError # methods for setting self.datasets, self.shape, self.time,

    def _close_datasets(self): pass
    
    # -------------------------------------------------------------------------
    # Data Fetchers/ Interactive grid components
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Dashboard
    # -------------------------------------------------------------------------
    
    def create_dashboard(self): raise NotImplementedError

    @classmethod
    def launch(cls, **kwargs):
        with cls(**kwargs) as visualizer:
            server = pn.serve( visualizer.create_dashboard(), show=True, threaded=True )
            try: input("Use the viewer, then press Enter here to close it...")
            finally: server.stop()
            
            
class AcousticScanViewer(AcousticViewer):
    dataset = param.Selector(default='hilbert', objects=['hilbert', 'waveforms', ])
    current_x = param.Integer(default=0)
    current_z = param.Integer(default=0)
    current_y = param.Integer(default=0)
    
    # -------------------------------------------------------------------------
    # Magic Methods
    # -------------------------------------------------------------------------
    
    def __init__(self, db, from_group = 'preprocessed_data', aspect_ratio = 10, **params):
        super().__init__(db, from_group = from_group, **params)
        
        self.aspect_ratio = aspect_ratio
        self._spatial_tap_streams = {}
        self._waveform_tap_stream = None
        
        self.current_x = self.shape[1] // 2
        self.current_z = self.shape[2] // 2
        self.current_y = self.shape[3] // 2
        self.time_axis = np.arange(len(self.time))

    # -------------------------------------------------------------------------
    # Database management
    # -------------------------------------------------------------------------

    def clear_cache(self): self._fetch_1d_waveform.cache_clear()    
    
    def _open_datasets(self):
        group = f"/{self.from_group}"
        self.waveforms = self.h5[f"{group}/waveforms"]
        self.hilbert = self.h5[f"{group}/hilbert"]
        
        self.shape = self.h5[f"{group}/waveforms"].shape
        self.time = self.waveforms.attrs['time_array']

    def _close_datasets(self):
        self.waveforms = None
        self.hilbert = None
    
    # -------------------------------------------------------------------------
    # Data Fetchers
    # -------------------------------------------------------------------------

    def get_2d_slice(self, w: int, y: int, dataset: str = 'hilbert') -> np.ndarray:
        source = getattr(self, f'{dataset}', None)
        if source is None:
            source = getattr(self, dataset, None)
        if source is None:
            raise ValueError(f"Unknown dataset: {dataset}")
        return source[int(w), :, :, int(y)]

    @lru_cache(maxsize=128)
    def _fetch_1d_waveform(self, w: int, x: int, z: int): return self.waveforms[w, x, z, :], self.hilbert[w, x, z, :]

    def get_1d_waveform(self, w: int, x: int, z: int) -> tuple: return self._fetch_1d_waveform(int(w), int(x), -int(z))
    
    # -------------------------------------------------------------------------
    # 2D Image Plot 
    # -------------------------------------------------------------------------
    
    #TODO: get rid of prints
    def build_spatial_grid(self, direction_idx: int):
        """
        Generates a 2D spatial scan grid for a specific waveform layout plane.
        Includes an active crosshair tracking where you click.
        """
        tap_stream = streams.Tap(x=self.current_x, y=self.current_z)

        def spatial_renderer(current_y, current_x, current_z, dataset, x, y):
            # Fetch slice matrix
            grid_data = self.get_2d_slice(direction_idx, current_y, dataset)
            
            # Create a HoloViews Image object for the 2D grid
            img = hv.Image(
                grid_data.T, 
                kdims=['x', 'z'], 
                vdims=['Intensity'],
                bounds=(0, 0, self.shape[1], self.shape[2]) #(x0, y0, x1, y1) # changing has no effect
            ).opts(
                title=f"{self.labels[direction_idx]} {dataset.title()} Grid (Y={current_y})",
                cmap='Viridis', colorbar=True, width=self.shape[1]*self.aspect_ratio, height=self.shape[2]*self.aspect_ratio,
                tools=['tap', 'hover'],
            )
            
            # Crosshair Overlay lines mapping active interaction coordinates
            crosshair = hv.VLine(current_x).opts(color='white', line_dash='dashed', line_width=1.5) * \
                        hv.HLine(current_z).opts(color='white', line_dash='dashed', line_width=1.5)
            
            return img * crosshair
            
        state_stream = streams.Params(
            parameterized=self,
            parameters=['current_y', 'current_x', 'current_z', 'dataset'],
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
    # 1D Waveform Plot
    # -------------------------------------------------------------------------
    
    def build_overlay_waveform_plot(self):
        """
        Plots waveforms for ALL labels overlayed in a single multi-colored layout graph.
        Includes a dynamic vertical indicator line showing the selected time step.
        """

        line_tap = streams.Tap(x=self.current_y)

        def waveform_renderer(current_x, current_z, current_y, x, y):
            overlay_elements = []
            
            for w_idx in range(self.shape[0]):
                acoustic_wave, hilbert_win = self.get_1d_waveform(w_idx, current_x, current_z)
                color = self.palette[w_idx % len(self.palette)]
                label_str = self.labels[w_idx]
                
                # Raw trace
                curve = hv.Curve(
                    (self.time_axis, acoustic_wave), kdims=['Time (Y)'], vdims=['Amplitude'], 
                    label=f"{label_str} Acoustic"
                ).opts(color=color, tools=['tap'], alpha=0.8)
                
                # Matching Hilbert trace (uses same color with a distinct dash pattern)
                envelope = hv.Curve(
                    (self.time_axis, hilbert_win), kdims=['Time (us)'], vdims=['Amplitude'], 
                    label=f"{label_str} Hilbert"
                ).opts(color=color, line_dash='dotdash', alpha=0.5)
                
                overlay_elements.extend([curve, envelope])
                
            # Vertical time step tracking line
            time_marker = hv.VLine(current_y).opts(color='red', line_width=2)
            overlay_elements.append(time_marker)
            
            return hv.Overlay(overlay_elements).opts(
                title=f"Waveform Comparison at Node (X: {current_x}, Z: {current_z})",
                width=self.shape[1]*self.aspect_ratio, height=self.shape[2]*self.aspect_ratio, 
                legend_position='right', show_legend=True, xformatter=self.time_formatter,

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
    # Dashboard
    # -------------------------------------------------------------------------
    def create_dashboard(self):
        """Assembles dynamically computed items into an integrated control dashboard."""
        dataset_widget = pn.widgets.Select(
            name='Dataset', options=list(self.param.dataset.objects), value=self.dataset
        )
        dataset_widget.link(self, value='dataset')

        # Create an individual 2D plot panel row matching each waveform array slice index
        grid_plots = [self.build_spatial_grid(w) for w in range(self.shape[0])]

        # Generate the shared, synchronized 1D timeline trace
        waveform_plot = self.build_overlay_waveform_plot()

        # Format elements cleanly into columns & rows
        dashboard_layout = pn.Column(
            pn.pane.Markdown(f"## 📊 Scanned waveform visualization, ({self.shape[0]} channels)"),
            pn.pane.Markdown("*Click an image to center the spatial crosshair. Click the line plot to select a specific time slice.*"),
            dataset_widget,
            pn.Column(
                pn.Column(*grid_plots), # Stacks image dimensions vertically based on array count
                pn.Card(waveform_plot, title="Synchronized Waveform Array View", margin=(0,10))
            )
        )
        return dashboard_layout

      
class AcousticScanViewerFFT(AcousticScanViewer):# TODO: fix shaping issues
    # current_x = param.Integer(default=0)
    # current_z = param.Integer(default=0)
    # current_y = param.Integer(default=0)
    dataset_options = ['waveforms', 'hilbert', 'auto_corr', 'x_corr', 'tof', 'amplitude', 'fft_magnitude', 'fft_phase', 'fft_group_delay']
    dataset = param.Selector(default='waveforms', objects=dataset_options)
    
    # -------------------------------------------------------------------------
    # Magic Methods
    # -------------------------------------------------------------------------
    
    def __init__(self, db, aspect_ratio=10, crop=[0, 2000], from_group=None, **params): # GOOD
        self.crop = crop # TODO: For now, about 5*1/2.25MHz/2e-9; eventually automatically group to 1/(self.parameters['transducerFrequency'])/(self.db.parameters['dt'])
        
        super().__init__(db, from_group=from_group, aspect_ratio=aspect_ratio, **params)
        
        self._dropdown_stream = None # which plot to display
        
        self.current_plot = 'hilbert' # which plot to display, 0-3
        self.freq_axis = np.arange(crop[1]-crop[0])
        self.freq_formatter = CustomJSTickFormatter( 
            args={'freqs': self.freq.tolist()},
            code="""
                const i = Math.round(tick);
                return i >= 0 && i < freqs.length
                    ? `${(freqs[i]/1e6).toFixed(2)}`
                    : "";
                """ )
        
    # -------------------------------------------------------------------------
    # Database management
    # -------------------------------------------------------------------------

    def clear_cache(self): self._fetch_1d_waveform.cache_clear()    
    
    def _open_datasets(self): # GOOD
        group = f"/{self.from_group}"
        for name in self.dataset_options:
            path = f"{group}/{name}"
            if path in self.h5: setattr(self, name, self.h5[path])

        self.waveforms = self.h5[f"{group}/waveforms"]
        self.hilbert = self.h5[f"{group}/hilbert"]

        print(f"Opening FFT datasets from group: {self.from_group}")
        self.fft = self.h5[f"{self.from_group}/fft_data/"] # since input is real, these are only the real portion of fft.
        self.fft_magnitude = self.db._abs(self.fft[..., self.crop[0]:self.crop[1]]) # TODO: decide whether disp divided by len?
        phase = self.db._angle(self.fft)
        self.fft_phase = phase[..., self.crop[0]:self.crop[1]]        
        self.fft_group_delay = self.db._group_delay(phase, self.fft.attrs['frequencies'])[..., self.crop[0]:self.crop[1]]
    
        self.shape = self.h5[f"{self.from_group}/waveforms"].shape
        self.time = self.waveforms.attrs['time_array'] # TODO: /1e3 here (ns->us) or change axis formatter?
        self.freq = self.fft.attrs['frequencies'] # TODO: do you want to /1e6 (Hz->MHz) here or change axis formatter?
        self.freq_axis = np.arange(self.crop[1]-self.crop[0])

    def _close_datasets(self):
        self.waveforms = None
        self.hilbert = None
        self.fft = None
        self.fft_magnitude = None
        self.fft_phase = None
        self.fft_group_delay = None
    
    # -------------------------------------------------------------------------
    # Data Fetchers - TODO: different cases and indexing for tof, amplitude, and fft plot displays
    # -------------------------------------------------------------------------

    def get_2d_slice(self, w: int, y: int, dataset: str = 'hilbert') -> np.ndarray: # GOOD
        '''Fetch a 2D slice from the specified dataset.
        
        -----------
        Parameters:
            w (int): Index of the waveform.
            y (int): Index of the time step.
            dataset (str): Dataset to fetch from.
        
        --------
        Returns:
            np.ndarray: 2D slice of the specified dataset.
        '''
        source = getattr(self, dataset)
        if source is None:
            raise ValueError(
                "dataset must be 'waveforms', 'hilbert',  'tof', 'amplitude', "
                "'magnitude', 'phase', or 'group_delay'"
            )
        if source == 'tof' or source == 'amplitude':
            return source[int(w)]
        
        return source[int(w), :, :, int(y)]

    # -------------------------------------------------------------------------
    # 2D Image Plot 
    # -------------------------------------------------------------------------
      
    # -------------------------------------------------------------------------
    # 1D Waveform Plots
    # -------------------------------------------------------------------------
    
    def build_overlay_magnitude_plot(self):
        """
        Plots waveforms for ALL labels overlayed in a single multi-colored layout graph.
        Includes a dynamic vertical indicator line showing the selected time step.
        """
        freq_line_tap = streams.Tap(x=self.current_y)

        def waveform_renderer(current_x, current_z, current_y, x, y):
            overlay_elements = []
            
            for w_idx in range(self.shape[0]):
                wave = self.fft_magnitude[w_idx, current_x, current_z]
                color = self.palette[w_idx % len(self.palette)]
                label_str = self.labels[w_idx]
                
                # Raw trace
                curve = hv.Curve(
                    (self.freq_axis, wave), kdims=['Frequency (MHz)'], vdims=['Amplitude (au)'], 
                    label=f"{label_str} Acoustic"
                ).opts(color=color, tools=['tap'], alpha=0.8)
                
                overlay_elements.extend([curve])
                
            # Vertical time step tracking line
            time_marker = hv.VLine(current_y).opts(color='red', line_width=2)
            overlay_elements.append(time_marker)
            
            return hv.Overlay(overlay_elements).opts(
                title=f"Waveform Comparison at Node (X: {current_x}, Z: {current_z})",
                width=self.shape[1]*self.aspect_ratio, height=self.shape[2]*self.aspect_ratio, 
                legend_position='right', show_legend=True, xformatter=self.freq_formatter,
                )
            
        state_stream = streams.Params(
            parameterized=self,
            parameters=['current_x', 'current_z', 'current_y'],
        )
        waveform_map = hv.DynamicMap(
            waveform_renderer,
            streams=[state_stream, freq_line_tap],
        )
        
        def update_time(x, y): 
            if x is not None: self.current_y = int(np.clip(x, 0, self.shape[3] - 1))

        freq_line_tap.add_subscriber(update_time)
        self._waveform_tap_stream = freq_line_tap
        return waveform_map

    def build_overlay_phase_plot(self):
        """
        Plots waveforms for ALL labels overlayed in a single multi-colored layout graph.
        Includes a dynamic vertical indicator line showing the selected time step.
        """
        freq_line_tap = streams.Tap(x=self.current_y)
        def waveform_renderer(current_x, current_z, current_y, x, y):
            overlay_elements = []
            
            for w_idx in range(self.shape[0]):
                wave = self.fft_phase[w_idx, current_x, current_z]
                color = self.palette[w_idx % len(self.palette)]
                label_str = self.labels[w_idx]
                
                # Raw trace
                curve = hv.Curve(
                    (self.freq_axis, wave), kdims=['Frequency (MHz)'], vdims=['Angle (rad)'], 
                    label=f"{label_str} Acoustic"
                ).opts(color=color, tools=['tap'], alpha=0.8, ylim=(-np.pi, np.pi))
                
                overlay_elements.extend([curve])
                
            # Vertical time step tracking line
            time_marker = hv.VLine(current_y).opts(color='red', line_width=2)
            overlay_elements.append(time_marker)
            
            return hv.Overlay(overlay_elements).opts(
                title=f"Waveform Comparison at Node (X: {current_x}, Z: {current_z})",
                width=self.shape[1]*self.aspect_ratio, height=self.shape[2]*self.aspect_ratio, 
                legend_position='right', show_legend=True, xformatter=self.freq_formatter,
            )
            
        state_stream = streams.Params(
            parameterized=self,
            parameters=['current_x', 'current_z', 'current_y'],
        )
        waveform_map = hv.DynamicMap(
            waveform_renderer,
            streams=[state_stream, freq_line_tap],
        )

        def update_time(x, y):
            if x is not None: self.current_y = int(np.clip(x, 0, self.shape[3] - 1))

        freq_line_tap.add_subscriber(update_time)
        self._waveform_tap_stream = freq_line_tap
        return waveform_map

    def build_overlay_group_delay_plot(self):
        """
        Plots waveforms for ALL labels overlayed in a single multi-colored layout graph.
        Includes a dynamic vertical indicator line showing the selected time step.
        """

        freq_line_tap = streams.Tap(x=self.current_y)

        def waveform_renderer(current_x, current_z, current_y, x, y):
            overlay_elements = []
            plotted_values = []
            
            for w_idx in range(self.shape[0]):
                wave = self.fft_group_delay[w_idx, current_x, current_z]
                plotted_values.append(np.asarray(wave)[np.isfinite(wave)])
                color = self.palette[w_idx % len(self.palette)]
                label_str = self.labels[w_idx]
                
                # Raw trace
                curve = hv.Curve(
                    (self.freq_axis, wave), kdims=['Frequency (MHz)'], vdims=['Delay (s)'], 
                    label=f"{label_str} Acoustic"
                ).opts(color=color, tools=['tap'], alpha=0.8, ylim=(wave.mean()-5*wave.std(), wave.mean()+5*wave.std()))
                
                overlay_elements.extend([curve])

            # HoloViews otherwise uses the absolute min/max, so a single
            # group-delay outlier can make the y-axis much larger than the
            # range containing the plotted traces.
            plotted_values = np.concatenate(plotted_values)
            if plotted_values.size:
                y_min, y_max = np.nanpercentile(plotted_values, [1, 99])
                padding = max((y_max - y_min) * 0.05, 1e-12)
                ylim = (y_min - padding, y_max + padding)
            else:
                ylim = None
                
            # Vertical time step tracking line
            time_marker = hv.VLine(current_y).opts(color='red', line_width=2)
            overlay_elements.append(time_marker)
            
            return hv.Overlay(overlay_elements).opts(
                title=f"Waveform Comparison at Node (X: {current_x}, Z: {current_z})",
                width=self.shape[1]*self.aspect_ratio, height=self.shape[2]*self.aspect_ratio, 
                legend_position='right', show_legend=True, xformatter=self.freq_formatter,
                ylim=ylim,

            )
            
        state_stream = streams.Params(
            parameterized=self,
            parameters=['current_x', 'current_z', 'current_y'],
        )
        waveform_map = hv.DynamicMap(
            waveform_renderer,
            streams=[state_stream, freq_line_tap],
        )

        def update_time(x, y):
            if x is not None: self.current_y = int(np.clip(x, 0, self.shape[3] - 1))

        freq_line_tap.add_subscriber(update_time)
        self._waveform_tap_stream = freq_line_tap
        return waveform_map

    # -------------------------------------------------------------------------
    # Dashboards
    # -------------------------------------------------------------------------
    
    def create_dashboard(self):
        """Assembles dynamically computed items into an integrated control dashboard."""
        # Create an individual 2D plot panel row matching each waveform array slice index        
        dataset_widget = pn.widgets.Select( name='Dataset', options=list(self.param.dataset.objects), value=self.dataset )
        dataset_widget.link(self, value='dataset')
        
        grid_plots = [self.build_spatial_grid(w) for w in range(self.shape[0])]
        spatial_panels = pn.Column(pn.Row(*grid_plots[:2]), 
                                   pn.Row(*grid_plots[2:])) # Stacks image dimensions vertically based on array count

        # Generate the shared, synchronized 1D timeline trace
        waveform_panels = pn.Column( pn.Row(self.build_overlay_waveform_plot(), self.build_overlay_magnitude_plot()),
                                     pn.Row(self.build_overlay_phase_plot(), self.build_overlay_group_delay_plot()) )
        
        # Format elements cleanly into columns & rows
        dashboard_layout = pn.Column(
            pn.pane.Markdown(f"## 📊 Scanned waveform visualization, ({self.shape[0]} channels)"),
            pn.pane.Markdown("*Click an image to center the spatial crosshair. Click the line plot to select a specific time slice. Select dataset to view in dropdown.*"),
            dataset_widget,
            spatial_panels,
            waveform_panels
        )
        return dashboard_layout



