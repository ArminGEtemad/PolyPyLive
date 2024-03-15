import os
import ctypes
import logging
import queue
import signalsnap as snp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from picosdk.ps4000a import ps4000a
from picosdk.functions import assert_pico_ok
from time import sleep

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors

logging.basicConfig(level=logging.INFO)

def set_ticks_for_plots(labelsize=22, majorsize=3.5*3, minorsize=2*3, major_width=0.8*2, minor_width=0.6*2):
        '''
        Verändert die Größe der x- und y-Ticks.
        Alle Werte sind mit Augenmaß erstellt worden

        @param labelsize: Größe der angezeigten Zahlen auf den Achsen
        @param majorsize: Größe der großen Striche
        @param major_width: Breite der großen Striche
        @param minorsize: Größe der kleinen Striche
        @param minor_width: Breite der kleinen Striche
        '''
        rc('xtick', labelsize=labelsize)
        rc('xtick.major', size=majorsize, width=major_width)
        rc('xtick.minor', size=minorsize, width=minor_width)
        rc('ytick', labelsize=labelsize)
        rc('ytick.major', size=majorsize, width=major_width)
        rc('ytick.minor', size=minorsize, width=minor_width)
        rc('axes', labelsize=labelsize, titlesize=labelsize)

set_ticks_for_plots()

def unit_converter(time_unit):
    """
    Helper function automatically converts units. What time unit can you use
    depends on the hardware (PicoScope) you are using.

    Parameters
    ----------
        - time_unit : str
            time unit

    Returns
    -------
        - freq_unit : str
            the corresponding frequency unit.
    """

    if time_unit == 's':
        freq_unit = 'Hz'
    elif time_unit == 'ms':
        freq_unit = 'kHz'
    elif time_unit == 'us':
        freq_unit = 'MHz'
    elif time_unit == 'ns':
        freq_unit = 'GHz'
    elif time_unit == 'ps':
        freq_unit = 'THz'
    elif time_unit == 'fs':
        freq_unit = 'PHz'

    return freq_unit


def arcsinh_scale(s_data, arcsinh_const=0.02):
    '''
    Helper function to scale data in arcsinh to improve the visibility of plot
    values. It is similar to log scale but with the advatage of working for
    negative values.

    Parameters
    ----------
        - s_data :
            the unscaled data
        - arcsinh_const : float, optional
            Constant to set amount of arcsinh scaling. The lower, the stronger.
            Default is 0.02.

    Returns
    -------
        - s_data:
            scaled data
    '''

    x_max = np.max(np.abs(s_data))
    alpha = 1 / (x_max * arcsinh_const)
    s_data = np.arcsinh(alpha * s_data) / alpha

    return s_data

def adc_converter(chunk, voltage_range_value, max_adc):
    """
    Helper function converts ADC counts to voltage in millivolts (mV).

    Parameters
    ----------
        - chunk : array-like, read-only
            Array of ADC counts to be converted.
        - voltage_range_value : float, read-only
            Maximum voltage range value corresponding to max ADC count.
        - max_adc : enum, read-only
            Maximum ADC count value, should be an enum with a 'value' attribute.

    Returns
    -------
    - converted_chunk : array-like
        Array of converted values in millivolts (mV).
    """
    # ---------calculate ADC to mV-----------
    # Voltage = (ADC Reading / ADC Resolution) * Reference Voltage

    max_adc_value = max_adc.value
    converted_chunk = (chunk * voltage_range_value) / max_adc_value
    
    return converted_chunk


class ChannelConfig:
    """
    A class to configure and validate channels for data acquisition.

    Class Attributes
    ----------------
        - RANGE_VALUES : list
            Available voltage ranges in millivolts for use in `_adc_converter` function only.
        - CHANNEL : dict
            Mappings of channel names to their IDs.
        - ACTIVITY : dict
            Mappings of activity statuses to their IDs.
        - RANGE_MAP : dict
            Mappings of voltage ranges to their IDs.
        - COUPLING : dict
            Mappings of coupling types to their IDs.

    Instance Attributes
    -------------------
        - channel_name : str
            The name of the channel.
        - range_map_name : str
            The name of the voltage range map.
        - coupling_name : str
            The name of the coupling type.
        - offset_str : str
            The voltage offset as a string.
        - channel : int
            The ID of the channel.
        - activity : int
            The ID of the activity status.
        - range_map : int
            The ID of the voltage range map.
        - coupling : int
            The ID of the coupling type.
        - v_offset : ctypes.c_float
            The voltage offset.
        - voltage_range_value : int
            The value of the voltage range in millivolts.
    """

    RANGE_VALUES = [10, 20, 50, 100,
                    200, 500, 1000, 2000,
                    5000, 10000, 20000, 50000,
                    100000, 200000]

    CHANNEL = {'A': 0,
               'B': 1,
               'C': 2,
               'D': 3}

    ACTIVITY = {'inactive': 0,
                'active': 1}

    RANGE_MAP = {'10mV': 0,
                 '20mV': 1,
                 '50mV': 2,
                 '100mV': 3,
                 '200mV': 4,
                 '500mV': 5,
                 '1V': 6,
                 '2V': 7,
                 '5V': 8,
                 '10V': 9,
                 '20V': 10,
                 '50V': 11,
                 '100V': 12,
                 '200V': 13}
    
    COUPLING = {'AC': 0,
                'DC': 1}

    def __init__(self, channel, activity, range_map, coupling, v_offset):
        if channel not in self.CHANNEL:
            raise ValueError(f'Invalid Channel {channel}. '
                             f'Must be one of {list(self.CHANNEL.keys())}')

        if activity not in self.ACTIVITY:
            raise ValueError(f'Invalid activity {activity}. '
                             f'Must be one of {list(self.ACTIVITY.keys())}')
        
        if range_map not in self.RANGE_MAP:
            raise ValueError(f'Invalid range_map {range_map}. '
                             f'Must be one of {list(self.RANGE_MAP.keys())}')
        
        if coupling not in self.COUPLING:
            raise ValueError(f'Invalid coupling {coupling}. '
                             f'Must be one of {list(self.COUPLING.keys())}')
        
        self.channel_name = channel
        self.range_map_name = range_map
        self.coupling_name = coupling
        self.offset_str = str(v_offset)
        

        self.channel = self.CHANNEL[channel]
        self.activity = self.ACTIVITY[activity]
        self.range_map = self.RANGE_MAP[range_map]
        self.coupling = self.COUPLING[coupling]
        self.v_offset = ctypes.c_float(v_offset)
        self.voltage_range_value = self.RANGE_VALUES[self.range_map]

    @staticmethod
    def configuration_dict(channel_config_list):
        return {config.channel: config for config in channel_config_list}
    

class DataAcqConfig:
    TIME_UNIT = {'fs': 0,
                 'ps': 1,
                 'ns': 2,
                 'us': 3,
                 'ms': 4,
                 's': 5}
    
    def __init__(self, buffer_size, buffer_to_capture, sampling_interval, time_unit, channel_config,
                 save=False, save_path=None, segment_idx=0, downsampling_ratio=1):

        if not isinstance(buffer_size, int) or buffer_size <= 0:
            raise ValueError(f'Buffer_size must be a positive interger. '
                             f'Not {type(buffer_size).__name__}')
        
        if buffer_to_capture is not None:
            if not isinstance(buffer_to_capture, int) or buffer_to_capture <= 0:
                raise ValueError('buffer_to_capture must be either None or a positive integer. '
                                 'Put None for unlimted streaming or positive int for limited.')
        
        if not isinstance(sampling_interval, int) or sampling_interval <= 0:
            raise ValueError(f'The sampling time is a positive integer. '
                             f'Not {type(sampling_interval).__name__}')
        
        if time_unit not in self.TIME_UNIT:
            raise ValueError(f'Invalid time_unit {time_unit}. '
                             f'Must be one of {list(self.TIME_UNIT.keys())}')

        self.buffer_size = buffer_size
        self.buffer_to_capture = buffer_to_capture
        self.sampling_interval = ctypes.c_int32(sampling_interval)
        self.time_unit = self.TIME_UNIT[time_unit]
        self.segment_idx = segment_idx
        
        self.save = save
        if save:
            self.save_path = save_path

        self.time_unit_str = time_unit
        self.downsampling_ratio = downsampling_ratio # no downsampling, hence = 1
        if buffer_to_capture is None:
            self.capture_limit = None
        else:
            self.capture_limit = buffer_size * buffer_to_capture
        

        self.buffer_max = {channel: np.zeros(shape=buffer_size, dtype=np.int16)
                           for channel, config in channel_config.items()
                           if config.activity == ChannelConfig.ACTIVITY['active']}

        self.freq_unit = unit_converter(time_unit)
        if time_unit == 'fs':
            self.capture_time = buffer_size * sampling_interval * 1E-15
        elif time_unit == 'ps':
            self.capture_time = buffer_size * sampling_interval * 1E-12
        elif time_unit == 'ns':
            self.capture_time = buffer_size * sampling_interval * 1E-9
        elif time_unit == 'us':
            self.capture_time = buffer_size * sampling_interval * 1E-6
        elif time_unit == 'ms':
            self.capture_time = buffer_size * sampling_interval * 1E-3
        elif time_unit == 's':
            self.capture_time = buffer_size * sampling_interval


class SignalConfig:
    '''
    Configuration class for signal processing settings. This class stores settings
    related to signal type, processing backend, and frequency filters.

    Class Attributes
    ----------------
    SIGNALS : dict
        A dictionary mapping signal names to their corresponding IDs.
    BACKENDS : dict
        A dictionary mapping backend names to their descriptive strings.

    Attributes
    ----------
    signal_choice : str
        The chosen signal type from the SIGNALS dictionary.
    signal_choice_ID : int
        The ID associated with the chosen signal type.
    backend : str
        The chosen processing backend from the BACKENDS dictionary.
    f_max : float
        The maximum frequency for the signal filter.
    f_min : float, optional
        The minimum frequency for the signal filter. Default is 0.0. Changing this number
        excludes S3 from the calculations
    m : int, optional
        Number of windows for the estimation of the cumulant
    m_var : int, optional
    coherent : bool, optional
        Indicates whether coherent signal processing is enabled. Default is False.
    '''

    SIGNALS = {'S2': 2,
               'S3': 3,
               'S4': 4}

    BACKENDS = {'cpu': 'CPU',
                'cuda': 'Nvidia - CUDA',
                'opencl': 'AMD - OpenCL'}
    
    def __init__(self, signal_choice, backend, m=200, m_var=10, f_max=None, f_min=0., coherent=False):
        self.signal_choice = signal_choice
        self.signal_choice_ID = self.SIGNALS[signal_choice]
        self.backend = backend
        self.m = m
        self.m_var = m_var
        self.f_max = f_max
        self.f_min = f_min
        self.coherent = coherent


class PlotConfig:
    '''
    Configuration class for plot settings. This class stores settings related to
    the visualization of data points, including scaling options and smoothing parameters.

    Attributes
    ----------
    data_points : int
        The number of data points to be plotted. The higher data_points the higher the resolution
    green_alpha : float
        Opacity of the green error.
    gray_alpha : float
        Opacity of the S2 error bar.
    arcsinh_scale : bool, optional
        Indicates whether arcsinh scaling should be applied to the data points to
        improve visibility. Default is False.
    arcsinh_const : float, optional
        The constant factor used in arcsinh scaling. Only relevant if arcsinh_scale
        is True. Default is 0.02.
    sigma : float, optional
        The standard deviation for Gaussian smoothing of the data. Default is 1.
    '''

    def __init__(self, data_points, green_alpha=0.4, gray_alpha=0.5, arcsinh_scale=False, arcsinh_const=0.02,
                 sigma=1):
        
        self.data_points = data_points
        self.green_alpha = green_alpha
        self.gray_alpha = gray_alpha
        self.arcsinh_scale = arcsinh_scale
        self.arcsinh_const = arcsinh_const
        self.sigma = sigma


class StreamSetup:
    def __init__(self, handle, channel_config, data_acq_config):
        self.handle = handle
        self.channel_config = channel_config
        self.data_acq_config = data_acq_config
        self.auto_stop = 0
        self.status = {}
    
    def open_unit(self):
        self.status['open_unit'] = ps4000a.ps4000aOpenUnit(ctypes.byref(self.handle), None)
        
        try:
            assert_pico_ok(self.status['open_unit'])
        
        except:
            power_status = self.status

            if power_status == 286:
                self.status = ps4000a.ps4000aChangePowerSource(self.handle, power_status)

            else:
                raise

            assert_pico_ok(self.status['open_unit'])
    
    def set_channel(self):
        for channel, config in self.channel_config.items():
            if config.activity == ChannelConfig.ACTIVITY['active']:
                self.status[f'set_channel_{channel}'] = ps4000a.ps4000aSetChannel(
                    self.handle,
                    config.channel,
                    config.activity,
                    config.coupling,
                    config.range_map,
                    config.v_offset)
                
                assert_pico_ok(self.status[f'set_channel_{channel}'])
    
    def set_buffer(self):
        for channel, config in self.channel_config.items():
            if config.activity == ChannelConfig.ACTIVITY['active']:
                self.status[f'set_buffer_{channel}'] = ps4000a.ps4000aSetDataBuffers(
                    self.handle,
                    config.channel,
                    self.data_acq_config.buffer_max[channel].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                    None,
                    self.data_acq_config.buffer_size,
                    self.data_acq_config.segment_idx,
                    0)
                
                assert_pico_ok(self.status[f'set_buffer_{channel}'])
    
    def set_stream(self):
        self.status['run_stream'] = ps4000a.ps4000aRunStreaming(
            self.handle, 
            ctypes.byref(self.data_acq_config.sampling_interval),
            self.data_acq_config.time_unit,
            0, # not pre-triggering
            self.data_acq_config.buffer_size,
            self.auto_stop,
            self.data_acq_config.downsampling_ratio,
            ps4000a.PS4000A_RATIO_MODE['PS4000A_RATIO_MODE_NONE'],
            self.data_acq_config.buffer_size)
        
        self.sampling_interval = self.data_acq_config.sampling_interval.value

        assert_pico_ok(self.status['run_stream'])



class Callback:
    def __init__(self, channel_config, data_acq_config):
        self.channel_config = channel_config
        self.data_acq_config = data_acq_config

        self.chunk_queue = {channel: queue.Queue() for channel, config in self.channel_config.items()
                             if config.activity == ChannelConfig.ACTIVITY['active']}
        self.chunk = {channel: np.zeros(self.data_acq_config.buffer_size, dtype=np.float32)
                      for channel, config in self.channel_config.items()
                      if config.activity == ChannelConfig.ACTIVITY['active']}

        self.next_sample = 0
        self.sample_counter = 0
        self.round = 1
        self.auto_stop_outer = False
        self.was_called_back = False

        # Convert the python method into a C function pointer.
        self.cFuncPtr = ps4000a.StreamingReadyType(self.streaming_callback_wrapper)

    def streaming_callback_wrapper(self, handle, no_of_samples, start_idx, overflow, triggerAt, triggered, auto_stop, param):
        self.streaming_callback(handle, no_of_samples, start_idx, overflow, triggerAt, triggered, auto_stop, param)

    def streaming_callback(self, handle, no_of_samples, start_idx, overflow, triggerAt, triggered, auto_stop, param):
        self.was_called_back = True
        print(f"chunk: {self.next_sample + no_of_samples}")
        source_end = start_idx + no_of_samples

        for channel, config in self.channel_config.items():
            if config.activity == ChannelConfig.ACTIVITY['active']:
                self.chunk[channel][self.next_sample:self.next_sample + no_of_samples] = self.data_acq_config.buffer_max[channel][start_idx:source_end]

        self.next_sample += no_of_samples
        self.sample_counter += no_of_samples

        if self.next_sample == self.data_acq_config.buffer_size:
            self.round += 1
            for channel, config in self.channel_config.items():
                if config.activity == ChannelConfig.ACTIVITY['active']:
                    self.chunk_queue[channel].put(self.chunk[channel].copy())
            
            self.next_sample = 0

        if self.data_acq_config.capture_limit is not None and self.sample_counter >= self.data_acq_config.capture_limit:
            self.auto_stop_outer = True


class StreamPlotter:
    def __init__(self, handle, channel_configs, data_acq_config, signal_config, plot_config, stream_setup, callback):
        self.handle = handle
        self.channel_configs = channel_configs
        self.data_acq_config = data_acq_config
        self.signal_config = signal_config
        self.plot_config = plot_config
        self.stream_setup = stream_setup
        self.callback = callback
        self.status = {}
        self.should_close = False
        self.colorbar_rt = None
        self.colorbar_avg = None

        if self.signal_config.f_min == 0:
            self.keys_to_process = [1, 2, 3, 4]
        else:
            self.keys_to_process = [1, 2, 4]

        self.colors = {'A': 'blue', 'B': 'green', 'C': 'red', 'D': 'black'}

        plt.ion() # interactive mode

        
    def plot(self):
        s_avg = {}
        serr_avg = {}
        fig, (ax_rt, ax_avg) = plt.subplots(1, 2, figsize=(20, 10))  # ax_rt for real-time, ax_avg for average
        
        print("Press q to quit")
        
        print(f"Plot updates {self.plot_config.data_points} points every {self.data_acq_config.buffer_size * self.stream_setup.sampling_interval} {self.data_acq_config.time_unit_str}")

        def on_key(event):
            if event.key == 'q':  # close when 'q' is pressed
                self.should_close = True
                plt.pause(0.01)
                

        fig.canvas.mpl_connect('key_press_event', on_key)

        while not self.callback.auto_stop_outer:

            self.callback.was_called_back = False
            self.status["getStreamingLastestValues"] = ps4000a.ps4000aGetStreamingLatestValues(self.handle, self.callback.cFuncPtr, None)
            
            if not self.callback.was_called_back:
                sleep(0.01)
            
            max_adc = ctypes.c_int16()
            self.status['max_value_adc'] = ps4000a.ps4000aMaximumValue(self.handle, ctypes.byref(max_adc))
            assert_pico_ok(self.status['max_value_adc'])
            
            for channel_name, channel_config in self.channel_configs.items():
                if channel_config.activity and not self.callback.chunk_queue[channel_name].empty():

                    chunk = self.callback.chunk_queue[channel_name].get()
                    converted_chunk = adc_converter(chunk, channel_config.voltage_range_value, max_adc)
                    config_ss = snp.SpectrumConfig(data=converted_chunk,
                                                delta_t=float(self.stream_setup.sampling_interval),
                                                f_unit=self.data_acq_config.freq_unit,
                                                spectrum_size=self.plot_config.data_points,
                                                order_in='all',
                                                f_max=self.signal_config.f_max,
                                                f_min=self.signal_config.f_min,
                                                backend=self.signal_config.backend,
                                                show_first_frame=False,
                                                m=self.signal_config.m,
                                                m_var=self.signal_config.m_var,
                                                coherent=self.signal_config.coherent)
                    
                    spec = snp.SpectrumCalculator(config_ss)
                    f, s, serr = spec.calc_spec()
                    
                    if self.signal_config.signal_choice == 'S2':
                        # Update the accumulated spectrum
                        for key in self.keys_to_process:
                            if key not in s_avg:
                                # Initialize the value in the dictionary if the key doesn't exist
                                s_avg[key] = np.real(s[key])
                                serr_avg[key] = np.real(serr[key])
                            else:
                                # Update the value if the key already exists
                                s_avg[key] += np.real(s[key])
                                serr_avg[key] = np.sqrt(serr_avg[key]**2 + np.real(serr[key])**2)

                        # Compute the average spectrum and update the dictionary
                        s_avg[key] /= 2
                        serr_avg[key] /= 2

                        print(self.callback.round)

                        if self.data_acq_config.save:

                            spec.S = s_avg
                            spec.freq = f
                            spec.S_err = serr_avg

                            save_path = self.data_acq_config.save_path
                            if save_path is None:
                                save_path = os.path.join(os.path.expanduser('~'), 'Desktop')
                            filename = f'data.pkl'
                            full_path = os.path.join(save_path, filename)

                            spec.save_spec(full_path)
                        else:
                            pass

                        ax_rt.clear()

                        if self.plot_config.arcsinh_scale:
                            data =  arcsinh_scale(np.real(s[2]), self.plot_config.arcsinh_const)
                            upper_bound_1 = arcsinh_scale(np.real(s[2]) + np.real(serr[2])*self.plot_config.sigma, self.plot_config.arcsinh_const)
                            lower_bound_1 = arcsinh_scale(np.real(s[2]) - np.real(serr[2])*self.plot_config.sigma, self.plot_config.arcsinh_const)
                        else:
                            data =  np.real(s[2])
                            upper_bound_1 = np.real(s[2]) + np.real(serr[2])*self.plot_config.sigma
                            lower_bound_1 = np.real(s[2]) - np.real(serr[2])*self.plot_config.sigma

                        
                        ax_rt.plot(f[2], data)
                        ax_rt.fill_between(f[2], lower_bound_1, upper_bound_1, color='gray', alpha=self.plot_config.gray_alpha, label='Error')
                        ax_rt.set_title('Real-Time Spectrum')
                        ax_rt.set_xlabel(r'$\omega / 2\pi$' + f' [{self.data_acq_config.freq_unit}]')
                        ax_rt.set_ylabel(r'$S^{(2)}_z$' + f' [{self.data_acq_config.freq_unit}' + r'$^{-1}$' + ']')

                        ax_avg.clear()

                        if self.plot_config.arcsinh_scale:
                            data_avg = arcsinh_scale(s_avg[2], self.plot_config.arcsinh_const)
                            upper_bound_2 = arcsinh_scale(s_avg[2] + serr_avg[2]*self.plot_config.sigma, self.plot_config.arcsinh_const)
                            lower_bound_2 = arcsinh_scale(s_avg[2] - serr_avg[2]*self.plot_config.sigma, self.plot_config.arcsinh_const)
                        else:
                            data_avg = s_avg[2]
                            upper_bound_2 = s_avg[2] + serr_avg[2]*self.plot_config.sigma
                            lower_bound_2 = s_avg[2] - serr_avg[2]*self.plot_config.sigma

                        ax_avg.plot(f[2], data_avg)
                        # Fill the area between the upper and lower bounds to represent the error
                        ax_avg.fill_between(f[2], lower_bound_2, upper_bound_2, color='gray', alpha=self.plot_config.gray_alpha, label='Error')
                        ax_avg.set_title('Averaged Spectrum')
                        ax_avg.set_xlabel(r'$\omega / 2\pi$' + f' [{self.data_acq_config.freq_unit}]')
                        plt.draw()
                        plt.tight_layout()
                        plt.pause(0.1)

                    elif self.signal_config.signal_choice != "S2":
                    
                        # Update the accumulated spectrum
                        for key in self.keys_to_process:
                            if key not in s_avg:
                                # Initialize the value in the dictionary if the key doesn't exist
                                s_avg[key] = np.real(s[key])
                                serr_avg[key] = np.real(serr[key])
                            else:
                                # Update the value if the key already exists
                                s_avg[key] += np.real(s[key])
                                serr_avg[key] = np.sqrt(serr_avg[key]**2 + np.real(serr[key])**2)

                        # Compute the average spectrum and update the dictionary
                        s_avg[key] /= 2
                        serr_avg[key] /= 2

                        print(self.callback.round)

                        if self.data_acq_config.save:

                            spec.S = s_avg
                            spec.freq = f
                            spec.S_err = serr_avg

                            save_path = self.data_acq_config.save_path
                            if save_path is None:
                                save_path = os.path.join(os.path.expanduser('~'), 'Desktop')
                            filename = f'data.pkl'
                            full_path = os.path.join(save_path, filename)

                            spec.save_spec(full_path)
                        else:
                            pass

                        if self.plot_config.arcsinh_scale:
                            ser_sigma = arcsinh_scale(np.real(serr[self.signal_config.signal_choice_ID])*self.plot_config.sigma, self.plot_config.arcsinh_const)
                            data =  arcsinh_scale(np.real(s[self.signal_config.signal_choice_ID]), self.plot_config.arcsinh_const)
                            err_matrix = np.zeros_like(data)
                            err_matrix[data < ser_sigma] = 1

                            ser_avg_sigma = arcsinh_scale(serr_avg[self.signal_config.signal_choice_ID]*self.plot_config.sigma, self.plot_config.arcsinh_const)
                            data_avg =  arcsinh_scale(s_avg[self.signal_config.signal_choice_ID], self.plot_config.arcsinh_const)
                            err_avg_matrix = np.zeros_like(data_avg)
                            err_avg_matrix[data_avg < ser_avg_sigma] = 1
                        else:
                            ser_sigma = np.real(serr[self.signal_config.signal_choice_ID])*self.plot_config.sigma
                            data =  np.real(s[self.signal_config.signal_choice_ID])
                            err_matrix = np.zeros_like(data)
                            err_matrix[data < ser_sigma] = 1

                            ser_avg_sigma = serr_avg[self.signal_config.signal_choice_ID]*self.plot_config.sigma
                            data_avg = s_avg[self.signal_config.signal_choice_ID]
                            err_avg_matrix = np.zeros_like(data_avg)
                            err_avg_matrix[data_avg < ser_avg_sigma] = 1

                        new_rt_min_value = np.min(data)
                        new_rt_max_value = np.max(data)
                        abs_rt_max = np.maximum(abs(new_rt_max_value), abs(new_rt_min_value))
                        norm_rt = colors.TwoSlopeNorm(vmin=-abs_rt_max, vcenter=0, vmax=abs_rt_max)

                        new_avg_min_value = np.min(data_avg)
                        new_avg_max_value = np.max(data_avg)
                        abs_avg_max = np.maximum(abs(new_avg_max_value), abs(new_avg_min_value))
                        norm_avg = colors.TwoSlopeNorm(vmin=-abs_avg_max, vcenter=0, vmax=abs_avg_max)

                        ax_rt.clear()
                        color_array = np.array([[0., 0., 0., 0.], [0., 0.5, 0., self.plot_config.green_alpha]])
                        cmap_sigma = colors.LinearSegmentedColormap.from_list(name='green_alpha', colors=color_array)

                        contour_rt = ax_rt.pcolormesh(np.real(f[self.signal_config.signal_choice_ID]),
                                                        np.real(f[self.signal_config.signal_choice_ID]),
                                                        err_matrix,
                                                        vmin=0, vmax=1,
                                                        cmap=cmap_sigma,
                                                        zorder=2)
                        
                        contour_rt = ax_rt.pcolormesh(np.real(f[self.signal_config.signal_choice_ID]),
                                                    np.real(f[self.signal_config.signal_choice_ID]),
                                                    data,
                                                    cmap='seismic',
                                                    norm=norm_rt)
                        
                        p = 1 -  self.signal_config.signal_choice_ID # correct exponential
                        
                        ax_rt.set_title('Real-Time Spectrum\n'
                                        fr'$S^{{{self.signal_config.signal_choice_ID}}}_z$' + f' [{self.data_acq_config.freq_unit}' + fr'$^{{{p}}}$' + ']')
                        
                        ax_rt.set_xlabel(r'$\omega_1 / 2\pi$' + f' [{self.data_acq_config.freq_unit}]')
                        ax_rt.set_ylabel(r'$\omega_2 / 2\pi$' + f' [{self.data_acq_config.freq_unit}]')
                        
                        if self.colorbar_rt is None:
                            self.colorbar_rt = plt.colorbar(contour_rt, ax=ax_rt)
                        else:
                            self.colorbar_rt.update_normal(contour_rt)

                        # Update the average spectrum plot
                        ax_avg.clear()
                        color_array = np.array([[0., 0., 0., 0.], [0., 0.5, 0., self.plot_config.green_alpha]])
                        cmap_sigma = colors.LinearSegmentedColormap.from_list(name='green_alpha', colors=color_array)

                        contour_avg = ax_avg.pcolormesh(np.real(f[self.signal_config.signal_choice_ID]),
                                                        np.real(f[self.signal_config.signal_choice_ID]),
                                                        err_avg_matrix,
                                                        vmin=0, vmax=1,
                                                        cmap=cmap_sigma,
                                                        zorder=2)
    
                        contour_avg = ax_avg.pcolormesh(np.real(f[self.signal_config.signal_choice_ID]),
                                                        np.real(f[self.signal_config.signal_choice_ID]),
                                                        data_avg,
                                                        cmap='seismic',
                                                        norm=norm_avg,
                                                        zorder=1)
                        
                        
                        ax_avg.set_title('Average Spectrum\n'
                                         fr'$S^{{{self.signal_config.signal_choice_ID}}}_z$' + f' [{self.data_acq_config.freq_unit}' + fr'$^{{{p}}}$' + ']')
                        
                        ax_avg.set_xlabel(r'$\omega_1 / 2\pi$' + f' [{self.data_acq_config.freq_unit}]')
                        
                        if self.colorbar_avg is None:
                            self.colorbar_avg = plt.colorbar(contour_avg, ax=ax_avg)
                        else:
                            self.colorbar_avg.update_normal(contour_avg)
                        
                        plt.draw()
                        plt.tight_layout()
                        plt.pause(0.1)
            

            if self.should_close: 
                break

        plt.close() 
        
    def stop(self):
        # End of interactive mode
        plt.ioff()
        plt.show()


class ClosingStream:
    def __init__(self, handle):
        self.handle = handle
        self.status = {}
        
    # stopping the picoscope
    def stop(self):
        self.status['stop'] = ps4000a.ps4000aStop(self.handle)
        assert_pico_ok(self.status['stop'])
    
    # closing the unit
    def close(self):
        self.status['close'] = ps4000a.ps4000aCloseUnit(self.handle)
        assert_pico_ok(self.status['close'])