import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import TimeTagger
import signalsnap as snp
import os


def _event_listener(event, stream_plotter_instance):
    """
    Helper function listens if 'q' is pressed to close the data capturing.
    Closing the evaluation occurs only after the captured data in a run are evaluated.
    """

    if event.key == 'q':
        stream_plotter_instance.should_close = True
        plt.close(event.canvas.figure)
        plt.pause(0.01)


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


def _unit_converter(time_unit):

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

    return freq_unit

def _picosec_converter(t_input, time_unit):

    if time_unit == 's':
        t_input *= 1e12

    elif time_unit == 'ms':
        t_input *= 1e9

    elif time_unit == 'us':
        t_input *= 1e6

    elif time_unit == 'ns':
        t_input *= 1e3

    elif time_unit == 'ps':
        t_input *= 1

    return t_input

def _time_stamp_unit_convert(t_input, measure):

    if measure == 's':
        t_input /= 1e12

    elif measure == 'ms':
        t_input /= 1e9

    elif measure == 'us':
        t_input /= 1e6

    elif measure == 'ns':
        t_input /= 1e3

    elif measure == 'ps':
        t_input /= 1

    return t_input



class _TaggerController:
    '''
    Controller for a TimeTagger device. This class provides an interface for
    creating a TimeTagger, enabling the test signal, and freeing the TimeTagger.

    Attributes
    ----------
    tagger : TimeTagger
        The TimeTagger device controlled by this controller.

    Methods
    -------
    enable_test_signal(channels):
        Enables the test signal on the specified channels.
    free():
        Frees the TimeTagger device.
    '''
    def __init__(self):
        self.tagger = TimeTagger.createTimeTagger()

    def enable_test_signal(self, channels):
        self.tagger.setTestSignal(channels, True)

    def free(self):
        TimeTagger.freeTimeTagger(self.tagger)


class _ChannelConfig:
    '''
    Configuration class for individual channels in a TimeTagger device. This class
    stores settings for a single channel, including trigger level, dead time, and
    other optional parameters.

    Attributes
    ----------
    channel : int
        The identifier for the channel. Negative values indicate falling edge
        detection if 'falling' is set to True during initialization.
    trigger_level : float
        The voltage level that triggers an event on this channel.
    dead_time : float
        The minimum time between two consecutive events on this channel to be
        considered separate events.
    divider : int, optional
        The division factor for the event rate on this channel. Default is 1,
        meaning no division.
    falling : bool, optional
        Indicates whether the trigger is set for falling edges. Default is False,
        meaning it is set for rising edges.

    '''

    def __init__(self, channel, trigger_level, dead_time, divider=1, falling=False):
        if falling:
            self.channel = -channel
        else:
            self.channel = channel

        self.trigger_level = trigger_level 
        self.dead_time = dead_time 
        self.divider = divider
    

class _DataAcqConfig:

    '''
    Configuration class for data acquisition settings in a measurement device. This class
    stores settings related to the acquisition buffer, timing, and optional data saving.

    Attributes
    ----------
    buffer_size : int
        The size of the buffer used for storing data during acquisition.
    duration : int
        The duration for which data is acquired, converted to picoseconds based on the time unit provided.
    delay : int
        The delay before starting data acquisition, converted to picoseconds based on the time unit provided.
    freq_unit : str
        The unit of frequency derived from the time unit provided.
    time_stamp_measure : bool
        Indicates whether time stamps should be measured.
    save : bool, optional
        Indicates whether the acquired data should be saved. Default is False.
    save_path : str or None, optional
        The file path where the data should be saved if saving is enabled. Default is None.
    '''

    def __init__(self, buffer_size, time_unit, duration, delay, time_stamp_measure, save=False, save_path=None):

        self.buffer_size = buffer_size
        self.duration = _picosec_converter(duration, time_unit)
        self.delay = _picosec_converter(delay, time_unit)
        self.freq_unit = _unit_converter(time_unit)
        self.time_stamp_measure = time_stamp_measure
        self.save = save
        if self.save:
            self.save_path = save_path



class _SignalConfig:
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
        The minimum frequency for the signal filter. Default is 0.0.
    m : int, optional
        Number of windows for the estimation of the cumulant
    m_var : int, optional
        m_var single spectra are used at a time for the calculation of the spectral errors
    coherent : bool, optional
        Indicates whether coherent signal processing is enabled. Default is False.
    '''

    SIGNALS = {'S2': 2,
               'S3': 3,
               'S4': 4}

    BACKENDS = {'cpu': 'CPU',
                'cuda': 'Nvidia - CUDA',
                'opencl': 'AMD - OpenCL'}
    
    def __init__(self, signal_choice, backend, f_max, f_min=0., m=5, m_var=3, coherent=False):
        self.signal_choice = signal_choice
        self.signal_choice_ID = self.SIGNALS[signal_choice]
        self.backend = backend
        self.f_max = f_max
        self.f_min = f_min
        self.m = m
        self.m_var = m_var
        self.coherent = coherent


class _PlotConfig:
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



class _StreamSetup:
    '''
    Setup class for configuring and starting a data stream with a TimeTagger device.
    This class encapsulates the configuration for the channel, data acquisition,
    signal processing, and plotting, and provides methods to apply these configurations
    and start the data stream.

    Attributes
    ----------
    tagger : TimeTagger
        The TimeTagger device to be configured and used for the data stream.
    channel_config : _ChannelConfig
        The configuration settings for the channel.
    data_acq_config : _DataAcqConfig
        The configuration settings for data acquisition.
    signal_config : _SignalConfig
        The configuration settings for signal processing.
    plot_config : _PlotConfig
        The configuration settings for plotting the data.
    stream : TimeTagStream or None
        The TimeTagStream object used for streaming data. None until the stream is started.
    colorbar_rt : UnknownType or None
        Real-time colorbar configuration. The type and use are not specified in the provided code snippet.
    colorbar_avg : UnknownType or None
        Average colorbar configuration. The type and use are not specified in the provided code snippet.
    keys_to_process : list of int
        The keys (channel numbers) that will be processed during the stream.

    '''
    def __init__(self, tagger, channel_config, data_acq_config, signal_config, plot_config):
        '''
        Initializes a new instance of the _StreamSetup class with the specified
        configurations for the TimeTagger device, channel, data acquisition, signal
        processing, and plotting.

        Parameters
        ----------
        tagger : TimeTagger
            An instance of the TimeTagger device to be used for the stream.
        channel_config : _ChannelConfig
            An instance of the _ChannelConfig class with the desired channel settings.
        data_acq_config : _DataAcqConfig
            An instance of the _DataAcqConfig class with the desired data acquisition settings.
        signal_config : _SignalConfig
            An instance of the _SignalConfig class with the desired signal processing settings.
        plot_config : _PlotConfig
            An instance of the _PlotConfig class with the desired plot settings.
        '''

        self.tagger = tagger
        self.channel_config = channel_config
        self.data_acq_config = data_acq_config
        self.signal_config = signal_config
        self.plot_config = plot_config
        self.stream = None
        self.colorbar_rt = None
        self.colorbar_avg = None

        # key_to_process are the spectra to be calculated. If the user chooses anything other than 0. for
        # f_min S3 will be excluded from the calculations. 

        if self.signal_config.f_min == 0.:
            self.keys_to_process = [1, 2, 3, 4]
        else:
            self.keys_to_process = [1, 2, 4]
    
    def apply_config(self):
        # These are working right now for only one channel.
        self.tagger.setTriggerLevel(self.channel_config.channel, self.channel_config.trigger_level)
        #self.tagger.setDelay(self.channel_config.delay)
        self.tagger.setDeadtime(self.channel_config.channel, self.channel_config.dead_time)
        # TODO setting the divider and the Delay

    def start(self):
        '''
        Applies the configuration settings and starts the data stream. This method
        initializes the TimeTagStream with the specified buffer size and channels,
        and starts the stream for the duration specified in the data acquisition
        configuration.
        '''
        self.apply_config()
        self.stream = TimeTagger.TimeTagStream(tagger=self.tagger,
                                               n_max_events=self.data_acq_config.buffer_size,
                                               channels=[self.channel_config.channel])
        self.stream.startFor(self.data_acq_config.duration)
    
    def process_data(self):
        chunk_counter = 0
        s_avg = {}
        serr_avg = {}
        plt.ion()
        
        fig, (ax_rt, ax_avg) = plt.subplots(2, 1, figsize=(20, 10))  # ax_rt for real-time, ax_avg for average
        # Register the event listener
        fig.canvas.mpl_connect('key_press_event', lambda event: _event_listener(event, self))

        # Add a new attribute to the instance to control the streaming process
        self.should_close = False

        while self.stream.isRunning() and not self.should_close:
            data = self.stream.getData()
            chunk_counter += 1

            if data.size == self.data_acq_config.buffer_size:
                print('TimeTagStream buffer is filled completely. Events arriving after the buffer has been filled have been discarded. Please increase the buffer size not to miss any events.')

            if data.size > 0:
                channel = data.getChannels()
                timestamps = data.getTimestamps()
                time_stamps_seconds = np.array([_time_stamp_unit_convert(t, self.data_acq_config.time_stamp_measure) for t in timestamps])
                print('time stamps : ', timestamps.size)

                config_ss = snp.SpectrumConfig(data=time_stamps_seconds,
                                            f_unit=self.data_acq_config.freq_unit,
                                            f_max=self.signal_config.f_max,
                                            f_min=self.signal_config.f_min,
                                            spectrum_size=self.plot_config.data_points,
                                            order_in='all',
                                            backend=self.signal_config.backend,
                                            show_first_frame=False,
                                            m=self.signal_config.m,
                                            m_var=self.signal_config.m_var,
                                            coherent=self.signal_config.coherent)
                
                RSspec = snp.SpectrumCalculator(config_ss)
                

                f, s, serr = RSspec.calc_spec_poisson(n_reps=1)
                    
                if self.signal_config.signal_choice == "S2":
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
                    #s_avg[key] /= chunk_counter
                    #serr_avg[key] /= chunk_counter

                    
                    if self.data_acq_config.save:

                        RSspec.S = {key: value / chunk_counter for key, value in s_avg.items()}
                        RSspec.freq = f
                        RSspec.S_err = {key: value / chunk_counter for key, value in serr_avg.items()}

                        save_path = self.data_acq_config.save_path
                        if save_path is None:
                            save_path = os.path.join(os.path.expanduser('~'), 'Desktop')
                        filename = f'data.pkl'
                        full_path = os.path.join(save_path, filename)

                        RSspec.save_spec(full_path)
                    else:
                        pass

                    # Plot the real-time spectrum
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
                    # Fill the area between the upper and lower bounds to represent the error
                    ax_rt.fill_between(f[2], lower_bound_1, upper_bound_1, color='gray', alpha=self.plot_config.gray_alpha , label='Error')

                    ax_rt.set_title('Real-Time Spectrum')
                    ax_rt.set_ylabel(r'$S^{(2)}_z$' + f' [Hz' + r'$^{-1}$' + ']')
                    ax_rt.legend()

                    # Plot the average spectrum
                    ax_avg.clear()
                    if self.plot_config.arcsinh_scale:
                        data_avg = arcsinh_scale(s_avg[2]/chunk_counter, self.plot_config.arcsinh_const)
                        upper_bound_2 = arcsinh_scale((s_avg[2] + serr_avg[2]*self.plot_config.sigma)/chunk_counter, self.plot_config.arcsinh_const)
                        lower_bound_2 = arcsinh_scale((s_avg[2] - serr_avg[2]*self.plot_config.sigma)/chunk_counter, self.plot_config.arcsinh_const)
                    else:
                        data_avg = (s_avg[2])/chunk_counter
                        upper_bound_2 = (s_avg[2] + serr_avg[2]*self.plot_config.sigma)/chunk_counter
                        lower_bound_2 = (s_avg[2] - serr_avg[2]*self.plot_config.sigma)/chunk_counter

                    ax_avg.plot(f[2], data_avg)
                    # Fill the area between the upper and lower bounds to represent the error
                    ax_avg.fill_between(f[2], lower_bound_2, upper_bound_2, color='gray', alpha=0.5, label='Error')

                    ax_avg.set_title('Average Spectrum')
                    ax_avg.legend()
                    ax_avg.set_xlabel(r'$\omega / 2\pi$' + f' [Hz]')
                    ax_avg.set_ylabel(r'$S^{(2)}_z$' + f' [Hz' + r'$^{-1}$' + ']')
                    

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
                    #s_avg[key] /= 2
                    #serr_avg[key] /= 2

                    if self.data_acq_config.save:

                        RSspec.S = s_avg/chunk_counter
                        RSspec.freq = f
                        RSspec.S_err = serr_avg/chunk_counter

                        save_path = self.data_acq_config.save_path
                        if save_path is None:
                            save_path = os.path.join(os.path.expanduser('~'), 'Desktop')
                        filename = f'data.pkl'
                        full_path = os.path.join(save_path, filename)

                        RSspec.save_spec(full_path)
                    else:
                        pass

                    if self.plot_config.arcsinh_scale:
                        ser_sigma = arcsinh_scale(np.real(serr[self.signal_config.signal_choice_ID])*self.plot_config.sigma, self.plot_config.arcsinh_const)
                        data =  arcsinh_scale(np.real(s[self.signal_config.signal_choice_ID]), self.plot_config.arcsinh_const)
                        err_matrix = np.zeros_like(data)
                        err_matrix[data < ser_sigma] = 1
                        
                        ser_avg_sigma = arcsinh_scale((serr_avg[self.signal_config.signal_choice_ID]*self.plot_config.sigma)/chunk_counter, self.plot_config.arcsinh_const)
                        data_avg =  arcsinh_scale((s_avg[self.signal_config.signal_choice_ID])/chunk_counter, self.plot_config.arcsinh_const)
                        err_avg_matrix = np.zeros_like(data_avg)
                        err_avg_matrix[data_avg < ser_avg_sigma] = 1
                    else:
                        ser_sigma = np.real(serr[self.signal_config.signal_choice_ID])*self.plot_config.sigma
                        data =  np.real(s[self.signal_config.signal_choice_ID])
                        err_matrix = np.zeros_like(data)
                        err_matrix[data < ser_sigma] = 1
                        
                        ser_avg_sigma = (serr_avg[self.signal_config.signal_choice_ID]*self.plot_config.sigma)/chunk_counter
                        data_avg = (s_avg[self.signal_config.signal_choice_ID])/chunk_counter
                        err_avg_matrix = np.zeros_like(data_avg)
                        err_avg_matrix[data_avg < ser_avg_sigma] = 1

                    new_rt_min_value = np.min(s[self.signal_config.signal_choice_ID])
                    new_rt_max_value = np.max(s[self.signal_config.signal_choice_ID])
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
                                                np.real(s[self.signal_config.signal_choice_ID]),
                                                cmap='seismic',
                                                norm=norm_rt)
                    
                    ax_rt.set_title('Real-Time Spectrum')
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
                    
                    ax_avg.set_title('Average Spectrum')
                    if self.colorbar_avg is None:
                        self.colorbar_avg = plt.colorbar(contour_avg, ax=ax_avg)
                    else:
                        self.colorbar_avg.update_normal(contour_avg)
                    
                    ax_avg.set_xlabel(r'$\omega_1 / 2\pi$' + f' [{self.data_acq_config.freq_unit}]')
                    ax_avg.set_ylabel(r'$\omega_2 / 2\pi$' + f' [{self.data_acq_config.freq_unit}]')
                    

                    fig.canvas.flush_events()

                    plt.draw()
                    plt.tight_layout()
                    plt.pause(0.1)

                    
                    #print(chunk_counter)
                    #chunk_counter += 1
                    if self.should_close:
                        break
                
        plt.ioff()

# ========================== Example usage ==========================

tagger_controller = _TaggerController()

channel_config = _ChannelConfig(channel=2,
                                trigger_level=1.5,
                                dead_time=0,
                                falling=False)

data_acq_config = _DataAcqConfig(buffer_size=10000000,
                                 time_unit='us',
                                 duration=int(30e7),
                                 delay=0,
                                 time_stamp_measure='s',
                                 save=False)

signal_config = _SignalConfig(signal_choice='S2',
                              backend='cuda',
                              m = 10,
                              f_max=4000,
                              coherent=False)

plot_config = _PlotConfig(data_points=200,
                          arcsinh_scale=False,
                          arcsinh_const=0.001,
                          sigma=3)

stream_setup = _StreamSetup(tagger_controller.tagger, channel_config, data_acq_config, signal_config, plot_config)

stream_setup.start()
stream_setup.process_data()  # Process and print the data
tagger_controller.free()
