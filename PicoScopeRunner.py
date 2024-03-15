import PicoScopeStreamer
import PicoScopeIntegrater

A = PicoScopeStreamer.ChannelConfig('A', 'active', '100mV', 'DC', 0)


channel_config = PicoScopeStreamer.ChannelConfig.configuration_dict([A])

data_acq = PicoScopeStreamer.DataAcqConfig(buffer_size=20000000,
                                           buffer_to_capture=None,
                                           sampling_interval=40,
                                           time_unit='ns',
                                           channel_config=channel_config,
                                           save=False)
"""
data_acq = PicoScopeStreamer.DataAcqConfig(buffer_size=100000,
                                           buffer_to_capture=None,
                                           sampling_interval=10,
                                           time_unit='us',
                                           channel_config=channel_config,
                                           save=False,
                                           save_path=None)
"""

signal = PicoScopeStreamer.SignalConfig('S4', 'cuda', m=300, f_max=0.001, f_min=0.00065)

plotter = PicoScopeStreamer.PlotConfig(data_points=300,
                                       green_alpha=0.4, 
                                       gray_alpha=0.4,
                                       arcsinh_scale=False,
                                       arcsinh_const=0.00002,
                                       sigma=6 )


runner = PicoScopeIntegrater.Runner(channel_config, data_acq, signal, plotter)

runner.setup()
runner.run()

runner.close()