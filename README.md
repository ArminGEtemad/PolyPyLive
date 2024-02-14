# PolyPyLive
PolyPyLive is designed to seamlessly integrate SignalSnap with streaming devices such as TimeTagger, thus enabling real-time data acquisition and processing. Moreover, PolyPyLive's architecture is crafted with compatibility in mind, allowing for the direct analysis of collected data and saved data in SignalSnap without the need for any modifications. It ensures a smooth workflow transition from real-time processing to in-depth analysis, providing a comprehensive toolset for measurement.

## Key Advantages

- **Real-Time Analysis**: PolyPyLive is unique in its ability to calculate and display higher-order spectra as data is being acquired.
- **Advanced Estimation Techniques**: The library employs SignalSnap, which utilizes an unbiased cumulant-based estimator for general signals and a moment-based estimator for coherent signals, ensuring accurate spectral computation.
- **Enhanced Visibility**: Users can opt for an arcsinh scale for signal display, improving the interpretability of results in real-time.

## Testing and Readiness

PolyPyLive has been tested in a controlled environment and is now ready for deployment in laboratory settings. As we strive to refine and enhance its functionality, we actively seek feedback from experimentalists. Your insights are invaluable to us for driving further improvements and ensuring that PolyPyLive meets the real-world demands of signal measurement research and development. We encourage users to share their experiences and suggestions. This revision emphasizes the readiness of the library for laboratory use while also making a clear call to action for feedback from the community, which is crucial for the iterative development process.

## Caution and Future Development

This initial alpha version of PolyPyLive supports single-channel data acquisition. We are actively working to extend its capabilities to include multi-channel support in upcoming releases. Please be aware that the software saves data with the filename `data.pkl` by default; users have the flexibility to choose the storage directory path. 

## Example

```Python
# This script can be used by cloning the repository and adjusting
# the parameters based on your needs.
# In the future, it will be developed into a Python package.
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
                                 save=True,
                                 save_path='your path')

signal_config = _SignalConfig(signal_choice='S2',
                              backend='cuda',
                              f_max=2000,
                              coherent=True)

plot_config = _PlotConfig(data_points=200,
                          arcsinh_scale=False,
                          arcsinh_const=0.001,
                          sigma=3)

stream_setup = _StreamSetup(tagger_controller.tagger, channel_config, data_acq_config, signal_config, plot_config)
stream_setup.start()
stream_setup.process_data()

tagger_controller.free()
```
We can look at this code in details in the following:

```Python
tagger_controller = _TaggerController()
```
It is always needed since it communicates with the hardware and creates the TimeTagger object.
If there is no signal generator, `setTestSignal` makes sure that you are still able to test the program by adding:
```Python
tagger_controller = _TaggerController()
tagger_controller.enable_test_signal([2])
```
to the program. The number `2` here is the number of channel in use.

Now it is time for the configuration of your data acquisition:
```Python
data_acq_config = _DataAcqConfig(buffer_size=10000000,
                                 time_unit='us',
                                 duration=int(30e7),
                                 delay=0,
                                 time_stamp_measure='s',
                                 save=True,
                                 save_path='your path')
```
Here ,you can set your buffer (higher buffer insures that your data are not being overwritten in TimeTagger).
Then, you set the time unit for the duration of your measurement and delay as well as the duration. You also  have to
choose the time stamps measure. Saving path is your choice. In case of `None`, it will be saved on your desktop.


Next part of the program allows you to configurate your Channel.
```Python
channel_config = _ChannelConfig(channel=2,
                                trigger_level=1.5,
                                dead_time=0,
                                falling=False)
```

In the next step, you can verify your real-time visualization spectrum. This means that even if you select 'S2', the power spectrum will be displayed. However, in the background, all 'S1', 'S2', 'S3', and 'S4', as well as their averages, are being calculated. You can select the backend according to signalsnap. You need to specify a maximum frequency in Hz. For example, if you set `f_max=2000`, it means 2000Hz. If you choose a coherent signal, the estimators will be moment-based. If you select `coherent=False`, the estimators will be cumulant-based.
```Python
signal_config = _SignalConfig(signal_choice='S2',
                              backend='cuda',
                              f_max=2000,
                              coherent=True)
```

In the end of the configuaraton, you make you plots to your liking:
```Python
plot_config = _PlotConfig(data_points=200,
                          arcsinh_scale=False,
                          arcsinh_const=0.001,
                          sigma=3)
```
The `data_points` parameter determines the resolution of your spectra. If you set `arcsine_scale` to `True`, it will ensure that small fluctuations are visible according to the `arcsinh_const` parameter. The smaller the value of this constant, the clearer the small fluctuations will be. The `sigma` parameter represents the significance value. It's important to note that these configurations will NOT be saved in your saved data.

Additionally, the end part of the program is designed to allow the configurations to communicate with each other and to close the TimeTagger after the measurement is complete.
```Python
stream_setup = _StreamSetup(tagger_controller.tagger, channel_config, data_acq_config, signal_config, plot_config)
stream_setup.start()
stream_setup.process_data()

tagger_controller.free()
```
