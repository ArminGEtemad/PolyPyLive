# PolyPyLive
PolyPyLive is designed to seamlessly integrating SignalSnap with streaming devices such as TimeTagger, thus enabling real-time data acquisition and processing. Moreover PolyPyLive's architecture is crafted with compatibility in mind, allowing for the direct analysis of collected data and saved data in SignalSnap without the need for any modifications. This ensures a smooth workflow transition from real-time processing to in-depth analysis, providing a comprehensive toolset for measurement.

## Key Advantages

- **Real-Time Analysis**: PolyPyLive is unique in its ability to calculate and display higher-order spectra as data is being acquired.
- **Advanced Estimation Techniques**: The library employs SignalSnap which utilizes an unbiased cumulant-based estimator for general signals and a moment-based estimator for coherent signals, ensuring accurate spectral computation.
- **Enhanced Visibility**: Users can opt for an arcsinh scale for signal display, improving the interpretability of results in real-time.

## Testing and Readiness

PolyPyLive has been tested in a controlled environment and is now ready for deployment in laboratory settings. As we strive to refine and enhance its functionality, we actively seek feedback from experimentalists. Your insights are invaluable to us for driving further improvements and ensuring that PolyPyLive meets the real-world demands of signal measurement research and development. We encourage users to share their experiences and suggestions. This revision emphasizes the readiness of the library for laboratory use while also making a clear call to action for feedback from the community, which is crucial for the iterative development process.

## Caution and Future Development

This initial alpha version of PolyPyLive supports single-channel data acquisition. We are actively working to extend its capabilities to include multi-channel support in upcoming releases. Please be aware that the software saves data with the filename `data.pkl` by default; users have the flexibility to choose the storage directory path. 

## Example

```Python
# This software is right now proided as a script and in the next updates
# will be upgraded to a python package. Right now, it is enough to clone
# this repository and and change the parameters of to your need after
# the line:
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
This is always needed since it communicates with the hardware and creates TimeTagger object.
If there is no signal generator, `setTestSignal` is making sure that you are still able to test the program by adding:
```Python
tagger_controller = _TaggerController()
tagger_controller.enable_test_signal([2])
```
to the program. The number `2` here is the number of channel in use.

Now it is time for configuration of you data acquisition:
```Python
data_acq_config = _DataAcqConfig(buffer_size=10000000,
                                 time_unit='us',
                                 duration=int(30e7),
                                 delay=0,
                                 time_stamp_measure='s',
                                 save=True,
                                 save_path='your path')
```
Here you can set your buffer (higher buffer makes sure that your data are not being overwritten in TimeTagger).
Then you set the time unit for the duration of your measurement and delay as well as the duration. You have to
also choose the time stamps measure. Saving path is your choice. In case of `None`, it will be save on your desktop.


Next part of the program allows you to configurate your Channel.
```Python
channel_config = _ChannelConfig(channel=2,
                                trigger_level=1.5,
                                dead_time=0,
                                falling=False)
```

In the next part, you can confing your **real-time visualization** spectrum. Meaning, even if you choose 'S2' only
power spectrum is being shown. In the background, however, all the 'S1', 'S2', 'S3' and 'S4' as well as their 
average are being calculated. The backend can be chosen according to signalsnap. You have to choose a maximum for
the frequency in $Hz$. In this example `f_max=2000` means $2000Hz$. If you choose a coherent signal the estimators
will be moment based. In case of `coherent=False` the estimators will be cumulant based.
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
The `data_points` are the resolution of your spectra. The `arcsing_scale`, if set to `True` makes sure that you can
see the small fluctuations too according the the `arcsinh_const`. The lower this constant is the more clear the small
fluctuations are. Tha value of `sigma` is the significance. These Configurations will NOT be saved in your saved data.

The ending part of the program is there to let the configurations communicate with each other and close the TimeTagger
after the measurement.
```Python
stream_setup = _StreamSetup(tagger_controller.tagger, channel_config, data_acq_config, signal_config, plot_config)
stream_setup.start()
stream_setup.process_data()

tagger_controller.free()
```
