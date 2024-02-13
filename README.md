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
