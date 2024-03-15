import PicoScopeStreamer
from ctypes import *

class Runner:
    def __init__(self, channel, data_acq, signal, plot):
        self.handle = c_int16()
        self.channel = channel
        self.data_acq = data_acq
        self.signal = signal
        self.plot = plot
        self.stream = None
        self.callback = None
        self.plotter = None
        
    def setup(self):
        self.stream = PicoScopeStreamer.StreamSetup(self.handle, self.channel, self.data_acq)
        self.stream.open_unit()
        self.stream.set_channel()
        self.stream.set_buffer()
        self.stream.set_stream()
        self.callback = PicoScopeStreamer.Callback(self.channel, self.data_acq)

        self.plotter = PicoScopeStreamer.StreamPlotter(self.handle, self.channel, self.data_acq, self.signal, self.plot, self.stream, self.callback)
    
    def run(self):
        self.plotter.plot()
        self.plotter.stop()

    
    def close(self):
        self.plotter.should_close = True
        self.closer = PicoScopeStreamer.ClosingStream(self.handle)
        self.closer.stop()
        self.closer.close()