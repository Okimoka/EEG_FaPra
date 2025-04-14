import threading
import time
from lsl_stream import LSLStream

"""
Parent class for SaccadeStreamer, FixationsStreamer, SurfaceBlinksStreamer, UnfoldAnalyzer

They all have the basic functionality of constantly pulling data from some LSL outlet(s), processing it, then streaming it to some outlet
"""

class Streamer:

    def __init__(self, sampling_rate=20, output_rate=20, **kwargs):
        self.connected = False
        self.latest_timestamp = 0
        self.sampling_rate = sampling_rate
        self.output_rate = output_rate

    # Constantly supplies the stream object with a recent chunk of samples
    # All LSL pulls need to be in separate threads, or else we get drifting timestamps
    def lsl_pull_thread(self, stream, print_lag=False):
        while True:

            timestamps, samples = stream.pull_chunk()
            # check if the timestamps are aligned to the current time. If not, it will be signaled in the console
            # If there are no prints of this kind, it is assumed all input streams are synchronized to real time
            for ts in timestamps:
                if(print_lag and int(time.time() - ts) > 0):
                    if(stream.name != "ccs-neon-001_Neon Gaze"):  #this can only lag in debug mode
                        print(f"[{stream.name.ljust(15)}] Lag:", int(time.time() - ts), " n samples: " + str(len(samples)))

            time.sleep(1/self.sampling_rate)

    def initialize(self, input_stream_1, input_stream_2, outlet):
        # All instances of Streamer use either one or two input LSL streams
        self.input_stream_1 = LSLStream(input_stream_1, track_history_seconds=3.0) # history has to be adjusted if unfold.jl window is longer than 3 seconds
        #If input_stream_2 is None, it will default to a dummy stream with connected=True
        self.input_stream_2 = LSLStream(input_stream_2, track_history_seconds=3.0)

        self.connected = self.input_stream_1.connected and self.input_stream_2.connected

        if(self.connected):
            self.outlet = outlet
        else:
            self.outlet = None


    def start(self, pull_1=False, pull_2=False, print_lag_1=False, print_lag_2=False, **kwargs):
        if not self.connected:
            print("Streamer is not connected to all required LSL streams.")
            return
            
        if pull_1:
            threading.Thread(target=self.lsl_pull_thread, args=(self.input_stream_1, print_lag_1), daemon=True).start()
        if pull_2:
            threading.Thread(target=self.lsl_pull_thread, args=(self.input_stream_2, print_lag_2), daemon=True).start()
