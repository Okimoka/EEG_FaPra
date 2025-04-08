    

import threading
import time
from pylsl import StreamInfo, StreamOutlet, local_clock
from lsl_stream import LSLStream
from streamer import Streamer

"""
Requires LSL stream UnicornEEG_Filtered
Produces LSL stream FakeEEG, which is simply the restreamed data

This class is needed when testing with an .xdf file, since the timestamps are from the time of recording
When re-streaming, they are "corrected" to be the current time

"""

class FakeEEGStreamer(Streamer):
    def __init__(self):
        super().__init__()


    def lsl_pull_thread(self, stream, _): #dont need print_lag 
        while True:
            timestamps, samples = stream.pull_chunk()
            if(samples[0] == [0]):
                continue

            for ts, sample in zip(timestamps, samples):
                self.outlet.push_sample(sample, time.time())
                self.latest_timestamp = time.time()

            time.sleep(1/self.sampling_rate)

    def initialize(self):
        super().initialize("UnicornEEG_Filtered", None, StreamOutlet(StreamInfo('FakeEEG', 'EEG', 8, 250, 'float32', 'fakeeeg_outlet')))

    def start(self):
        super().start(pull_1=True)
