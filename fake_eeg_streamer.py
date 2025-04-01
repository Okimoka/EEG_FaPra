    

import threading
import time
from pylsl import StreamInfo, StreamOutlet, local_clock
from lsl_stream import LSLStream


"""
Requires LSL stream UnicornEEG_Filtered
Produces LSL stream FakeEEG, which is simply the restreamed data

This class is needed when testing with an .xdf file, since the timestamps are from the time of recording
When re-streaming, they are "corrected" to be the current time

"""

class FakeEEGStreamer:
    def __init__(self):
        self.connected = False
        self.latest_timestamp = 0

    def lsl_pull_thread(self, stream):
        while True:
            timestamp, sample = stream.pull_sample()

            self.outlet.push_sample(sample, time.time())
            self.latest_timestamp = time.time()
            time.sleep(1/250)
        
    def initialize(self):
        self.eeg_stream = LSLStream("UnicornEEG_Filtered", track_history_seconds=3.0)
        self.outlet = StreamOutlet(StreamInfo('FakeEEG', 'Markers', 8, 250, 'float32', 'fakeeeg_outlet'))
        self.connected = self.eeg_stream.connected

    def start(self):
        self._start_lsl_thread()

    def _start_lsl_thread(self):
        self.pull_thread = threading.Thread(target=self.lsl_pull_thread, args=(self.eeg_stream,))
        self.pull_thread.daemon = True
        self.pull_thread.start()

