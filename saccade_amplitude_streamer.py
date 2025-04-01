    

import threading
import numpy as np
import time
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_stream, resolve_streams, local_clock
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lsl_stream import LSLStream
import datetime
from types import SimpleNamespace


"""
Requires Fixations LSL stream.
Saccade amplitude can either be calculated by the euclidean distance of the gaze on the surface before and after the saccade
Or by difference of eye angles as calculated in idt
For the first option, we additionally require the SurfaceGaze_0 stream

Produces Saccades LSL stream, which is just the amplitude of the most recent saccade
"""


class SaccadeStreamer:
    def __init__(self, callback, random_amplitude=False, amplitude_method="surface"):
        self.connected = False
        self.latest_timestamp = 0
        self.callback = callback
        self.random_amplitude = random_amplitude
        self.amplitude_method = amplitude_method
        
    def initialize(self):
        self.fixations_stream = LSLStream("Fixations", track_history_seconds=3.0)
        if(self.random_amplitude):
            self.surface_stream = SimpleNamespace(connected=True,history=[(99,[99,99,99]),(99,[99,99,99]),(99,[99,99,99])])
        else:
            if(self.amplitude_method == "surface"):
                self.surface_stream = LSLStream("SurfaceGaze_0", track_history_seconds=3.0)
            elif(self.amplitude_method == "angle"):
                self.surface_stream = self.fixations_stream
        
        self.outlet = StreamOutlet(StreamInfo('Saccades', 'Markers', 1, 20, 'float32', 'saccade_outlet'))
        self.connected = self.fixations_stream.connected and self.surface_stream.connected
    

    def start(self):
        threading.Thread(target=self.lsl_pull_thread, args=(self.fixations_stream, "Gaze    "), daemon=True).start()
        if(not self.random_amplitude):
            threading.Thread(target=self.lsl_pull_thread, args=(self.surface_stream,   "Surface "), daemon=True).start()

        self._start_process_thread()

    def lsl_pull_thread(self, stream, label):
        while True:
            ts, sample = stream.pull_sample()
            if(int(time.time() - ts) > 0):
                print(f"[{label}] Lag:", int(time.time() - ts))
            time.sleep(0.001)


    def _start_process_thread(self):
        self.process_thread = threading.Thread(target=self.saccade_amplitude_thread, args=(self.outlet,))
        self.process_thread.daemon = True
        self.process_thread.start()


    def saccade_amplitude_thread(self, outlet):
        process_interval = 1.0 / 20

        last_fixation_state = None
        last_position = None
        saccade_start_position = None
        saccade_start_timestamp = None
        last_amplitude = 0.0
        last_processed_fixation_timestamp = 0

        #surface_buffer = [(timestamp, [x,y,blinks]), (timestamp, [x,y,blinks]), ...]
        #fixations_timestamps = [timestamp1, timestamp2, ...]
        #fixations_buffer = [0, 1, 0, 1, ...]

        while True:
            time.sleep(process_interval)

            if(len(self.fixations_stream.history)>0 and self.fixations_stream.history[-1][0] < last_processed_fixation_timestamp): #happens in sample data when it loops
                self.fixations_stream.history.clear()
                if(not self.random_amplitude):
                    self.surface_stream.history.clear() #For sample data, surface stream will not be resupplied, so we should not clear it
                last_fixation_state = None
                last_position = None
                saccade_start_position = None
                saccade_start_timestamp = None
                last_amplitude = 0.0
                last_processed_fixation_timestamp = 0
                #continue

            # Get all new fixations since the last one we processed
            new_fixations = [sample for sample in self.fixations_stream.history if sample[0] > last_processed_fixation_timestamp]

            for fixation_sample in new_fixations:
                fixation_timestamp, fixation_data = fixation_sample

                current_fixation_state = fixation_data[0]

                # Find the closest surface gaze point to this fixation timestamp
                surface_sample = min(self.surface_stream.history, key=lambda s: abs(s[0] - fixation_timestamp), default=None)
                if surface_sample is None:
                    continue  # No surface sample yet

                current_position = surface_sample[1][:2]

                if last_fixation_state is not None:

                    if last_fixation_state == 1 and current_fixation_state == 0:
                        # Saccade starts
                        saccade_start_position = last_position
                        saccade_start_timestamp = fixation_timestamp

                    # Transition from saccade to fixation
                    elif last_fixation_state == 0 and current_fixation_state == 1 and saccade_start_position is not None:
                        # Saccade ends, calculate amplitude
                        if(self.amplitude_method == "surface"):
                            dx = current_position[0] - saccade_start_position[0]
                            dy = current_position[1] - saccade_start_position[1]
                            last_amplitude = np.sqrt(dx ** 2 + dy ** 2)

                        elif(self.amplitude_method == "angle"):
                            # The window is slightly offset to the past, since the sliding window of idt unavoidably causes offset timings
                            saccade_window = [fixation_timestamp - (fixation_timestamp - saccade_start_timestamp)*2, fixation_timestamp]
                            surface_stream_history = self.surface_stream.history
                            
                            #trim surface_stream_history to the saccade window
                            surface_stream_history = [sample for sample in surface_stream_history if sample[0] > saccade_window[0] and sample[0] < saccade_window[1]]

                            timestamps, samples = zip(*self.surface_stream.history)
                            angles = [sample[1] for sample in samples]
    
                            last_amplitude = max(angles) - min(angles)
                        
                        if self.random_amplitude:
                            last_amplitude = np.random.rand() * 10
                        
                        self.callback(last_amplitude)


                self.latest_timestamp = time.time()

                #For debugging
                #print(f"time.time():     {time.time()}")
                #print(f"local_clock():   {local_clock()}")
                #print(f"diff:            {time.time() - local_clock()}")
                #print("-----------------------------------------    ")

                # YouQuantified needs constant supply of samples, so we always push the last known amplitude, even if it hasn't updated.
                outlet.push_sample([last_amplitude], time.time())

                # Update last known position and fixation state
                last_fixation_state = current_fixation_state
                last_position = current_position
                last_processed_fixation_timestamp = fixation_timestamp
