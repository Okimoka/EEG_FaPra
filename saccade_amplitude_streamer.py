    

import threading
import numpy as np
import time
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop, resolve_streams, local_clock
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lsl_stream import LSLStream
import datetime
from types import SimpleNamespace

from streamer import Streamer


"""
Requires Fixations LSL stream.
Saccade amplitude can either be calculated by the euclidean distance of the gaze on the surface before and after the saccade
Or by difference of eye angles as calculated in idt
For the first option, we additionally require the SurfaceGaze_0 stream

Produces Saccades LSL stream, which is just the amplitude of the most recent saccade
"""


class SaccadeStreamer(Streamer):
    def __init__(self, callback, amplitude_method="surface"):
        super().__init__()
        self.callback = callback
        self.amplitude_method = amplitude_method
        
    def initialize(self):

        surface_stream_name = "SurfaceGaze_0" if self.amplitude_method == "surface" else "Fixations"
        outlet = StreamOutlet(StreamInfo('Saccades', 'Markers', 1, 20, 'float32', 'saccade_outlet'))
        super().initialize("Fixations", surface_stream_name, outlet)


    def start(self):
        super().start(pull_1=True, pull_2=True, print_lag_1=True, print_lag_2=True)
        threading.Thread(target=self.saccade_amplitude_thread, args=(self.outlet,), daemon=True).start()


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

            if(len(self.input_stream_1.history)>0 and self.input_stream_1.history[-1][0] < last_processed_fixation_timestamp): #happens in sample data when it loops
                print("Detected loop in fixation data, resetting Saccade variables")
                self.input_stream_1.history.clear()
                self.input_stream_2.history.clear()
                last_fixation_state = None
                last_position = None
                saccade_start_position = None
                saccade_start_timestamp = None
                last_amplitude = 0.0
                last_processed_fixation_timestamp = 0
                #continue

            # Get all new fixations since the last one we processed
            new_fixations = [sample for sample in self.input_stream_1.history if sample[0] > last_processed_fixation_timestamp]
            number_zeros = [sample[1][0] for sample in new_fixations].count(0.0)

            for fixation_sample in new_fixations:
                fixation_timestamp, fixation_data = fixation_sample

                current_fixation_state = fixation_data[0]

                # Find the closest surface gaze point to this fixation timestamp
                surface_sample = min(self.input_stream_2.history, key=lambda s: abs(s[0] - fixation_timestamp), default=None)
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
                            surface_stream_history = self.input_stream_2.history
                            
                            #trim surface_stream_history to the saccade window
                            surface_stream_history = [sample for sample in surface_stream_history if sample[0] > saccade_window[0] and sample[0] < saccade_window[1]]

                            timestamps, samples = zip(*self.input_stream_2.history)
                            angles = [sample[1] for sample in samples]
    
                            last_amplitude = max(angles) - min(angles)
                        
                        self.callback(last_amplitude)


                self.latest_timestamp = time.time()

                #For debugging
                #print(f"time.time():     {time.time()}")
                #print(f"local_clock():   {local_clock()}")
                #print(f"diff:            {time.time() - local_clock()}")
                #print("-----------------------------------------    ")

                # YouQuantified needs constant supply of samples, so we always push the last known amplitude, even if it hasn't updated.
                outlet.push_sample([last_amplitude], time.time())
                #print("Pushed sample:", last_amplitude, "at", time.time())

                # Update last known position and fixation state
                last_fixation_state = current_fixation_state
                last_position = current_position
                last_processed_fixation_timestamp = fixation_timestamp
