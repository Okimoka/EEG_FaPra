    

import threading
import numpy as np
import time
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop, resolve_streams, local_clock
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
from types import SimpleNamespace

from streamer import Streamer


"""
Requires Fixations LSL stream.
Saccade amplitude can either be calculated by the euclidean distance of the gaze on the surface before and after the saccade
Or by difference of gaze angles as calculated in idt
For the first option, the SurfaceGaze_0 stream is required

Produces Saccades LSL stream, which is just a single channel of the amplitude of the most recent saccade

"""


class SaccadeStreamer(Streamer):
    def __init__(self, callback_object, amplitude_method="surface"):
        super().__init__()
        self.callback_object = callback_object
        self.amplitude_method = amplitude_method
        
    def initialize(self):
        # The Fixations LSL stream contains the gaze angle in its second channel
        surface_stream_name = "SurfaceGaze_0" if self.amplitude_method == "surface" else "Fixations"
        outlet = StreamOutlet(StreamInfo('Saccades', 'Markers', 1, 20, 'float32', 'saccade_outlet'))
        # Constantly fill "input_stream_1.history" with the most recent samples of the Fixations LSL stream
        super().initialize("Fixations", surface_stream_name, outlet)


    def start(self):
        # Pull both from Fixations as well as whatever is in surface_stream_name (SurfaceGaze or Fixations)
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

        while True:
            time.sleep(process_interval)

            ###Should not happen unless LSL timestamps jump back in time
            ###if(len(self.input_stream_1.history)>0 and self.input_stream_1.history[-1][0] < last_processed_fixation_timestamp): #happens in sample data when it loops
            ###    print("Detected loop in fixation data, resetting Saccade variables")
            ###    self.input_stream_1.history.clear()
            ###    self.input_stream_2.history.clear()
            ###    last_fixation_state = None
            ###    last_position = None
            ###    saccade_start_position = None
            ###    saccade_start_timestamp = None
            ###    last_amplitude = 0.0
            ###    last_processed_fixation_timestamp = 0
            ###    #continue

            # Get all new fixations since the last one that was processed
            new_fixations = [sample for sample in self.input_stream_1.history if sample[0] > last_processed_fixation_timestamp] # Example: [1,1,1,0,0,0,1,1,1]

            for fixation_sample in new_fixations:
                fixation_timestamp, fixation_data = fixation_sample

                current_fixation_state = fixation_data[0]

                # Find the closest surface gaze point to this fixation timestamp
                surface_sample = min(self.input_stream_2.history, key=lambda s: abs(s[0] - fixation_timestamp), default=None)
                if surface_sample is None:
                    continue  # No surface sample yet
                    
                current_position = surface_sample[1][:2] #x,y of most recent sample

                if last_fixation_state is not None:
					
                    # Saccade starts
                    if last_fixation_state == 1 and current_fixation_state == 0:
                        saccade_start_position = last_position #only relevant for surface
                        saccade_start_timestamp = fixation_timestamp

                    # Saccade Ends, calculate amplitude
                    elif last_fixation_state == 0 and current_fixation_state == 1 and saccade_start_position is not None:
                        if(self.amplitude_method == "surface"):
                            
                            dx = current_position[0] - saccade_start_position[0]
                            dy = current_position[1] - saccade_start_position[1]
                            last_amplitude = np.sqrt(dx ** 2 + dy ** 2)

                        elif(self.amplitude_method == "angle"):
                            # The window is slightly offset to the past, since the sliding window of idt unavoidably causes offset timings
                            saccade_window = [fixation_timestamp - (fixation_timestamp - saccade_start_timestamp)*2, fixation_timestamp]

                            #trim surface_stream_history to the saccade window
                            surface_stream_history = [sample for sample in self.input_stream_2.history if sample[0] > saccade_window[0] and sample[0] < saccade_window[1]]

                            if(len(surface_stream_history) == 0):
                                print("Warning! angle stream history is empty")
                                continue
							
                            timestamps, samples = zip(*surface_stream_history) #Convert [(ts,sample),(ts,sample),...] to [(ts, ts, ..), (sample, sample, ...)]
                            angles = [sample[1] for sample in samples]
    
                            last_amplitude = max(angles) - min(angles)
                        
                        # callback_object is the UnfoldAnalyzer
                        if(self.callback_object.connected):
                            self.callback_object.add_saccade(last_amplitude)


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
