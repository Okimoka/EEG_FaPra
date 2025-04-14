

import threading
import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet
from helpers import _calculate_gaze_angles, fixation_detection_idt, DataBuffer
from streamer import Streamer
from plotter import Plotter

"""
Requires the LSL stream "ccs-neon-001_Neon Gaze" (produced directly from the Neon Companion app)
Produces:
    1. LSL stream "Fixations" that indicates whether the eye is currently in a saccade or not (determined by idt)
    2. (Optional) A real-time plot of the gaze angles, to allow for understanding and tuning idt

The "Fixations" stream has two channels: [fixation, gaze_angle], where fixation is a binary number (1 == not inside saccade, 0 == inside saccade)
gaze angle is used for plotting
"""



class FixationsStreamer(Streamer, Plotter):
    def __init__(self, draw_plot=False, **kwargs):
        self.data_buffer = DataBuffer()
        Streamer.__init__(self)
        Plotter.__init__(self, draw_plot=draw_plot, data_buffer=self.data_buffer)


    def initialize(self):
        super().initialize("ccs-neon-001_Neon Gaze", None, StreamOutlet(StreamInfo('Fixations', 'Markers', 2, 20, 'float32', 'fixation_outlet')))

    def start(self):
        # Constantly pull from ccs-neon-001_Neon Gaze and write it into data_buffer
        Streamer.start(self, pull_1=True, print_lag_1=True)
        Plotter.start(self)
        # Process the pulled data (compute fixation state) and push it into the outlet
        threading.Thread(target=self.process_thread, args=(self.input_stream_1, self.outlet, self.data_buffer), daemon=True).start()



    def process_thread(self, lsl_stream, outlet, data_buffer, cooldown_time=0.3):
        process_interval = 1.0 / 20  # 20 times per second, example on physio website was 30
        last_saccade_end_time = None
        last_fixation_state = 1
        while True:
            time.sleep(process_interval)
            buffer = lsl_stream.history
            if buffer:
                ###Should not happen unless LSL timestamps jump back in time
                ###if(len(lsl_stream.history) != None and last_saccade_end_time != None and len(lsl_stream.history)>0 and lsl_stream.history[-1][0] < last_saccade_end_time):
                ###    print("Detected loop in fixation data, resetting Fixations variables")
                ###    last_saccade_end_time = None
                ###    last_fixation_state = 1
                ###    #continue

                #Closely follows the usage of idt from https://physiolabxrdocs.readthedocs.io/en/latest/FixationDetection.html
                timestamps = np.array([item[0] for item in buffer])
                gaze_xyz = np.array([item[1] for item in buffer])
                gaze_angles = list(_calculate_gaze_angles(gaze_xyz.T))

                # For plotting gaze angles
                data_buffer.update_gaze_data(list(timestamps), gaze_angles)

                # Parameters have been tuned from the defaults, but can still be improved
                fixations, last_window_start = fixation_detection_idt(gaze_xyz.T, timestamps, 0.175, 1.2, 2, True)
                lsl_stream.clear_history_to_index(last_window_start)
				
                if(len(fixations) == 0):
                    continue
				
                # Sometimes saccades are registered multiple times in quick succession, even though it should be just one. E.g. when blinking
                # To mitigate this, any new saccades are blocked from starting until a specified cooldown is over
                for i, (timestamp, state) in enumerate(zip(timestamps, fixations[0])):
                    if(state == 0): #start of a saccade
                        if last_saccade_end_time is not None and (timestamp < last_saccade_end_time + cooldown_time): #if within cooldown
                            fixations[0][i] = 1 #overwrite with 1 (fixation)
                
                    if(state == 1 and last_fixation_state == 0): # Exiting saccade
                        last_saccade_end_time = timestamp
                
                    last_fixation_state = state


                # fixations[1] == timestamps, fixations[0] == binary array of data
                data_buffer.update_fixation_data(list(fixations[1]), fixations[0])

                timestamps = [timestamp + data_buffer.fixations_time_offset for timestamp in timestamps]
                self.latest_timestamp = timestamps[-1]

                combined_samples = [[fx, ga] for fx, ga in zip(fixations[0].tolist(), gaze_angles[1:-1])]
                outlet.push_chunk(combined_samples, timestamps)
