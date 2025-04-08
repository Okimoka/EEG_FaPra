

import threading
import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lsl_stream import LSLStream
import datetime
import sys
import multiprocessing
from helpers import _calculate_gaze_angles, fixation_detection_idt, DataBuffer
from streamer import Streamer
from live_visualization import Plotter

"""
Requires an LSL stream "ccs-neon-001_Neon Gaze" (produced directly from the Neon Companion app)
Produces:
    1. Binary LSL stream "Fixations" that indicates whether the eye is currently in a saccade or not (determined by idt)
    2. A real-time plot of the gaze angles, to allow for following the idt decisions
"""



class FixationsStreamer(Streamer, Plotter):
    def __init__(self, draw_plot=False, **kwargs):
        self.data_buffer = DataBuffer()
        #super().__init__(draw_plot=draw_plot, data_buffer=self.data_buffer)
        Streamer.__init__(self)
        Plotter.__init__(self, draw_plot=draw_plot, data_buffer=self.data_buffer)



    def initialize(self):
        super().initialize("ccs-neon-001_Neon Gaze", None, StreamOutlet(StreamInfo('Fixations', 'Markers', 2, 20, 'float32', 'fixation_outlet')))

    # We need one thread to fill the buffer constantly using the LSL stream from the Neon Companion app (pull_thread)
    # The other thread (process_thread) will process the data in the buffer and push the fixations to the outlet
    def start(self):
        #pull thread
        ###super().start(pull_1=True, label_1="Gaze    ")
        #super().start(pull_1=True, print_lag_1=True)
        Streamer.start(self, pull_1=True, print_lag_1=True)
        Plotter.start(self)
        #push thread
        threading.Thread(target=self.process_thread, args=(self.input_stream_1, self.outlet, self.data_buffer), daemon=True).start()



    def process_thread(self, lsl_stream, outlet, data_buffer, cooldown_time=0.3):
        process_interval = 1.0 / 20  # 20 times per second, example on physio website was 30
        last_saccade_end_time = None
        last_fixation_state = 1
        while True:
            time.sleep(process_interval)
            buffer = lsl_stream.history
            if buffer:
                if(len(lsl_stream.history) != None and last_saccade_end_time != None and len(lsl_stream.history)>0 and lsl_stream.history[-1][0] < last_saccade_end_time):
                    print("Detected loop in fixation data, resetting Fixations variables")
                    last_saccade_end_time = None
                    last_fixation_state = 1
                    #continue

                #Closely follows the usage of idt from https://physiolabxrdocs.readthedocs.io/en/latest/FixationDetection.html
                timestamps = np.array([item[0] for item in buffer])
                gaze_xyz = np.array([item[1] for item in buffer])
                gaze_angles = list(_calculate_gaze_angles(gaze_xyz.T))

                # For plotting gaze angles
                data_buffer.update_gaze_data(list(timestamps), gaze_angles)

                # Parameters have been tuned from the defaults. Can still be improved
                fixations, last_window_start = fixation_detection_idt(gaze_xyz.T, timestamps, 0.175, 1.2, 2, True)
                lsl_stream.clear_history_to_index(last_window_start)



                for i, (timestamp, state) in enumerate(zip(timestamps, fixations[0])):
                    if(state == 0):
                        if last_saccade_end_time is not None and (timestamp < last_saccade_end_time + cooldown_time):
                            fixations[0][i] = 1
                
                #for i, (timestamp, state) in enumerate(zip(timestamps, fixations[0])):
                    if(state == 1 and last_fixation_state == 0): # Exiting saccade
                        last_saccade_end_time = timestamp
                
                    last_fixation_state = state



                data_buffer.update_fixation_data(list(fixations[1]), fixations[0])

                timestamps = [timestamp + data_buffer.fixations_time_offset for timestamp in timestamps]
                self.latest_timestamp = timestamps[-1]

                combined_samples = [[fx, ga] for fx, ga in zip(fixations[0].tolist(), gaze_angles[1:-1])]
                outlet.push_chunk(combined_samples, timestamps)
