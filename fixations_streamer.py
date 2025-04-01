

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


"""
Requires an LSL stream "ccs-neon-001_Neon Gaze" (produced directly from the Neon Companion app)
Produces:
    1. Binary LSL stream "Fixations" that indicates whether the eye is currently in a saccade or not (determined by idt)
    2. A real-time plot of the gaze angles, to allow for following the idt decisions
"""


#fixation_detection_idt and _calculate_gaze_angles are taken directly from the PhysioLabXR GitHub: 
#https://github.com/PhysioLabXR/PhysioLabXR-Community/blob/ffb75e1b43625806689bb96cbd90f3645948e1b7/physiolabxr/scripting/physio/eyetracking.py#L77
def _calculate_gaze_angles(gaze_vector):
    """
    gaze vectors should be 3D vectors in the eye coordinate system, with the z axis pointing out of the eye straight ahead
    @param gaze_vector:
    @param head_rotation_xy_degree:
    @return:
    """
    reference_vector = np.array([0, 0, 1])
    dot_products = np.dot(gaze_vector.T, reference_vector)
    magnitudes = np.linalg.norm(gaze_vector, axis=0)
    reference_magnitude = np.linalg.norm(reference_vector)
    cosine_angles = dot_products / (magnitudes * reference_magnitude)
    angles_rad = np.arccos(cosine_angles)
    angles_deg = np.degrees(angles_rad)

    return angles_deg


def fixation_detection_idt(gaze_xyz, timestamps, window_size=0.175, dispersion_threshold_degree=0.5, saccade_min_sample=2, return_last_window_start=False):
    """

    @param gaze_xyz:
    @param timestamps:
    @param window_size:
    @param dispersion_threshold_degree:
    @param saccade_min_sample: the minimal number of samples between consecutive fixations to be considered as a saccade
    @return:
    """

    try:

        assert window_size > 0, "fixation_detection_idt: window size must be positive"
        gaze_angles_degree = _calculate_gaze_angles(gaze_xyz)
        windows = [(i, np.argmin(np.abs(timestamps - (t + window_size)))) for i, t in enumerate(timestamps)]


        last_window_start = 0
        fixations = []
        for start, end in windows:
            if end >= len(timestamps):
                break
            if end - start < saccade_min_sample:
                continue
            center_time = timestamps[start] + window_size / 2
            if np.std(gaze_angles_degree[start:end]) < dispersion_threshold_degree:
                fixations.append([1, center_time])  # 1 for fixation
            else:
                fixations.append([0, center_time])
            last_window_start = start
        if return_last_window_start:
            return np.array(fixations).T, last_window_start
        else:
            return np.array(fixations).T
    except Exception as e:
        print(e)
        return None


# Constantly supplies the stream object with a recent chunk of samples (from the "ccs-neon-001_Neon Gaze" stream)
def lsl_pull_thread(stream):
    while True:
        stream.pull_chunk()
        time.sleep(0.001)


# Helper class for the real-time plot
# The LSL streams of Pupil devices use "Pupil Time" for their timestamps. Since we use Unix timestamps for everything else, we adjust using an offset
# This is the methodology described in https://docs.pupil-labs.com/core/developer/#convert-pupil-time-to-system-time
# The buffer also automatically only keeps the most recent 2000 samples to plot
class DataBuffer:
    def __init__(self):
        self.gaze_angles_buffer = []
        self.gaze_angles_timestamps = []
        self.gaze_angles_time_offset = 0
        self.fixations_buffer = []
        self.fixations_timestamps = []
        self.fixations_time_offset = 0

    def update_gaze_data(self, timestamps, gaze_angles):
        if self.gaze_angles_time_offset == 0:
            self.gaze_angles_time_offset = time.time() - timestamps[0]
        self.gaze_angles_timestamps += timestamps
        self.gaze_angles_buffer += gaze_angles
        self.trim_buffers()

    def update_fixation_data(self, timestamps, fixations):
        #resync for when replaying from .xdf file
        #causes timestamps to get out of order
        if self.fixations_time_offset == 0 or (timestamps[0] + self.fixations_time_offset - time.time()) < -0.5:
            self.fixations_time_offset = time.time() - timestamps[0]
        self.fixations_timestamps += timestamps
        self.fixations_buffer += [20 if x == 1 else 0 for x in fixations]
        self.trim_buffers()

    def trim_buffers(self):
        if len(self.gaze_angles_timestamps) > 2000:
            self.gaze_angles_timestamps = self.gaze_angles_timestamps[-2000:]
            self.gaze_angles_buffer = self.gaze_angles_buffer[-2000:]
        if len(self.fixations_timestamps) > 2000:
            self.fixations_timestamps = self.fixations_timestamps[-2000:]
            self.fixations_buffer = self.fixations_buffer[-2000:]


# Using matplotlibs "FuncAnimation" to produce a real-time plot.
# The construct using queues is needed because this needs to run in a subprocess in order to be non-blocking (thread does not work)
def run_plotting(queue):
    def update_plot(frame):
        while not queue.empty():
            data_buffer = queue.get()
            ax.clear()
            ax.plot(data_buffer.gaze_angles_timestamps, data_buffer.gaze_angles_buffer, label='Data Line')
            ax.plot(data_buffer.fixations_timestamps, data_buffer.fixations_buffer, label='Fixation')
            ax.legend(loc='upper left')
            plt.xlabel('X values')
            plt.ylabel('Y values')
            plt.title('Real-time plot of X and Y values')

    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, update_plot, interval=20)
    plt.show()




class FixationsStreamer:
    def __init__(self, draw_plot=False):
        self.draw_plot = draw_plot
        self.connected = False
        self.latest_timestamp = 0
        # These two are just for plotting
        self.queue = multiprocessing.Queue()
        self.data_buffer = DataBuffer()

    # Again, for allowing communication with the subprocess
    def _handle_queue_updates(self):
        while True:
            if self.data_buffer:
                self.queue.put(self.data_buffer)
                time.sleep(0.05)

    def initialize(self):
        self.lsl_stream = LSLStream("ccs-neon-001_Neon Gaze", track_history_seconds=3.0)
        self.outlet = StreamOutlet(StreamInfo('Fixations', 'Markers', 2, 20, 'float32', 'fixation_outlet'))
        self.connected = self.lsl_stream.connected

    # We need one thread to fill the buffer constantly using the LSL stream from the Neon Companion app (pull_thread)
    # The other thread (process_thread) will process the data in the buffer and push the fixations to the outlet
    def start(self):
        #pull thread
        threading.Thread(target=lsl_pull_thread, args=(self.lsl_stream,), daemon=True).start()
        #push thread
        threading.Thread(target=self.process_thread, args=(self.lsl_stream, self.outlet, self.data_buffer), daemon=True).start()


        if self.draw_plot:
            self.plot_process = multiprocessing.Process(target=run_plotting, args=(self.queue,))
            self.plot_process.start()
            self.queue_thread = threading.Thread(target=self._handle_queue_updates)
            self.queue_thread.daemon = True
            self.queue_thread.start()


    def process_thread(self, lsl_stream, outlet, data_buffer):
        process_interval = 1.0 / 20  # 20 times per second, example on physio website was 30
        while True:
            time.sleep(process_interval)
            buffer = lsl_stream.history
            if buffer:
                #Closely follows the usage of idt from https://physiolabxrdocs.readthedocs.io/en/latest/FixationDetection.html
                timestamps = np.array([item[0] for item in buffer])
                gaze_xyz = np.array([item[1] for item in buffer])
                gaze_angles = list(_calculate_gaze_angles(gaze_xyz.T))

                # For plotting gaze angles
                data_buffer.update_gaze_data(list(timestamps), gaze_angles)

                # Parameters have been tuned from the defaults. Can still be improved
                fixations, last_window_start = fixation_detection_idt(gaze_xyz.T, timestamps, 0.175, 1.2, 2, True)
                
                # Has just happened once for unknown reason
                if(fixations is None or len(fixations) < 2):
                    print("Warning: Fixations is None or less than 2")
                    print(buffer)
                    print(gaze_xyz.T)
                    print(fixations)
                    continue
                    
                lsl_stream.clear_history_to_index(last_window_start)
                data_buffer.update_fixation_data(list(fixations[1]), fixations[0])

                timestamps = [timestamp + data_buffer.fixations_time_offset for timestamp in timestamps]
                self.latest_timestamp = timestamps[-1]

                combined_samples = [[fx, ga] for fx, ga in zip(fixations[0].tolist(), gaze_angles[1:-1])]
                outlet.push_chunk(combined_samples, timestamps)