

import threading
import numpy as np
import time
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_stream, resolve_streams


#TODO
"""
Like in the example https://physiolabxrdocs.readthedocs.io/en/latest/FixationDetection.html
Clear buffer until last_window_start instead of full clear
"""

#This and all relating functions are taken directly from the PhysioLabXR GitHub: 
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


#Slighly modified to also return timestamps
class LSLStream:
    def __init__(self, name, track_history_seconds=0):

        self.streams = None
        self.inlet = None
        self.sampling_rate = None
        self.max_history_length = None
        self.history = []
        self.track_history_seconds = track_history_seconds

        all_streams = resolve_streams()
        if(name in [stream.name() for stream in all_streams]):
            self.streams = resolve_stream('name', name)
            self.inlet = StreamInlet(self.streams[0])
            self.sampling_rate = self.inlet.info().nominal_srate()
            self.max_history_length = int(self.sampling_rate * self.track_history_seconds)
            print("Connected to stream: " + name)
        else:
            print("Stream " + name + " not found.")
        
    
    def pull_sample(self):
        try:
            sample, timestamp = self.inlet.pull_sample(timeout=0.5)
            if sample:
                if self.track_history_seconds > 0:
                    self.history.append((timestamp, sample))
                    # Remove samples that are older than the specified track_history_seconds
                    while self.history and (timestamp - self.history[0][0]) > self.track_history_seconds:
                        self.history.pop(0)
                return timestamp, sample
            else:
                #print("Stream is not sending data!") todo
                return (0,0)
        except Exception as e:
            #print(e) Todo
            return (0,0)



def lsl_pull_thread(stream, buffer, buffer_lock):
    while True:
        timestamp, sample = stream.pull_sample()
        if sample != (0, 0):
            gaze_data = sample[6:9]  #channels 3, 4, and 5
            #print("Gaze Data:", gaze_data)
            with buffer_lock:
                buffer.append((timestamp, gaze_data))
        time.sleep(0.001)

def process_thread(buffer, buffer_lock, outlet):
    process_interval = 1.0 / 20  # 20 times per second, example was 30
    while True:
        time.sleep(process_interval)
        with buffer_lock:
            if buffer:
                timestamps = np.array([item[0] for item in buffer])
                gaze_xyz = np.array([item[1] for item in buffer])
                fixations, last_window_start = fixation_detection_idt(gaze_xyz.T, timestamps)
                #print(fixations.T)

                for i, fixation in enumerate(fixations.T):
                    #print("Pushing " + str(fixation) + " at " + str(timestamps[i]))
                    #outlet.push_sample(fixation, timestamps[i])
                    outlet.push_sample([fixation], timestamps[i])
                buffer.clear()

if __name__ == "__main__":
    stream_name = "ccs-neon-001_Neon Gaze"
    lsl_stream = LSLStream(stream_name, track_history_seconds=1.0)
    
    info = StreamInfo('Fixations', 'Markers', 1, 20, 'float32', 'fixation_outlet')


    outlet = StreamOutlet(info)


    buffer = []
    buffer_lock = threading.Lock()

    # LSL thread
    pull_thread = threading.Thread(target=lsl_pull_thread, args=(lsl_stream, buffer, buffer_lock))
    pull_thread.daemon = True
    pull_thread.start()

    # IDT processing thread
    process_thread = threading.Thread(target=process_thread, args=(buffer, buffer_lock, outlet))
    process_thread.daemon = True
    process_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")