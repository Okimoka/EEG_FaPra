

import numpy as np
import time

from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

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



class RingBuffer:
    def __init__(self, size: int, dtype=np.float32):
        """
        Initializes the ring buffer.
        :param size: Maximum size of the buffer.
        :param dtype: Data type of the stored values.
        """
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)
        self.index = 0
        self.full = False

    def append(self, values):
        """
        Adds one or more values to the buffer.
        :param values: A single value or a numpy array of values.
        """
        values = np.atleast_1d(values)  # Convert values to an array if they are scalar values
        n_values = len(values)

        if n_values >= self.size:
            # Only insert the last `size` values, as older ones are overwritten
            self.buffer[:] = values[-self.size:]
            self.index = 0
            self.full = True
        else:
            end_index = (self.index + n_values) % self.size
            if end_index >= self.index:
                self.buffer[self.index:end_index] = values
            else:
                split_point = self.size - self.index
                self.buffer[self.index:] = values[:split_point]
                self.buffer[:end_index] = values[split_point:]
            self.index = end_index
            if self.index == 0:
                self.full = True

    def get(self, rel_index: int, length: int) -> np.ndarray:
        """
        Returns a subset of the buffer based on a relative index and the length.
        :param rel_index: Relative index (negative means backwards, 0 is the latest value).
        :param length: Length of the returned array.
        :return: A numpy array with the values.
        """
        if length > self.size:
            raise ValueError("Length may not be larger than length of buffer")
        
        end_index = (self.index + rel_index) % self.size
        start_index = (end_index - length) % self.size
        if start_index < 0:
            start_index += self.size
        
        if start_index < end_index:
            return self.buffer[start_index:end_index]
        else:
            return np.concatenate((self.buffer[start_index:], self.buffer[:end_index]))


def create_bandpass_filterblock(lowcut, highcut, fs, order=3, num_channels=8):
    """
    Returns Bandpass filter with status for multiple channels
    :param lowcut: Lowcut in Hz
    :param highcut: Highcut in Hz
    :param fs: Sampling Frequency
    :param order: Order of the used butterworth filter. Defaults to 3
    :param num_channels: Number of channels, on which the filter should be applied. Defaults to 8
    :return: filter parameter and stati
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    zi = [lfilter_zi(b, a) for _ in range(num_channels)]  # One state for each channel
    return b, a, zi

def create_bandpass_filter(lowcut, highcut, fs, order=3):
    """
    Returns Bandpass filter with status for a single channel
    :param lowcut: Lowcut in Hz
    :param highcut: Highcut in Hz
    :param fs: Sampling Frequency
    :param order: Order of the used butterworth filter. Defaults to 3
    :return: filter parameter and status
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    zi = lfilter_zi(b, a)
    return b, a, zi


# Highpass-Filter initialization with state
def create_highpass_filter(cutoff, fs, order=4, num_channels=8):
    """
    Returns Highpass filter with status for multiple channels
    :param cutoff: Lowcut in Hz
    :param fs: Sampling Frequency
    :param order: Order of the used butterworth filter. Defaults to 4
    :param num_channels: Number of channels, on which the filter should be applied. Defaults to 8
    :return: filter parameter and stati
    """
    nyquist = 0.5 * fs  
    normalized_cutoff = cutoff / nyquist  
    b, a = butter(order, normalized_cutoff, btype='high') 
    zi = [lfilter_zi(b, a) for _ in range(num_channels)] 
    return b, a, zi

# Notch-Filter initialization with state
def create_notch_filter(freq, fs, quality_factor=10, num_channels=8):
    """
    Returns Highpass filter with status for multiple channels
    :param freq: Notch in Hz
    :param fs: Sampling Frequency
    :param quality_factor: quality factor of the used iirnotch filter. Defaults to 10
    :param num_channels: Number of channels, on which the filter should be applied. Defaults to 8
    :return: filter parameter and stati
    """
    nyquist = 0.5 * fs  # Nyquist-Frequenz
    normalized_notch = freq / nyquist  # Normierte Grenzfrequenz
    b, a = iirnotch(normalized_notch, quality_factor, fs)
    zi = [lfilter_zi(b, a) for _ in range(num_channels)]  # Zustand für jeden Kanal
    return b, a, zi


# Notch-Filter initialization with state for multiple frequencies
def create_cascaded_notch_filters(freq, iterations, fs, quality_factor=30, num_channels=8):
    """
    Returns cascaded notch filter with status for multiple channels (on freq and freq*2)
    :param freq: Notch in Hz
    :param iterations: number of iterations of the cascading
    :param fs: Sampling Frequency
    :param quality_factor: quality factor of the used iirnotch filter. Defaults to 30
    :param num_channels: Number of channels, on which the filter should be applied. Defaults to 8
    :return: filter parameter and stati
    """
    filters = []
    for i in range(iterations):
        b, a = iirnotch(freq, quality_factor, fs)
        zi = [lfilter_zi(b, a) for _ in range(num_channels)]  # Initialzustand für jeden Kanal
        filters.append((b, a, zi))
        b, a = iirnotch(freq*2, quality_factor, fs)
        zi = [lfilter_zi(b, a) for _ in range(num_channels)]  # Initialzustand für jeden Kanal
        filters.append((b, a, zi))
    return filters

# Function to apply cascaded Notch-Filter
def apply_cascaded_notch_filters(samples, filters):
    """
    Returns the result of the aplication of the cascaded notch filter multiple channels on a given input
    :param samples: input signal
    :param filters: filter parameter and stati
    :return: filtered signal and updated filter parameter and stati
    """
    num_channels = samples.shape[0] 
    y = samples.copy() 
    updated_filters = []

    # Apply all filter iteratively
    for b, a, zi_list in filters:
        new_zi_list = []
        for channel in range(num_channels):
            # Apply filter for each channel
            filtered_channel, zi_channel = lfilter(b, a, y[channel], zi=zi_list[channel])
            y[channel] = filtered_channel
            new_zi_list.append(zi_channel)
        updated_filters.append((b, a, new_zi_list))  # Updated status for this filter

    return y, updated_filters


def apply_filter_block(samples, b, a, zi):
    """
    Returns the result of the aplication of a filter block on a given input
    :param samples: input signal
    :param b: filter parameter
    :param a: filter parameter 
    :param zi: stati (array that should match samples.shape[0])
    :return: filtered signal and updated stati
    """
    num_channels = samples.shape[0]  
    y = np.zeros_like(samples) 
    updated_zi = []

    for channel in range(num_channels):
        # Filtering for each channel separately
        filtered_channel, zi_channel = lfilter(b, a, samples[channel], zi=zi[channel])
        y[channel] = filtered_channel
        updated_zi.append(zi_channel)

    return y, updated_zi 


def apply_filter(samples, b, a, zi):
    """
    Returns the result of the aplication of a filter on a given input
    :param samples: input signal
    :param b: filter parameter
    :param a: filter parameter 
    :param zi: status (single value)
    :return: filtered signal and updated status
    """
    y, zi_update = lfilter(b, a, samples, zi=zi)
    return y, zi_update