import numpy as np
import logging
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pylsl import StreamInfo, StreamOutlet
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

QUANTILE = 0.05
BUFFERSIZE = 500 # in seconds
LOWCUT = 2 # in Hz
HIGHCUT = 70 # in Hz
NOTCHFREQ = 50 # in Hz

# Activate logging for debugging
logging.basicConfig(level=logging.DEBUG)

# (Lowcut, Highcut, List of relevant channels)
band_definitions = {
    "Delta": (0.5, 4, [0, 2]),
    "Theta": (4, 8, [0, 2]),
    "Alpha": (8, 12, [4,5,6,7]),
    "Beta": (12, 30, [2])    
}

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
            raise ValueError("Länge darf nicht größer als die Größe des Buffers sein.")
        
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


# Hochpass-Filterinitialisierung mit Zustand
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

# Funktion zum Anwenden der kaskadierten Notch-Filter
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




# Main program for initializing and starting the threads
if __name__ == "__main__":
    params = BrainFlowInputParams()
    board_id = BoardIds.UNICORN_BOARD.value
    connected = False

    try:
        # Connect to unicorn device
        board = BoardShim(board_id, params)
        board.prepare_session()
        print("Successfully connected to Unicorn device.")
        connected = True
    except Exception as e:
        print(f"Error when connecting to the Unicorn device: {e}")

    if connected:
        try:
            board.start_stream()
            print("EEG stream successfully launched.")

            # Initializing the streams
            timestamp_channel_index = BoardShim.get_timestamp_channel(board_id)

            num_eeg_channels = 8
            sampling_rate = BoardShim.get_sampling_rate(board_id)
            info = StreamInfo('UnicornEEG_Filtered', 'EEG', num_eeg_channels, sampling_rate, 'float32', 'unicorn_filtered')
            outlet = StreamOutlet(info)

            info_power = StreamInfo('UnicornEEG_Power', 'EEG_Power', len(band_definitions), sampling_rate, 'float32', 'unicorn_power')
            outlet_power = StreamOutlet(info_power)


            # Initialization of the filters
            # Bandpass for general noise
            b_band, a_band, zi_band = create_bandpass_filterblock(LOWCUT, HIGHCUT, sampling_rate, num_channels=num_eeg_channels)
            # Notchfilter for line noise
            cascaded_notch_filters = create_cascaded_notch_filters(NOTCHFREQ, 5, sampling_rate, num_channels=num_eeg_channels)

            # Initialize Ringbuffers and corresponding filter for each power band
            b_bands = []
            a_bands = []
            zi_bands = []
            ringbuffers = []
            ringbuffers_long = []
            # Iterate over each powerband
            for band, (lowcut, highcut, list_of_channels) in band_definitions.items():
                ringbuffers_long.append(RingBuffer(sampling_rate*BUFFERSIZE))
                # Iterate over each channel important to the powerband
                for i in range(len(list_of_channels)):
                    b, a, zi = create_bandpass_filter(lowcut, highcut, sampling_rate)
                    b_bands.append(b)
                    a_bands.append(a)
                    zi_bands.append(zi)
                    ringbuffers.append(RingBuffer(sampling_rate))


            while True:
                data = board.get_board_data()
                if data.shape[1] > 0:
                    number_of_datapoints = data.shape[1]
                    
                    timestamps = data[timestamp_channel_index, :]

                    most_recent_timestamp = timestamps[-1]

                    # Basic filtering and notch filtering
                    eeg_data = data[:num_eeg_channels, :]
                    eeg_data_band_filtered , zi_band = apply_filter_block(eeg_data, b_band, a_band, zi_band)
                    eeg_data_final, cascaded_notch_filters = apply_cascaded_notch_filters(eeg_data_band_filtered, cascaded_notch_filters)

                    # Transpose the data to (samples x channels) for LSL
                    eeg_data_final_LSL = eeg_data_final.T

                    # Push the filtered data as a chunk to the outlet
                    outlet.push_chunk(eeg_data_final_LSL.tolist(), timestamp=most_recent_timestamp)

                    # Compute band powers (uses variances of bands as heuristic)
                    j=0
                    variances = []

                    # Iterate over each powerband
                    for l, (band, (lowcut, highcut, list_of_channels)) in enumerate(band_definitions.items()):
                        variances_band = []
                        # Iterate over each channel important to the powerband
                        for channel in list_of_channels:
                            b_band_loop = b_bands[j] 
                            a_band_loop = a_bands[j]
                            zi_band_loop = zi_bands[j]

                            # Bandpass filter to needed powerband
                            eeg_data_band_loop , zi_band_loop = apply_filter(eeg_data[channel], b_band_loop, a_band_loop, zi_band_loop)
                            zi_bands[j] = zi_band_loop

                            # Save bandpass filtered signals
                            ringbuffers[j].append(eeg_data_band_loop)

                            variances_channel = []
                            # Iterate over each new timestamp
                            for k in range(-number_of_datapoints+1, 1):
                                # Compute variances for each new timestamp (on interval of half a second)
                                variances_channel.append(np.var(ringbuffers[j].get(k,int(sampling_rate/2))))
                            variances_band.append(variances_channel)

                            j+=1

                        # Compute the mean of variances for each timestep over the interesting channels
                        final_variances_band = np.mean(variances_band, axis=0)
                        ringbuffers_long[l].append(final_variances_band)

                        # Normalize the variances by low and high quantiles (value of low equals 0 and value of high equals 1, under-/overshoot is possible)
                        variances_history_band = ringbuffers_long[l].get(0, sampling_rate*BUFFERSIZE)
                        variances_history_band_cleaned = variances_history_band[variances_history_band != 0]
                        low_quantile = np.quantile(variances_history_band_cleaned, QUANTILE)
                        high_quantile = np.quantile(variances_history_band_cleaned, 1-QUANTILE)

                        denom = high_quantile - low_quantile
                        if denom == 0:
                            denom = 1
                        variances.append((final_variances_band - low_quantile)/denom)
                    # Send variances via LSL
                    variances = np.array(variances)
                    outlet_power.push_chunk(variances.T.tolist(), timestamp=most_recent_timestamp)

            print("Streaming filtered EEG data... Press CTRL+C to end the stream.")


        except KeyboardInterrupt:
            print("Streaming terminated by user.")
        finally:
            board.stop_stream()
            board.release_session()
            print("Connection closed.")
