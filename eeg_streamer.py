import threading
import numpy as np
import logging
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pylsl import StreamInfo, StreamOutlet
from helpers import RingBuffer, create_bandpass_filter, apply_filter, create_bandpass_filterblock, apply_filter_block, create_cascaded_notch_filters, apply_cascaded_notch_filters

# Logging für Debugging aktivieren
#logging.basicConfig(level=logging.DEBUG)

QUANTILE = 0.05
BUFFERSIZE = 50 # in seconds
LOWCUT = 2 # in Hz
HIGHCUT = 70 # in Hz
NOTCHFREQ = 50 # in Hz

band_definitions = {
    "Delta": (0.5, 4, [0, 2]),
    "Theta": (4, 8, [0, 2]),
    "Alpha": (8, 12, [4,5,6,7]),
    "Beta": (12, 30, [2])    
}


class EEGStreamer:
    def __init__(self):
        """
        Initialize EEGStreamer with required parameters.
        """
        self.params = BrainFlowInputParams()
        self.board_id = BoardIds.UNICORN_BOARD.value
        self.connected = False
        self.board = None
        self.outlet = None
        self.outlet_power = None
        self.cascaded_notch_filters = None
        self.num_eeg_channels = 8
        
        # Initialize Ringbuffers and corresponding filter for each power band
        self.b_bands = []
        self.a_bands = []
        self.zi_bands = []
        self.ringbuffers = []
        self.ringbuffers_long = []
        

    def initialize(self):
        """
        Attempts to connect to the EEG device and set up filters and stream outlets.
        """
        try:
            # Connect to unicorn device
            self.board = BoardShim(self.board_id, self.params)
            self.board.prepare_session()
            print("Successfully connected to Unicorn device.")
            self.connected = True
        except Exception as e:
            print(f"Error when connecting to the Unicorn device: {e}")
            return

        if self.connected:
            self.board.start_stream()
            print("EEG stream successfully launched.")

            sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            self.outlet = StreamOutlet(StreamInfo('UnicornEEG_Filtered', 'EEG', self.num_eeg_channels, sampling_rate, 'float32', 'unicorn_filtered'))
            self.outlet_power = StreamOutlet(StreamInfo('UnicornEEG_Power', 'EEG_Power', len(band_definitions), sampling_rate, 'float32', 'unicorn_power'))

            # Bandpass for general noise
            self.b_band, self.a_band, self.zi_band = create_bandpass_filterblock(LOWCUT, HIGHCUT, sampling_rate, num_channels=self.num_eeg_channels)
            # Notchfilter for line noise
            self.cascaded_notch_filters = create_cascaded_notch_filters(NOTCHFREQ, 5, sampling_rate, num_channels=self.num_eeg_channels)

            # Iterate over each powerband
            for band, (lowcut, highcut, list_of_channels) in band_definitions.items():
                self.ringbuffers_long.append(RingBuffer(sampling_rate*BUFFERSIZE))
                # Iterate over each channel important to the powerband
                for i in range(len(list_of_channels)):
                    b, a, zi = create_bandpass_filter(lowcut, highcut, sampling_rate)
                    self.b_bands.append(b)
                    self.a_bands.append(a)
                    self.zi_bands.append(zi)
                    self.ringbuffers.append(RingBuffer(sampling_rate))


    def process_eeg(self):
        try:
            while True:
                data = self.board.get_board_data()
                sampling_rate = BoardShim.get_sampling_rate(self.board_id)
                
                if data.shape[1] > 0:
                    number_of_datapoints = data.shape[1]
                    # Initializing the streams
                    timestamp_channel_index = BoardShim.get_timestamp_channel(self.board_id)
                    most_recent_timestamp = data[timestamp_channel_index, -1]

                    # Basic filtering and notch filtering
                    eeg_data = data[:self.num_eeg_channels, :]
                    eeg_data_band_filtered, self.zi_band = apply_filter_block(eeg_data, self.b_band, self.a_band, self.zi_band)
                    eeg_data_final, self.cascaded_notch_filters = apply_cascaded_notch_filters(eeg_data_band_filtered, self.cascaded_notch_filters)

                    # Transpose the data to (samples x channels) for LSL and push the filtered data as a chunk to the outlet
                    self.outlet.push_chunk(eeg_data_final.T.tolist(), timestamp=most_recent_timestamp)

                    # Compute band powers (uses variances of bands as heuristic)
                    j = 0
                    variances = []
                    # Iterate over each powerband
                    for l, (band, (lowcut, highcut, list_of_channels)) in enumerate(band_definitions.items()):
                        variances_band = []
                        # Iterate over each channel important to the powerband
                        for channel in list_of_channels:
                            b_band_loop = self.b_bands[j]
                            a_band_loop = self.a_bands[j]
                            zi_band_loop = self.zi_bands[j]

                            # Bandpass filter to needed powerband
                            eeg_data_band_loop , zi_band_loop = apply_filter(eeg_data[channel], b_band_loop, a_band_loop, zi_band_loop)
                            self.zi_bands[j] = zi_band_loop

                            # Save bandpass filtered signals
                            self.ringbuffers[j].append(eeg_data_band_loop)

                            variances_channel = []
                            # Iterate over each new timestamp
                            for k in range(-number_of_datapoints+1, 1):
                                # Compute variances for each new timestamp (on interval of half a second)
                                variances_channel.append(np.var(self.ringbuffers[j].get(k,int(sampling_rate/2))))
                            variances_band.append(variances_channel)

                            j += 1

                        # Compute the mean of variances for each timestep over the interesting channels
                        final_variances_band = np.mean(variances_band, axis=0)
                        self.ringbuffers_long[l].append(final_variances_band)

                        # Normalize the variances by low and high quantiles (value of low equals 0 and value of high equals 1, under-/overshoot is possible)
                        variances_history_band = self.ringbuffers_long[l].get(0, sampling_rate*BUFFERSIZE)
                        variances_history_band_cleaned = variances_history_band[variances_history_band != 0]
                        low_quantile = np.quantile(variances_history_band_cleaned, QUANTILE)
                        high_quantile = np.quantile(variances_history_band_cleaned, 1-QUANTILE)

                        denom = high_quantile - low_quantile
                        if denom == 0:
                            denom = 1
                        variances.append((final_variances_band - low_quantile) / denom)
                    
                    # Send variances via LSL
                    self.outlet_power.push_chunk(np.array(variances).T.tolist(), timestamp=most_recent_timestamp)

        except KeyboardInterrupt:
            print("Streaming terminated by user.")
        finally:
            self.board.stop_stream()
            self.board.release_session()
            print("Connection closed.")


    def start(self):
        """
        Starts the EEG data stream processing.
        """
        if not self.connected:
            print("EEG-Streamer not connected")
            return

        threading.Thread(target=self.process_eeg, daemon=True).start()

