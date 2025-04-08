import threading
import numpy as np
import logging
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pylsl import StreamInfo, StreamOutlet
from helpers import RingBuffer, create_bandpass_filter, apply_filter, create_bandpass_filterblock, apply_filter_block, create_cascaded_notch_filters, apply_cascaded_notch_filters

# Logging für Debugging aktivieren
#logging.basicConfig(level=logging.DEBUG)

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
        self.ringbuffers = []
        self.ringbuffers_long = []
        self.b_bands = []
        self.a_bands = []
        self.zi_bands = []
        self.cascaded_notch_filters = None

    def initialize(self):
        """
        Attempts to connect to the EEG device and set up filters and stream outlets.
        """
        try:
            self.board = BoardShim(self.board_id, self.params)
            self.board.prepare_session()
            print("Erfolgreich mit Unicorn-Gerät verbunden.")
            self.connected = True
        except Exception as e:
            print(f"Fehler bei der Verbindung mit dem Unicorn-Gerät: {e}")
            return

        if self.connected:
            self.board.start_stream()
            print("EEG-Stream erfolgreich gestartet.")

            num_eeg_channels = 8
            sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            self.outlet = StreamOutlet(StreamInfo('UnicornEEG_Filtered', 'EEG', num_eeg_channels, sampling_rate, 'float32', 'unicorn_filtered'))
            self.outlet_power = StreamOutlet(StreamInfo('UnicornEEG_Power', 'EEG_Power', len(band_definitions), sampling_rate, 'float32', 'unicorn_power'))

            # Initialisierung der Filter
            self.b_band, self.a_band, self.zi_band = create_bandpass_filterblock(2, 70, sampling_rate, num_channels=num_eeg_channels)
            # b_band, a_band, zi_band = create_highpass_filter(2, sampling_rate, num_channels=num_eeg_channels)
            #b_notch, a_notch, zi_notch = create_notch_filter(50, sampling_rate, num_channels=num_eeg_channels)
            # Definiere die Frequenzen, die herausgefiltert werden sollen
            #notch_frequencies = [50, 100, 150]  # Beispiel: Netzfrequenz und Oberschwingungen
            self.cascaded_notch_filters = create_cascaded_notch_filters(50, 5, sampling_rate, num_channels=num_eeg_channels)

            # Initialize bandpass filters per band
            for band, (lowcut, highcut, list_of_channels) in band_definitions.items():
                self.ringbuffers_long.append(RingBuffer(sampling_rate * 500))
                for _ in list_of_channels:
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
                    timestamp_channel_index = BoardShim.get_timestamp_channel(self.board_id)
                    most_recent_timestamp = data[timestamp_channel_index, -1]

                    eeg_data = data[:8, :]
                    eeg_data_band_filtered, self.zi_band = apply_filter_block(eeg_data, self.b_band, self.a_band, self.zi_band)
                    eeg_data_final, self.cascaded_notch_filters = apply_cascaded_notch_filters(eeg_data_band_filtered, self.cascaded_notch_filters)

                    self.outlet.push_chunk(eeg_data_final.T.tolist(), timestamp=most_recent_timestamp)

                    j = 0
                    variances = []
                    for l, (band, (lowcut, highcut, list_of_channels)) in enumerate(band_definitions.items()):
                        variances_band = []
                        for channel in list_of_channels:
                            b_band_loop = self.b_bands[j]
                            a_band_loop = self.a_bands[j]
                            zi_band_loop = self.zi_bands[j]
                            eeg_data_band_loop, zi_band_loop = apply_filter(eeg_data[channel], b_band_loop, a_band_loop, zi_band_loop)
                            self.zi_bands[j] = zi_band_loop
                            self.ringbuffers[j].append(eeg_data_band_loop)

                            variances_channel = []
                            for k in range(-number_of_datapoints + 1, 1):
                                variances_channel.append(np.var(self.ringbuffers[j].get(k, int(BoardShim.get_sampling_rate(self.board_id) / 2))))
                            variances_band.append(variances_channel)

                            j += 1

                        final_variances_band = np.mean(variances_band, axis=0)
                        self.ringbuffers_long[l].append(final_variances_band)
                        variances_history_band = self.ringbuffers_long[l].get(0, sampling_rate * 500)
                        variances_history_band_cleaned = variances_history_band[variances_history_band != 0]
                        low_quantile = np.quantile(variances_history_band_cleaned, 0.05)
                        high_quantile = np.quantile(variances_history_band_cleaned, 0.95)

                        denom = high_quantile - low_quantile
                        if denom == 0:
                            denom = 1
                        variances.append((final_variances_band - low_quantile) / denom)

                    self.outlet_power.push_chunk(np.array(variances).T.tolist(), timestamp=most_recent_timestamp)

        except KeyboardInterrupt:
            print("Streaming beendet durch Benutzer.")
        finally:
            self.board.stop_stream()
            self.board.release_session()
            print("Verbindung geschlossen.")


    def start(self):
        """
        Starts the EEG data stream processing.
        """
        if not self.connected:
            print("EEG-Streamer ist nicht verbunden. Rufe initialize() auf, bevor du startest.")
            return

        threading.Thread(target=self.process_eeg, daemon=True).start()

