import numpy as np
import logging
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pylsl import StreamInfo, StreamOutlet
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi

# Logging für Debugging aktivieren
logging.basicConfig(level=logging.DEBUG)

band_definitions = {
    "Delta": (0.5, 4, [0, 2]),
    "Theta": (4, 8, [0, 2]),
    "Alpha": (8, 12, [4,5,6,7]),
    "Beta": (12, 30, [2])    
}

class RingBuffer:
    def __init__(self, size: int, dtype=np.float32):
        """
        Initialisiert den Ringbuffer.
        :param size: Maximale Größe des Buffers.
        :param dtype: Datentyp der gespeicherten Werte.
        """
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)
        self.index = 0
        self.full = False

    def append(self, values):
        """
        Fügt einen oder mehrere Werte zum Buffer hinzu.
        :param values: Ein einzelner Wert oder ein numpy-Array von Werten.
        """
        values = np.atleast_1d(values)  # Werte in ein Array konvertieren, falls sie skalare Werte sind
        n_values = len(values)

        if n_values >= self.size:
            # Nur die letzten `size` Werte einfügen, da ältere überschrieben werden
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
        Gibt eine Teilmenge des Buffers basierend auf einem relativen Index und der Länge zurück.
        :param rel_index: Relativer Index (negativ bedeutet zurück, 0 ist der neueste Wert).
        :param length: Länge des zurückgegebenen Arrays.
        :return: Ein numpy-Array mit den Werten.
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


# Bandpass-Filterinitialisierung mit Zustand
def create_bandpass_filterblock(lowcut, highcut, fs, order=3, num_channels=8):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    zi = [lfilter_zi(b, a) for _ in range(num_channels)]  # Zustand für jeden Kanal
    return b, a, zi

def create_bandpass_filter(lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    zi = lfilter_zi(b, a)
    return b, a, zi


# Hochpass-Filterinitialisierung mit Zustand
def create_highpass_filter(cutoff, fs, order=4, num_channels=8):
    nyquist = 0.5 * fs  # Nyquist-Frequenz
    normalized_cutoff = cutoff / nyquist  # Normierte Grenzfrequenz
    b, a = butter(order, normalized_cutoff, btype='high')  # Hochpassfilter entwerfen
    zi = [lfilter_zi(b, a) for _ in range(num_channels)]  # Initialzustand für jeden Kanal
    return b, a, zi


# Notch-Filterinitialisierung mit Zustand
def create_notch_filter(freq, fs, quality_factor=10, num_channels=8):
    nyquist = 0.5 * fs  # Nyquist-Frequenz
    normalized_notch = freq / nyquist  # Normierte Grenzfrequenz
    b, a = iirnotch(normalized_notch, quality_factor, fs)
    zi = [lfilter_zi(b, a) for _ in range(num_channels)]  # Zustand für jeden Kanal
    return b, a, zi

# Notch-Filterinitialisierung mit Zustand für mehrere Frequenzen
def create_cascaded_notch_filters(freq, iterations, fs, quality_factor=30, num_channels=8):
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
    num_channels = samples.shape[0]  # Anzahl der Kanäle
    y = samples.copy()  # Kopiere die Eingabedaten
    updated_filters = []

    for b, a, zi_list in filters:
        new_zi_list = []
        for channel in range(num_channels):
            # Filter für jeden Kanal anwenden
            filtered_channel, zi_channel = lfilter(b, a, y[channel], zi=zi_list[channel])
            y[channel] = filtered_channel
            new_zi_list.append(zi_channel)
        updated_filters.append((b, a, new_zi_list))  # Aktualisierter Zustand für diesen Filter

    return y, updated_filters


# Beispiel für die Filterung eines Blocks von Samples
def apply_filter_block(samples, b, a, zi):
    num_channels = samples.shape[0]  # Anzahl der Kanäle
    y = np.zeros_like(samples)  # Output mit derselben Form wie die Eingabe
    updated_zi = []

    for channel in range(num_channels):
        # Filterung für jeden Kanal separat
        filtered_channel, zi_channel = lfilter(b, a, samples[channel], zi=zi[channel])
        y[channel] = filtered_channel
        updated_zi.append(zi_channel)

    return y, updated_zi  # Gibt gefilterte Samples und aktualisierten Zustand zurück

def apply_filter(samples, b, a, zi):
    y, zi_update = lfilter(b, a, samples, zi=zi)
    return y, zi_update




# Hauptprogramm zur Initialisierung und zum Starten der Threads
if __name__ == "__main__":
    params = BrainFlowInputParams()
    board_id = BoardIds.UNICORN_BOARD.value
    connected = False

    try:
        board = BoardShim(board_id, params)
        board.prepare_session()
        print("Erfolgreich mit Unicorn-Gerät verbunden.")
        connected = True
    except Exception as e:
        print(f"Fehler bei der Verbindung mit dem Unicorn-Gerät: {e}")

    if connected:
        try:
            board.start_stream()
            print("EEG-Stream erfolgreich gestartet.")

            timestamp_channel_index = BoardShim.get_timestamp_channel(board_id)

            num_eeg_channels = 8
            sampling_rate = BoardShim.get_sampling_rate(board_id)
            info = StreamInfo('UnicornEEG_Filtered', 'EEG', num_eeg_channels, sampling_rate, 'float32', 'unicorn_filtered')
            outlet = StreamOutlet(info)

            info_power = StreamInfo('UnicornEEG_Power', 'EEG_Power', len(band_definitions), sampling_rate, 'float32', 'unicorn_power')
            outlet_power = StreamOutlet(info_power)


            # Initialisierung der Filter
            b_band, a_band, zi_band = create_bandpass_filterblock(2, 70, sampling_rate, num_channels=num_eeg_channels)
           # b_band, a_band, zi_band = create_highpass_filter(2, sampling_rate, num_channels=num_eeg_channels)
            #b_notch, a_notch, zi_notch = create_notch_filter(50, sampling_rate, num_channels=num_eeg_channels)
            # Definiere die Frequenzen, die herausgefiltert werden sollen
            #notch_frequencies = [50, 100, 150]  # Beispiel: Netzfrequenz und Oberschwingungen
            cascaded_notch_filters = create_cascaded_notch_filters(50, 5, sampling_rate, num_channels=num_eeg_channels)
            
            b_bands = []
            a_bands = []
            zi_bands = []
            ringbuffers = []
            ringbuffers_long = []
            for band, (lowcut, highcut, list_of_channels) in band_definitions.items():
                ringbuffers_long.append(RingBuffer(sampling_rate*500))
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

                    eeg_data = data[:num_eeg_channels, :]
                    eeg_data_band_filtered , zi_band = apply_filter_block(eeg_data, b_band, a_band, zi_band)
                    #eeg_data_final, zi_notch = apply_filter_block(eeg_data_band_filtered, b_notch, a_notch, zi_notch)
                    eeg_data_final, cascaded_notch_filters = apply_cascaded_notch_filters(eeg_data_band_filtered, cascaded_notch_filters)

                    # Transpose the data to (samples x channels) for LSL
                    eeg_data_final_LSL = eeg_data_final.T

                    # Push the filtered data as a chunk to the outlet
                    outlet.push_chunk(eeg_data_final_LSL.tolist(), timestamp=most_recent_timestamp)

                    j=0
                    variances = []
                    for l, (band, (lowcut, highcut, list_of_channels)) in enumerate(band_definitions.items()):
                        variances_band = []
                        for channel in list_of_channels:
                            b_band_loop = b_bands[j] 
                            a_band_loop = a_bands[j]
                            zi_band_loop = zi_bands[j]
                            eeg_data_band_loop , zi_band_loop = apply_filter(eeg_data[channel], b_band_loop, a_band_loop, zi_band_loop)
                            zi_bands[j] = zi_band_loop

                            ringbuffers[j].append(eeg_data_band_loop)
                        
                            variances_channel = []
                            for k in range(-number_of_datapoints+1, 1):
                                variances_channel.append(np.var(ringbuffers[j].get(k,int(sampling_rate/2))))
                            variances_band.append(variances_channel)

                            j+=1
                        
                        final_variances_band = np.mean(variances_band, axis=0)
                        ringbuffers_long[l].append(final_variances_band)
                        variances_history_band = ringbuffers_long[l].get(0, sampling_rate*500)
                        variances_history_band_cleaned = variances_history_band[variances_history_band != 0]
                        low_quantile = np.quantile(variances_history_band_cleaned, 0.2)
                        high_quantile = np.quantile(variances_history_band_cleaned, 0.8)

                        denom = high_quantile - low_quantile
                        if denom == 0:
                            denom = 1
                        variances.append((final_variances_band - low_quantile)/denom)
                    variances = np.array(variances)
                    outlet_power.push_chunk(variances.T.tolist(), timestamp=most_recent_timestamp)



            print("Streaming gefilterte EEG-Daten... Drücke CTRL+C, um den Stream zu beenden.")


        except KeyboardInterrupt:
            print("Streaming beendet durch Benutzer.")
        finally:
            board.stop_stream()
            board.release_session()
            print("Verbindung geschlossen.")
