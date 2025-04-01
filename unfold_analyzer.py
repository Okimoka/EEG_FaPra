#from pupil_labs.real_time_screen_gaze import marker_generator
from matplotlib import pyplot as plt
import numpy as np
import threading


from juliacall import Pkg as jlPkg
from juliacall import Main as jl
import time
from pylsl import StreamInfo, StreamOutlet

from mne.datasets.limo import load_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lsl_stream import LSLStream



class UnfoldAnalyzer:
    def __init__(self, data_queue):
        self.connected = False
        self.latest_timestamp = 0
        self.window = [-0.2, 0.8]
        self.saccade_history = []
        self.saccade_amplitude_history = []
        self.max_saccade_history = 30
        self.first_timestamps = []
        self.data_ready = False
        self.data_queue = data_queue


    def init_julia(self):
        jlPkg.activate("MyProject")
        packages = ["Unfold", "DataFrames", "PyMNE"] # UnfoldMakie,CairoMakie
        for package in packages:
            if(jl.seval('Base.find_package("'+package+'") == nothing')):
                print("Installing package " + package)
                jlPkg.add(package)
            else:
                print("Package " + package + " already installed.")
        for package in packages:   
            print("Using package " + package)
            jl.seval("using "+package)

    def initialize(self):

        #self.fixations_stream = LSLStream("Fixations", track_history_seconds=3.0) #binary fixation
        #self.eeg_stream = LSLStream("UnicornEEG_Filtered", track_history_seconds=3.0) #8 channel eeg data
        self.eeg_stream = LSLStream("FakeEEG", track_history_seconds=3.0) #8 channel eeg data
        #self.saccade_stream = LSLStream("Saccades", track_history_seconds=3.0) #most recent saccade amplitude
        self.pupil_stream = LSLStream("ccs-neon-001_Neon Gaze", track_history_seconds=3.0) #pupil size

        self.outlet = StreamOutlet(StreamInfo('Unfold', 'Markers', 2, 20, 'float32', 'fixation_outlet'))

        #self.connected = self.fixations_stream.connected and self.eeg_stream.connected and self.saccade_stream.connected
        self.connected = self.eeg_stream.connected


    
    def lsl_pull_thread(self, stream, label):
        while True:
            ts, sample = stream.pull_sample()
            if(int(time.time() - ts) > 0):
                print(f"[{label}] Lag:", int(time.time() - ts))
            time.sleep(0.001)


    def start(self):
        # All pulls need to be in separate threads, or else we get drifting timestamps
        #threading.Thread(target=self.lsl_pull_thread, args=(self.fixations_stream, "Gaze    "), daemon=True).start()
        threading.Thread(target=self.lsl_pull_thread, args=(self.eeg_stream, "EEG     "), daemon=True).start()
        #threading.Thread(target=self.lsl_pull_thread, args=(self.saccade_stream, "Saccades"), daemon=True).start()
    

    def add_saccade(self, amplitude):
        t = threading.Timer(self.window[1], lambda: self._process_saccade_event(amplitude)) #needs to be non-blocking. also juliacall can only run in main thread
        t.start()
        

    def _process_saccade_event(self, amplitude):
        t_end = time.time()
        t_start = t_end + self.window[0] - self.window[1]

        def extract_window(stream):
            return [
                (ts, data) for ts, data in stream.history
                if t_start <= ts <= t_end
            ]

        # Extract data within time window
        eeg_window = extract_window(self.eeg_stream)
        #fixations_window = extract_window(self.fixations_stream)
        #saccades_window = extract_window(self.saccade_stream)

        if(len(eeg_window) == 0): #or len(fixations_window) == 0 or len(saccades_window) == 0):
            print("No data in window")
            return

        print("len(samples) (3s), len(samples) (windowlength), newest timestamp - currenttime, oldest time - currenttime")
        current_time = time.time()
        print(str(len(eeg_window)) + "  " + str(len(self.eeg_stream.history)) + "  " + str(max([sample[0] for sample in self.eeg_stream.history])-current_time) + "  " + str(min([sample[0] for sample in self.eeg_stream.history])-current_time))
        print("--------------------------")

        self.saccade_history.append([eeg_window, amplitude])

        if len(self.saccade_history) > self.max_saccade_history:
            self.saccade_history.pop(0)

        self.data_ready = True


    #saccade_history holds a list of [eeg data, amplitude]
    #where eeg data is a list of samples within the specified time window, around a specific saccade of the amplitude
    def get_latest_eeg_epochs(self):
        if not self.saccade_history:
            return np.empty((8, 0, 0))

        #get data from the saccade history into the correct shape
        epochs = []
        for eeg_window, _ in self.saccade_history:
            eeg_data = np.array([sample for _, sample in eeg_window])
            if eeg_data.shape[0] == 0:
                continue
            eeg_data = eeg_data.T
            epochs.append(eeg_data)

        if not epochs:
            return np.empty((8, 0, 0))

        #the epoch with the smallest number of samples determines what shape we use
        #probably not the best solution to simply cut these samples off
        min_length = min(epoch.shape[1] for epoch in epochs)
        epochs = [epoch[:, :min_length] for epoch in epochs]
        return np.stack(epochs, axis=-1)
    

    def get_latest_fixation_metadata(self):
        if not self.saccade_history:
            return pd.DataFrame(columns=["saccade_amplitude"])

        return pd.DataFrame({
            "saccade_amplitude": [amp for _, amp in self.saccade_history]
        })
    

    #def plot_channel_coef(results_py, channel_id):
    #    df = results_py[results_py.channel == channel_id]
    #    ax = sns.lineplot(data=df, x='time', y='estimate', hue='coefname')
    #    ax.set(title=f'Channel {channel_id}', xlabel='Time [s]', ylabel='Beta Coefficient')
    #    plt.show()

    def live_fit_and_plot(self):
        eeg_data = self.get_latest_eeg_epochs()  # (n_channels, n_times, n_epochs)
        metadata = self.get_latest_fixation_metadata()  # DataFrame with saccade_amplitude

        (n_channels, n_times, n_epochs) = eeg_data.shape
        print("EEG data shape:", eeg_data.shape)
        #print("Metadata shape:", metadata.shape)

        times = np.linspace(self.window[0], self.window[1], n_times)#[1:-1]
        
        jl_formula = jl.seval("@formula 0 ~ 1 + saccade_amplitude")

        jl_event_df = jl.DataFrame(saccade_amplitude=metadata['saccade_amplitude'].values)

        m = jl.Unfold.fit(jl.Unfold.UnfoldModel, jl_formula, jl_event_df, eeg_data, times)
        # Extract & plot
        results_jl = jl.Unfold.coeftable(m)

        results_py = pd.DataFrame({
            'channel': results_jl.channel,
            'coefname': results_jl.coefname,
            'estimate': results_jl.estimate,
            'time': results_jl.time
        })

        #print(results_py)
        self.emit_model_results_to_js(results_py, 1)
        

    def emit_model_results_to_js(self, results_py, channel_id):
        df = results_py[results_py.channel == channel_id]

        # Reshape dataframe into a json object
        grouped_data = {}
        for coef in df['coefname'].unique():
            coef_df = df[df['coefname'] == coef]
            grouped_data[coef] = {
                "time": coef_df["time"].tolist(),
                "estimate": coef_df["estimate"].tolist()
            }

        payload = {
            "channel": int(channel_id),
            "model_results": grouped_data
        }

        #data queue of the flask websocket
        self.data_queue.put(payload)

