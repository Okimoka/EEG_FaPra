import random
from matplotlib import pyplot as plt
import numpy as np
import threading
import pandas as pd
import time

from juliacall import Pkg as jlPkg
from juliacall import Main as jl

from streamer import Streamer
from plotter import Plotter

"""

TODO comment this class

"""



class UnfoldAnalyzer(Streamer, Plotter):
    def __init__(self, data_queue, PLOT_UNFOLD_LOCALLY):
        self.plot_unfold_locally = PLOT_UNFOLD_LOCALLY
        self.data_buffer = None #Just for plotting
        Streamer.__init__(self)
        Plotter.__init__(self, draw_plot=self.plot_unfold_locally, data_buffer=self.data_buffer)
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

    def initialize(self, DEBUG_MODE):
        if(DEBUG_MODE):
            super().initialize("FakeEEG", "ccs-neon-001_Neon Gaze", None)
        else:
            super().initialize("UnicornEEG_Filtered", "ccs-neon-001_Neon Gaze", None)


    def start(self):
        # The saccade amplitude lsl stream is not really needed, instead the callback from saccade_amplitude_streamer is used
        Streamer.start(self, pull_1=True, print_lag_1=True)

        if(self.plot_unfold_locally):
            Plotter.start(self)
        
        # All pulls need to be in separate threads, or else we get drifting timestamps
        #threading.Thread(target=self.lsl_pull_thread, args=(self.fixations_stream, "Gaze    "), daemon=True).start()
        ###threading.Thread(target=self.lsl_pull_thread, args=(self.input_stream_1, "EEG     "), daemon=True).start()
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
        eeg_window = extract_window(self.input_stream_1)
        #fixations_window = extract_window(self.fixations_stream)
        #saccades_window = extract_window(self.saccade_stream)

        if(len(eeg_window) == 0): #or len(fixations_window) == 0 or len(saccades_window) == 0):
            print("No data in window")
            return

        print_start_time = min(sample[0] for sample in eeg_window) - time.time() + self.window[1] # add window[1] for both since we waited window[1] seconds
        print_end_time = max(sample[0] for sample in eeg_window) - time.time() + self.window[1]
        print_variance = np.var([sample[1] for sample in eeg_window], axis=0).astype(int)

        print(f"Captured {len(eeg_window)} samples between {print_start_time:.2f} and {print_end_time:.2f} with variance {print_variance.tolist()}")

        #print("Captured " + str(len(eeg_window)) + " samples between " + str(min([sample[0] for sample in eeg_window])-time.time()) + " and " + str(max([sample[0] for sample in eeg_window])-time.time()) + " with variance " + str(np.var([sample[1] for sample in eeg_window], axis=0)))
        print("Random sample in window: " + str(random.choice(eeg_window)))
        #print("len(samples) (3s), len(samples) (windowlength), newest timestamp - currenttime, oldest time - currenttime")
        #current_time = time.time()
        #print(str(len(eeg_window)) + "  " + str(len(self.input_stream_1.history)) + "  " + str(max([sample[0] for sample in self.input_stream_1.history])-current_time) + "  " + str(min([sample[0] for sample in self.input_stream_1.history])-current_time))
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

        # Sample output
        #    channel           coefname   estimate  time
        #0           1        (Intercept)  -0.055425  -0.2
        #1           2        (Intercept)  -0.371453  -0.2
        #2           3        (Intercept)  -0.257530  -0.2
        #3           4        (Intercept)  -0.603811  -0.2
        #4           5        (Intercept)  -0.462772  -0.2
        #...       ...                ...        ...   ...
        #3547        4  saccade_amplitude   1.134658   0.8
        #3548        5  saccade_amplitude   9.985816   0.8
        #3549        6  saccade_amplitude  14.477165   0.8
        #3550        7  saccade_amplitude   4.530100   0.8
        #3551        8  saccade_amplitude   2.348762   0.8


        if(self.plot_unfold_locally):
            self.data_buffer = results_py[results_py.channel == 2]
        else:
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