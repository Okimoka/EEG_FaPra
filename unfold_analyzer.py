import random
from matplotlib import pyplot as plt
import numpy as np
import threading
import pandas as pd
import time
from scipy.signal import resample

from juliacall import Pkg as jlPkg
from juliacall import Main as jl

from streamer import Streamer
from plotter import Plotter

"""
Requires LSL stream "UnicornEEG_Filtered" (produced by eeg_streamer.py)
Produces either a live matplotlib plot of the Unfold model results, or sends the Unfold results dataframe to a flask websocket

Whenever a saccade happens, there is a delay of window[1] (time until all neccessary eeg data is collected)
Then the eeg data within the specified window is taken from the stream history and added as a trial to the unfold model

Information about saccades (timing and amplitude) is handled by the saccade_amplitude_streamer.py
It is responsible for calling add_saccade(amplitude), triggering the whole model update

"""



class UnfoldAnalyzer(Streamer, Plotter):
    def __init__(self, data_queue, PLOT_UNFOLD_LOCALLY):
        self.plot_unfold_locally = PLOT_UNFOLD_LOCALLY
        self.data_buffer = None #Just for plotting
        Streamer.__init__(self)
        Plotter.__init__(self, draw_plot=self.plot_unfold_locally, data_buffer=self.data_buffer)
        self.window = [-0.2, 0.8] #window in which to capture the EEG data
        self.sampling_rate = 250 # gtec unicorn sampling rate in Hz
        self.saccade_history = []
        self.saccade_amplitude_history = []
        self.max_saccade_history = 100 # number of trials 
        self.data_ready = False # is the model ready to take new data? 
        self.data_queue = data_queue # to communicate with websocket subprocess
        

    # Import all necessary Julia packages
    def init_julia(self):
        jlPkg.activate("eegProject") # activate project environment
        packages = ["Unfold", "DataFrames", "PyMNE"]
        for package in packages:
            if(jl.seval('Base.find_package("'+package+'") == nothing')): # only import if not already imported
                print("Installing package " + package)
                jlPkg.add(package)
            else:
                print("Package " + package + " already installed.")
        for package in packages:   
            print("Using package " + package)
            jl.seval("using "+package) # seval is probably not optimal here, but it works

    def initialize(self):
        # only a steady stream of EEG data is required
        # saccade amplitude is passed though add_saccade, hence no second input stream is needed
        # output doesnt happen through an LSL outlet but through a websocket/direct plot
        Streamer.initialize(self, "UnicornEEG_Filtered", None, None)


    def start(self):
        Streamer.start(self, pull_1=True, print_lag_1=True)

        if(self.plot_unfold_locally):
            Plotter.start(self)
    

    def add_saccade(self, amplitude):
        # wait window[1] seconds
        t = threading.Timer(self.window[1], lambda: self.process_saccade_event(amplitude)) #needs to be non-blocking. also juliacall can only run in main thread
        t.start()
        

    # prepare all the data for the Unfold model (saccade_history, eeg_window)
    # and set data_ready to True so the main thread knows to start calling the julia functions
    def process_saccade_event(self, amplitude):
        # we waited window[1] seconds at this point
        t_end = time.time()
        t_start = t_end + self.window[0] - self.window[1] #window[0] is negative

        # get all samples from the stream history whose timestamps are within the time window
        def extract_window(stream):
            return [
                (ts, data) for ts, data in stream.history
                if t_start <= ts <= t_end
            ]

        eeg_window = extract_window(self.input_stream_1)

        if(len(eeg_window) == 0):
            print("No data in window")
            return

        
        print_valid_sample_count = len([sample for sample in eeg_window if sample[1] is not None and len(sample[1]) == 8])
        print_start_time = min(sample[0] for sample in eeg_window) - time.time() + self.window[1] #earliest sample in the window
        print_end_time = max(sample[0] for sample in eeg_window) - time.time() + self.window[1] #latest sample in the window
        print_variance = np.var([sample[1] for sample in eeg_window], axis=0).astype(int)

        print(f"Captured {print_valid_sample_count} samples between {print_start_time:.2f} and {print_end_time:.2f} with variances {print_variance.tolist()}")
        print("Random sample in window: " + str(random.choice(eeg_window)))
        print("--------------------------")

        self.saccade_history.append([eeg_window, amplitude])

        if len(self.saccade_history) > self.max_saccade_history:
            self.saccade_history.pop(0)

        self.data_ready = True


    #saccade_history holds a list of [eeg_window, amplitude]
    #eeg_window is a list of samples (ts,[8 channels]) within the specified time window, around a specific saccade of that amplitude
    def get_latest_eeg_epochs(self):
        if not self.saccade_history:
            return np.empty((8, 0, 0))

        # desired number of samples per epoch
        n_timepoints = int((self.window[1] - self.window[0]) * self.sampling_rate)

        #get data from the saccade history into the correct shape
        #basically just add all valid transposed eeg samples to epochs
        epochs = []
        for eeg_window, _ in self.saccade_history:
            eeg_data = np.array([sample for _, sample in eeg_window])
            if eeg_data.shape[0] == 0:
                print("Warning: Invalid sample in eeg_window")
                continue
            eeg_data = eeg_data.T

            # amount of samples varies slightly for every saccade, so it is normalized to an average of windowsize*sampling_rate
            epochs.append(resample(eeg_data, n_timepoints, axis=1))

        if not epochs: #no valid samples
            return np.empty((8, 0, 0))

        return np.stack(epochs, axis=-1)
    


    def get_latest_fixation_metadata(self):
        # empty dataframe if no saccades have been detected yet
        if not self.saccade_history:
            return pd.DataFrame(columns=["saccade_amplitude"])

        # dataframe with one column "saccade_amplitude" and as many rows as there are saccades
        return pd.DataFrame({
            "saccade_amplitude": [amp for _, amp in self.saccade_history]
        })
    



    def live_fit_and_plot(self):

        eeg_data = self.get_latest_eeg_epochs()  # ndarray of shape (n_channels, n_times, n_epochs)
        metadata = self.get_latest_fixation_metadata()  # DataFrame with saccade_amplitude column

        (n_channels, n_times, n_epochs) = eeg_data.shape
        print("EEG data shape:", eeg_data.shape)
        #print("Metadata shape:", metadata.shape)

        times = np.linspace(self.window[0], self.window[1], n_times)
        
        # the next lines were adapted from
        # https://github.com/unfoldtoolbox/Unfold.jl/blob/bd6fbe671b8bace3f0b72b825c77d1a89bae048d/docs/src/HowTo/juliacall_unfold.ipynb
        jl_formula = jl.seval("@formula 0 ~ 1 + saccade_amplitude")
        jl_event_df = jl.DataFrame(saccade_amplitude=metadata['saccade_amplitude'].values)
        m = jl.Unfold.fit(jl.Unfold.UnfoldModel, jl_formula, jl_event_df, eeg_data, times)
        results_jl = jl.Unfold.coeftable(m)

        results_py = pd.DataFrame({
            'channel': results_jl.channel,
            'coefname': results_jl.coefname,
            'estimate': results_jl.estimate,
            'time': results_jl.time
        })

        # Sample output of results_py
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

        #Ch5 corresponds to the center back electrode
        if(self.plot_unfold_locally):
            self.data_buffer = results_py[results_py.channel == 5]
        else:
            self.emit_model_results_to_js(results_py, 5)
        

    def emit_model_results_to_js(self, results_py, channel_id):
        df = results_py[results_py.channel == channel_id]

        # Reshape dataframe into a json object
        grouped = df.groupby('coefname').agg({
            'time': list,
            'estimate': list
        }).to_dict(orient='index')

        payload = {
            "channel": int(channel_id),
            "model_results": grouped
        }

        #data queue of the flask websocket
        self.data_queue.put(payload)
