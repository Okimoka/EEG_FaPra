#from pupil_labs.real_time_screen_gaze import marker_generator
from matplotlib import pyplot as plt
from pupil_labs.realtime_api.simple import discover_one_device
from pupil_labs.real_time_screen_gaze.gaze_mapper import GazeMapper
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_stream
import random

import time
from collections import deque
import nest_asyncio
import numpy as np
from pupil_labs.realtime_api.simple import discover_one_device, discover_devices
from blink_detector.blink_detector import blink_detection_pipeline
from blink_detector.helper import (
    stream_images_and_timestamps,
    update_array,
    compute_blink_rate,
    plot_blink_rate,
)

import threading
import queue
from collections import deque

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from juliacall import Pkg as jlPkg
from juliacall import Main as jl
import juliacall
import time
import mne
import seaborn
import pandas
import math

from mne.datasets.limo import load_data
import pandas as pd
import seaborn as sns



def blink_detection_thread(blink_queue, device):
    left_images, right_images, timestamps = stream_images_and_timestamps(device)
    blink_generator = blink_detection_pipeline(left_images, right_images, timestamps)

    while True:
        try:
            blink_event = next(blink_generator)
            blink_queue.put(1)  # Put a '1' in the queue to indicate a blink
        except StopIteration:
            continue  # No blink detected, just continue checking


def main_eyetracking_and_streaming():
    nest_asyncio.apply()
    device = discover_one_device()
    calibration = device.get_calibration()
    gaze_mapper = GazeMapper(calibration)

    left_images, right_images, timestamps = stream_images_and_timestamps(device)
    blink_generator = blink_detection_pipeline(left_images, right_images, timestamps)

    blink_queue = queue.Queue()
    blink_counter = 0 # 0: no blink, 1: blink detected
    threading.Thread(target=blink_detection_thread, args=(blink_queue, device), daemon=True).start()

    scale = 24
    margin = 32
    screen_width = 1920
    screen_height = 1080
    image_width = scale * 8  # 8x8 pixels scaled by 'scale'
    image_height = scale * 8

    marker_verts = {
        0: [  # top left
            (margin, margin),  # Top left corner of the top-left image
            (margin + image_width, margin),  # Top right corner of the top-left image
            (margin + image_width, margin + image_height),  # Bottom right corner of the top-left image
            (margin, margin + image_height),  # Bottom left corner of the top-left image
        ],
        1: [  # top right
            (screen_width - margin - image_width, margin),  # Top left corner of the top-right image
            (screen_width - margin, margin),  # Top right corner of the top-right image
            (screen_width - margin, margin + image_height),  # Bottom right corner of the top-right image
            (screen_width - margin - image_width, margin + image_height),  # Bottom left corner of the top-right image
        ],
        2: [  # bottom left
            (margin, screen_height - margin - image_height),  # Top left corner of the bottom-left image
            (margin + image_width, screen_height - margin - image_height),  # Top right corner of the bottom-left image
            (margin + image_width, screen_height - margin),  # Bottom right corner of the bottom-left image
            (margin, screen_height - margin),  # Bottom left corner of the bottom-left image
        ],
        3: [  # bottom right
            (screen_width - margin - image_width, screen_height - margin - image_height),  # Top left corner of the bottom-right image
            (screen_width - margin, screen_height - margin - image_height),  # Top right corner of the bottom-right image
            (screen_width - margin, screen_height - margin),  # Bottom right corner of the bottom-right image
            (screen_width - margin - image_width, screen_height - margin),  # Bottom left corner of the bottom-right image
        ],
    }

    screen_size = (1920, 1080)

    screen_surface = gaze_mapper.add_surface(
        marker_verts,
        screen_size
    )


    # Create multiple LSL stream infos and outlets for surface gaze with bogus data
    # Currently, we dont send any bogus data. This was used to test the limits of LSL
    outlets = []
    n = 1  # Number of SurfaceGaze outlets
    bogus_channels_number = 0
    for i in range(n):
        info = StreamInfo(f'SurfaceGaze_{i}', 'Gaze', bogus_channels_number+3, 200, 'float32', f'surface_gaze_id_{i}')
        outlet = StreamOutlet(info)
        outlets.append(outlet)

    print("LSL outlets created. Streaming surface gaze data with bogus channels...")


    try:
        while True:
            frame, gaze = device.receive_matched_scene_video_frame_and_gaze()
            result = gaze_mapper.process_frame(frame, gaze)

            # Check the queue for blink events
            if not blink_queue.empty():
                blink_queue.get()  # Remove the event from the queue
                blink_counter += 1


            for surface_gaze in result.mapped_gaze[screen_surface.uid]:
                # Generate bogus data (10 random channels)
                bogus_data = [random.random() for _ in range(bogus_channels_number)]
                # Combine gaze data and bogus data
                sample = [surface_gaze.x, surface_gaze.y, blink_counter] + bogus_data
                # Stream combined data via all LSL outlets
                for outlet in outlets:
                    outlet.push_sample(sample)
                print(f"Gaze at {surface_gaze.x}, {surface_gaze.y}, Bogus: {bogus_data}")   


            
    except KeyboardInterrupt:
        print("Streaming stopped.")


# get all trials to same sample length
# due to fluctuations in sampling speed this might not always be the case
def adjust_lists(nested_lists, m):
    adjusted_lists = []
    for lst in nested_lists:
        if len(lst) > m:
            # Cut the list to the desired length m
            adjusted_lists.append(lst[:m])
        else:
            # Extend the list if it's shorter than m
            if lst:  # Make sure the list is not empty to avoid IndexError
                last_element = lst[-1]
                adjusted_lists.append(lst + [last_element] * (m - len(lst)))
            else:
                # If the list is empty, consider what to extend with, here I use a placeholder None
                adjusted_lists.append([None] * m)
    return adjusted_lists



def process_eeg_data(collected_samples):
    global shared_data
    # Process the EEG data
    print("Processing collected EEG data...")
    print(collected_samples)  # For demonstration, just print the data

    """
    current shape:
    [[(timestamp, [8 channels]), ...], [fuer alle trials]]
    """

    #face_trials = []
    #contrast_trials = []

    #for i, trial in enumerate(collected_samples):
    #    if(i%2 == 0):
    #        face_trials.append(trial)
    #    else:
    #        contrast_trials.append(trial)
    shared_data = collected_samples
    set_data_ready()



def process_eeg_data_jl():

    collected_samples = shared_data
    srate = 250 #self.streams["UnicornEEG_Filtered"].sampling_rate
    trials = 10
    window = [-0.2, 1] #hardcoded again further down, fix in the future
    window_size = window[1] - window[0]
    n_points = int(np.floor(srate * window_size) + 1)+2
    times = np.linspace(window[0], window[1], n_points)[1:-1]

    collected_samples = adjust_lists(collected_samples, len(times))

    #####print(eeg_data)

    data = np.random.uniform(0, 0, (8,len(times),trials))

    #(channels, samples, trials)
    #(8, 176, 5)
    for i, trial in enumerate(collected_samples):
        for j, (timestamp, sample) in enumerate(trial):
            for k, channel in enumerate(sample):
                data[k, j, i] = channel

    print("Data shape:", data.shape)
    metadata = pd.DataFrame({'stimulus_onset': [1.0] * trials, 'condition': ['face','base'] * 5})
    metadata_jl = jl.DataFrame(stimulus_onset=metadata["stimulus_onset"].values, condition=metadata["condition"].values)
    ###OVERWRITING WITH RANDOM DATA
    ##data2 = np.random.uniform(-1.61, 1.64, (len(eeg_data),))
    ##data = np.random.uniform(-87, 95, (8,len(times),len(eeg_data)))

    #self.draw_plot(latency,data)
    #compute phase coherence

    #df_string = "DataFrame((condition=String"+str(condition)+",latency=Float64"+str(latency)+"))"

    formula = jl.seval("@formula 0 ~ 1 + stimulus_onset + condition")
    m = jl.Unfold.fit(jl.Unfold.UnfoldModel,formula,metadata_jl,data,times)
    ####len(times) = floor(srate * windowsize)+1

    results_jl = jl.Unfold.coeftable(m)

    results_py = pd.DataFrame({'channel': results_jl.channel,
                            'coefname': results_jl.coefname,
                            'estimate': results_jl.estimate,
                            'time': results_jl.time})

    print(results_py)

    results_ch43 = results_py[results_py.channel == 3]

    # Plot the coefficient estimates over time

    # Extract coefficient table
    #results_jl = Unfold.coeftable(m)
    print("-------------------------RESULT-------------------------")
    print(results_ch43.time)
    print(results_ch43.estimate)


    #(results_ch43.time,results_ch43.estimate)

    sns.lineplot(
    x=results_ch43["time"],
    y=results_ch43["estimate"],
    hue=results_ch43["coefname"]
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Coefficient Estimate")
    plt.title("Effect of Visual Stimulus on Channel 1")
    plt.show()

    #DataFrame((continuous=Float64[1, 2], condition=String[3, 4], latency=Float64[5, 6]))

    #DataFrame((a=[1, 2], b=[3, 4]))

    # Row │ a      b
    #     │ Int64  Int64
    #─────┼──────────────----
    #   1 │     1      3    5
    #   2 │     2      4    6


    #  Row │ continuous  condition  latency
    #      │ Float64     String     Int64
    #──────┼────────────────────────────────
    #    1 │   2.77778   car             62
    #    2 │  -5.0       face           132
    #    3 │  -1.66667   car            196
    #    4 │  -5.0       car            249
    #    5 │   5.0       car            303
    #    6 │  -0.555556  car            366
    #    7 │  -2.77778   car            432
    #  ⋮   │     ⋮           ⋮         ⋮
    # 1994 │   3.88889   car         119798
    # 1995 │   0.555556  car         119856
    # 1996 │   0.555556  face        119925
    # 1997 │  -3.88889   face        119978
    # 1998 │  -3.88889   car         120030
    # 1999 │  -0.555556  face        120096
    # 2000 │  -1.66667   face        120154




def init_unfold():
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



class LSLHandler:
    def __init__(self, stream_name='UnicornEEG_Filtered'):
        self.stream_name = stream_name
        #self.stream_type = stream_type
        self.inlet = None
        #self.initialize_stream()

    def initialize_stream(self):
        print("Looking for an EEG stream...")
        try:
            streams = resolve_stream('name', self.stream_name)
            self.inlet = StreamInlet(streams[0])
            print("Stream found and connected!")
            return True
        except Exception as e:
            print(f"Failed to connect to the stream: {e}")
            return False

    def fetch_eeg_data(self, event_time, delays, time_window, callback):
        results = []
        delays.sort()
        for delay_ms in delays:
            delay_s = delay_ms / 1000.0
            absolute_event_time = event_time + delay_s
            start_time = absolute_event_time + time_window[0]
            end_time = absolute_event_time + time_window[1]

            collected_samples = []
            while True:
                sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                if timestamp is None:
                    continue
                if timestamp > end_time:
                    break
                if timestamp >= start_time:
                    collected_samples.append((timestamp, sample))
            
            if collected_samples:
                results.append(collected_samples)
                print(f"Collected {len(collected_samples)} samples for delay {delay_ms}ms")
        
        #print(results)

        callback(results)





# need a cors server for serving images in yq
app = Flask(__name__, static_folder='static')
CORS(app)
lsl_handler = LSLHandler()


@app.route('/event', methods=['POST'])
def handle_event():
    print("hallo???")
    data = request.json
    event_timestamp = data['timestamp'] / 1000.0 
    delays = data['onsets'] 


    delays = [delay + offset for delay in delays for offset in (0, 2000)] #also add events for when face disappears (contrast/baseline)

    time_window = np.array([-0.2, 1])  # time window, dont hardcode in the future

    if not lsl_handler.inlet:
        if not lsl_handler.initialize_stream():
            return jsonify({"status": "error", "message": "Failed to connect to EEG stream"}), 500

    threading.Thread(target=lambda: lsl_handler.fetch_eeg_data(event_timestamp, delays, time_window, process_eeg_data)).start()
    print("Event Triggered with:", data)
    return jsonify({"status": "success"}), 200


@app.route('/images/<path:path>')
def serve_image(path):
    return send_from_directory('static', path)

def run_flask_server():
    app.run(host='0.0.0.0', port=5000, use_reloader=False)


def set_data_ready():
    data_ready.set()

def clear_data_ready():
    data_ready.clear()

data_ready = threading.Event()
shared_data = None

if __name__ == "__main__":
    init_unfold()
    threading.Thread(target=run_flask_server, daemon=True).start()
    threading.Thread(target=main_eyetracking_and_streaming, daemon=True).start()
    
    try:
        while True:
            time.sleep(1)

            if data_ready.is_set():
                process_eeg_data_jl()
                clear_data_ready()

    except KeyboardInterrupt:
        print("Shutting down all services.")



