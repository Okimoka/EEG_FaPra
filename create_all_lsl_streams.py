import logging
logging.basicConfig(level=logging.WARNING)
from multiprocessing import Process, Queue
import time
from flask import jsonify, request
import threading

from eeg_streamer import EEGStreamer
from fixations_streamer import FixationsStreamer
from flask_server import FlaskServer
from surface_blinks_streamer import SurfaceBlinksStreamer
from saccade_amplitude_streamer import SaccadeStreamer
from unfold_analyzer import UnfoldAnalyzer
from fake_eeg_streamer import FakeEEGStreamer


# Turn off logging
for logger in logging.Logger.manager.loggerDict.values():
    if isinstance(logger, logging.Logger):
        logger.setLevel(logging.WARNING)


"""
Directly from the Companion App
- ccs-neon-001_Neon Gaze
- ccs-neon-001_Neon Events

EEG streamer
- UnicornEEG_Filtered
- UnicornEEG_Power

Surface Streamer
- SurfaceGaze_0

Blink Streamer
- blinks

Fixation Streamer
- fixations
"""

DEBUG_MODE = True
PLOT_UNFOLD_LOCALLY = True

# These three are not used, but could allow for sending data from YouQuantified to the Flask Server via POST
def example_post_event_handler():
    print("Event received")

def example_event_function():
    print("Received event")
    data = request.json
    return jsonify({"status": "success"})

def example_init_function():
    print("Server initialized")


data_queue = Queue()
unfold_analyzer = UnfoldAnalyzer(data_queue, PLOT_UNFOLD_LOCALLY)
unfold_analyzer.init_julia()


def saccade_event(amplitude):
    if(unfold_analyzer.connected):
        unfold_analyzer.add_saccade(amplitude)


def start_flask_server(queue):
    server = FlaskServer(example_post_event_handler, example_event_function, example_init_function)
    server.initialize()
    server.set_queue(queue)  # store the queue in your class
    server.start()



if __name__ == "__main__":
    
    flask_process = Process(target=start_flask_server, args=(data_queue,))
    flask_process.start()



    # Requires ccs-neon-001_Neon Gaze
    #Creates Fixations = [binaryFixation]
    fixations_streamer = FixationsStreamer(draw_plot=True)
    fixations_streamer.initialize()
    if fixations_streamer.connected:
        fixations_streamer.start()

    ## Requires device.receive_matched_scene_video_frame_and_gaze()
    #####Creates SurfaceGaze_0  =  [x,y,blinks]
    ##surface_blinks_streamer = SurfaceBlinksStreamer()
    ##surface_blinks_streamer.initialize()
    ##if surface_blinks_streamer.connected:
    ##    surface_blinks_streamer.start()

    #Requires SurfaceGaze_0 and Fixations
    #Creates Saccades = [saccade_amplitude]
    saccade_streamer = SaccadeStreamer(callback=saccade_event, amplitude_method="angle") #angle or surface
    saccade_streamer.initialize()
    if saccade_streamer.connected:
        saccade_streamer.start()
    

    if(DEBUG_MODE):
        #Replaying from xdf
        fake_eeg_streamer = FakeEEGStreamer()
        fake_eeg_streamer.initialize()
        if fake_eeg_streamer.connected:
            fake_eeg_streamer.start()
    else:
        #Using real device
        fake_eeg_streamer = EEGStreamer()
        fake_eeg_streamer.initialize()
        if fake_eeg_streamer.connected:
            fake_eeg_streamer.start()
        print("eeg_streamer started")




    unfold_analyzer.initialize(DEBUG_MODE)
    if unfold_analyzer.connected:
        print("Started")
        unfold_analyzer.start()

    


    try:
        while True:
            current_time = time.time()
            if(unfold_analyzer.data_ready):
                unfold_analyzer.data_ready = False
                print("Data ready")
                #threading.Thread(target=unfold_analyzer.live_fit_and_plot(), daemon=True).start()
                unfold_analyzer.live_fit_and_plot()
                
            time.sleep(0.1)  # Adjust for real-time monitoring
    except KeyboardInterrupt:
        print("Shutting down all services.")





