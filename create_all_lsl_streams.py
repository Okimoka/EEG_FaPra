import logging
logging.basicConfig(level=logging.WARNING)
from multiprocessing import Process, Queue
import time
from flask import jsonify, request
import argparse

from eeg_streamer import EEGStreamer
from fixations_streamer import FixationsStreamer
from flask_server import FlaskServer
from saccade_amplitude_streamer import SaccadeStreamer
from unfold_analyzer import UnfoldAnalyzer
from surface_blinks_streamer import SurfaceBlinksStreamer


# Turn off all loggers
for logger in logging.Logger.manager.loggerDict.values():
    if isinstance(logger, logging.Logger):
        logger.setLevel(logging.WARNING)


parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=bool, default=False, help='When replaying from .xdf')
parser.add_argument('--plot-gaze', type=bool, default=True, help='Live plot of the gaze angles and fixation state')
# If this is on, the Unfold.jl plot will be drawn locally using matplotlib. If not, it will be sent via websocket to YouQuantified
parser.add_argument('--plot-unfold', type=bool, default=True, help='Live plot of the Unfold.jl model results')
#parser.add_argument('--plot-eeg', type=bool, default=False, help='Live plot of the eeg channels') TODO
args = parser.parse_args()


if __name__ == "__main__":

    # Initialization of the UnfoldAnalyzer class
    # Unfold results queue is a shared queue allowing communication between the UnfoldAnalyzer class (pushes results of Unfold.jl into queue)
    # and the Flask server (emits these results into a websocket)
    unfold_results_queue = Queue()
    unfold_analyzer = UnfoldAnalyzer(unfold_results_queue, args.plot_unfold)
    # init_julia has to be the first function of main, as it causes some lag that might impact the timings of other streams
    # it also needs to be called from the main thread, as otherwise the import of modules will fail
    unfold_analyzer.init_julia() 


    def start_flask_server(queue):
        server = FlaskServer()
        server.initialize()
        server.set_queue(queue)
        server.start()
    
    # Flask server needs a separate process in order to be non-blocking
    flask_process = Process(target=start_flask_server, args=(unfold_results_queue,))
    flask_process.start()

    fixations_streamer = FixationsStreamer(draw_plot=args.plot_gaze)
    fixations_streamer.initialize()
    if fixations_streamer.connected:
        fixations_streamer.start()

    # SurfaceBlinksStreamer cannot be simulated using an .xdf, as it requires a connection to the actual device
    if(not args.debug):
        surface_blinks_streamer = SurfaceBlinksStreamer()
        surface_blinks_streamer.initialize()
        if surface_blinks_streamer.connected:
            surface_blinks_streamer.start()

    saccade_streamer = SaccadeStreamer(callback_object=unfold_analyzer, amplitude_method="angle") #angle or surface
    saccade_streamer.initialize()
    if saccade_streamer.connected:
        saccade_streamer.start()
    

    eeg_streamer = EEGStreamer()
    eeg_streamer.initialize()
    if eeg_streamer.connected:
        eeg_streamer.start()


    unfold_analyzer.initialize()
    if unfold_analyzer.connected:
        unfold_analyzer.start()

    # juliacall functions have to be called from the main loop
    try:
        while True:
            if(unfold_analyzer.data_ready):
                unfold_analyzer.data_ready = False
                # Putting this in a thread unfortunately does not help removing the long delay for the first trial. Subprocess probably works
                unfold_analyzer.live_fit_and_plot() 
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down all services.")
