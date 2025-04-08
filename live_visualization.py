import multiprocessing
import threading
import time

from matplotlib.animation import FuncAnimation
from lsl_stream import LSLStream
from helpers import DataBuffer
import matplotlib.pyplot as plt

import pandas as pd

class Plotter:

    # Again, for allowing communication with the subprocess
    def _handle_queue_updates(self):
        while True:
            if(type(self.data_buffer) == DataBuffer):
                if self.data_buffer:
                    self.queue.put(self.data_buffer)
                    time.sleep(0.05)
            elif isinstance(self.data_buffer, pd.DataFrame):
                if not self.data_buffer.empty:
                    self.queue.put(self.data_buffer)
                    time.sleep(0.05)

    # Using matplotlibs "FuncAnimation" to produce a real-time plot.
    # The construct using queues is needed because this needs to run in a subprocess in order to be non-blocking (thread does not work)
    def run_plotting(self, queue):
        def update_plot(frame):
            while not queue.empty():
                data_buffer = queue.get()
                ax.clear()

                if(type(data_buffer) == DataBuffer):
                    ax.plot(data_buffer.gaze_angles_timestamps, data_buffer.gaze_angles_buffer, label='Data Line')
                    ax.plot(data_buffer.fixations_timestamps, data_buffer.fixations_buffer, label='Fixation')
                    ax.legend(loc='upper left')
                    plt.xlabel('Time')
                    plt.ylabel('Gaze angle')
                    plt.title('Saccade detection using idt')
                
                elif isinstance(data_buffer, pd.DataFrame):
                    x_points1 = data_buffer[(data_buffer['coefname'] == '(Intercept)')]["time"].tolist()
                    x_points2 = data_buffer[(data_buffer['coefname'] == 'saccade_amplitude')]["time"].tolist()
                    y_points1 = data_buffer[(data_buffer['coefname'] == '(Intercept)')]["estimate"].tolist()
                    y_points2 = data_buffer[(data_buffer['coefname'] == 'saccade_amplitude')]["estimate"].tolist()

                    ax.plot(x_points1, y_points1, label='(Intercept)')
                    ax.plot(x_points2, y_points2, label='saccade_amplitude')
                    ax.legend(loc='upper left')
                    plt.xlabel('Time')
                    plt.ylabel('estimate')
                    plt.title('Unfold')

        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, update_plot, interval=20)
        plt.show()

    def start(self, **kwargs):
        if self.draw_plot:
            print("Starting plot process")
            self.plot_process = multiprocessing.Process(target=self.run_plotting, args=(self.queue,))
            self.plot_process.start()
            self.queue_thread = threading.Thread(target=self._handle_queue_updates)
            self.queue_thread.daemon = True
            self.queue_thread.start()


    def __init__(self, draw_plot=False, data_buffer=None, **kwargs):
        self.draw_plot = draw_plot
        # These two are just for plotting
        self.queue = multiprocessing.Queue()
        self.data_buffer = data_buffer





