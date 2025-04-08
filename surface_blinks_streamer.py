from pupil_labs.realtime_api.simple import discover_one_device
from pupil_labs.real_time_screen_gaze.gaze_mapper import GazeMapper
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop
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
import time

from streamer import Streamer


"""
Requires connection to the Neon Companion App
Produces SurfaceGaze_0 LSL stream, which has samples of shape [x,y,blink_count]
x,y are in interval [0,1] and describe the position in relation to the aruco markers on screen
blink_count is the amount of blinks since the stream start

TODO comment this class

"""


class SurfaceBlinksStreamer(Streamer):
    def __init__(self):
        super().__init__()
        self.blink_queue = queue.Queue()
        self.device = None
        self.gaze_mapper = None
        self.blink_counter = 0

    def initialize(self):
        try:
            self.device = discover_one_device()
            calibration = self.device.get_calibration()
            self.gaze_mapper = GazeMapper(calibration)
            self.connected = True
            print("Device initialized and connected.")
        except Exception as e:
            print(f"Failed to initialize device: {e}")
            self.connected = False

    def blink_detection_thread(self):
        left_images, right_images, timestamps = stream_images_and_timestamps(self.device)
        blink_generator = blink_detection_pipeline(left_images, right_images, timestamps)

        while True:
            try:
                blink_event = next(blink_generator)
                self.blink_queue.put(1) #1 == blink
            except StopIteration:
                continue  # no blink


    def start(self):
        if not self.connected:
            print("Device is not connected. Please initialize first.")
            return

        threading.Thread(target=self.blink_detection_thread, daemon=True).start()
        threading.Thread(target=self.main_streaming_loop, daemon=True).start()


    def main_streaming_loop(self):
        # We need to define the exact setup of the aruco markers for realtime api to work

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

        screen_size = (screen_width, screen_height)

        screen_surface = self.gaze_mapper.add_surface(
            marker_verts,
            screen_size
        )

        outlet = StreamOutlet(StreamInfo(f'SurfaceGaze_0', 'Gaze', 3, 200, 'float32', f'surface_gaze_id_0'))

        print("LSL outlets created. Streaming surface gaze data...")

        try:
            while True:
                frame, gaze = self.device.receive_matched_scene_video_frame_and_gaze()
                result = self.gaze_mapper.process_frame(frame, gaze)

                # Check the queue for blink events
                if not self.blink_queue.empty():
                    self.blink_queue.get()  # Remove the event from the queue
                    self.blink_counter += 1

                for surface_gaze in result.mapped_gaze[screen_surface.uid]:
                    outlet.push_sample([surface_gaze.x, surface_gaze.y, self.blink_counter], time.time())
                    self.latest_timestamp = time.time()
                    ##print(f"Gaze at {surface_gaze.x}, {surface_gaze.y}")

        except KeyboardInterrupt:
            print("Streaming stopped.")



