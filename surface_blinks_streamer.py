import asyncio
import threading
import queue
import time
import nest_asyncio
import numpy as np

from pupil_labs.realtime_api.simple import discover_one_device
from pupil_labs.real_time_screen_gaze.gaze_mapper import GazeMapper
from pylsl import StreamInfo, StreamOutlet

from blink_detector.blink_detector import blink_detection_pipeline
from blink_detector.helper import stream_images_and_timestamps
from streamer import Streamer


"""
Requires connection to the Neon Companion App
Produces SurfaceGaze_0 LSL stream, which has samples of shape [x,y,blink_count]
x,y are in interval [0,1] and describe the position in relation to the aruco markers on screen
blink_count is the amount of blinks since the stream has started


The implementation closely follows the Readmes of the real-time-screen-gaze and real-time-blink-detection repos:
https://github.com/pupil-labs/real-time-screen-gaze
https://github.com/pupil-labs/real-time-blink-detection/

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
            # Device connection process from the real-time-screen-gaze repo
            # This frequently fails for unknown reasons
            self.device = discover_one_device()
            calibration = self.device.get_calibration()
            self.gaze_mapper = GazeMapper(calibration)
            self.connected = True
            print("Device initialized and connected.")
        except Exception as e:
            print(f"Failed to initialize device: {e}")
            self.connected = False

    def blink_detection_thread(self):
        #from real-time-blink-detection
        left_images, right_images, timestamps = stream_images_and_timestamps(self.device)
        blink_generator = blink_detection_pipeline(left_images, right_images, timestamps)

        while True:
            try:
                next(blink_generator)
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
        # definition of the exact setup of the aruco markers for real-time-screen-gaze
        # Curiously, this still works even when scale, margin and screen size are not exactly accurate
        scale = 24 # size of a single pixel of the aruco marker
        margin = 32 # should be >= scale
        screen_width = 1920
        screen_height = 1080
        image_width = scale * 8  # the aruco markers 8x8 pixels
        image_height = scale * 8

        marker_verts = {
            0: [  # top left
                (margin, margin),
                (margin + image_width, margin),
                (margin + image_width, margin + image_height),
                (margin, margin + image_height),
            ],
            1: [  # top right
                (screen_width - margin - image_width, margin),
                (screen_width - margin, margin),
                (screen_width - margin, margin + image_height),
                (screen_width - margin - image_width, margin + image_height),
            ],
            2: [  # bottom left
                (margin, screen_height - margin - image_height),
                (margin + image_width, screen_height - margin - image_height),
                (margin + image_width, screen_height - margin),
                (margin, screen_height - margin),
            ],
            3: [  # bottom right
                (screen_width - margin - image_width, screen_height - margin - image_height),
                (screen_width - margin, screen_height - margin - image_height),
                (screen_width - margin, screen_height - margin),
                (screen_width - margin - image_width, screen_height - margin),
            ],
        }

        screen_size = (screen_width, screen_height)

        screen_surface = self.gaze_mapper.add_surface(
            marker_verts,
            screen_size
        )

        outlet = StreamOutlet(StreamInfo(f'SurfaceGaze_0', 'Gaze', 3, 20, 'float32', f'surface_gaze_id_0'))

        print("LSL outlets created. Streaming surface gaze data...")

        try:
            # to make the stream more robust, the last valid sample is tracked so it can be pushed when no valid sample is found
            last_valid_sample = [0.5, 0.5, 0]
            while True:
                frame, gaze = self.device.receive_matched_scene_video_frame_and_gaze()
                result = self.gaze_mapper.process_frame(frame, gaze)

                #Check the queue for blink events
                if not self.blink_queue.empty():
                    self.blink_queue.get()  # Remove the event from the queue
                    self.blink_counter += 1

                valid_sample_found = False

                for surface_gaze in result.mapped_gaze.get(screen_surface.uid, []):
                    # check whether within bounds of the aruco markers
                    if 0.0 <= surface_gaze.x <= 1.0 and 0.0 <= surface_gaze.y <= 1.0:
                        last_valid_sample = [surface_gaze.x, surface_gaze.y, self.blink_counter]
                        outlet.push_sample(last_valid_sample, time.time())
                        self.latest_timestamp = time.time()
                        print(f"Gaze at {surface_gaze.x}, {surface_gaze.y}")
                        valid_sample_found = True
                        break # only push the first valid sample, since multiple samples within a single frame are unneccessary
                
                # fallback
                if not valid_sample_found:
                    outlet.push_sample(last_valid_sample, time.time())
                    self.latest_timestamp = time.time()

        except KeyboardInterrupt:
            print("Streaming stopped.")



