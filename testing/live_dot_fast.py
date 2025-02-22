import random
import cv2
import numpy as np
import pygame
from pygame.locals import *
from pylsl import StreamInlet, resolve_byprop, resolve_streams
import sys
import time
import threading
import nest_asyncio
import asyncio
from pupil_labs.realtime_api.simple import discover_one_device, discover_devices
from pupil_labs.real_time_screen_gaze.gaze_mapper import GazeMapper

class Markers:
    def __init__(self, pixelSizeArucos=24, border=32, screen_width=1800, screen_height=900):
        #predefined apriltags from pupil website
        arucos = [[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,1,1,1,0,1,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,1,0,0,1,0,0,1,0,1,0,0,1,1,1,0,0,1,0,0,1,1,1,0,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0]]
        arucos = [np.array(arucos[i], dtype=np.uint8).reshape(8, 8) * 255 for i in range(4)]
        arucos = [np.kron(aruco, np.ones((pixelSizeArucos, pixelSizeArucos))) for aruco in arucos]
        arucos = [np.rot90(aruco, k=1) for aruco in arucos]
        arucos = [np.flipud(aruco) for aruco in arucos]

        #convert marker images to pygame surfaces
        marker_surfaces = [pygame.surfarray.make_surface(np.stack([marker]*3, axis=-1)) for marker in arucos]

        positions = [
            (screen_width-pixelSizeArucos*8-border, screen_height-pixelSizeArucos*8-border),
            (border, screen_height-pixelSizeArucos*8-border), 
            (screen_width-pixelSizeArucos*8-border, border),
            (border, border)
        ]

        self.marker_surfaces = marker_surfaces
        self.positions = positions
    
    def draw(self, screen):
        for surf, pos in zip(self.marker_surfaces, self.positions):
            screen.blit(surf, pos)



class Painter:
    def __init__(self, game, markers):
        self.game = game
        self.markers = markers

        self.alpha = 0.2  # Smoothing factor
        self.prev_x = None
        self.prev_y = None
  
        self.newest_sample = []



        nest_asyncio.apply()
        self.device = discover_one_device()
        calibration = self.device.get_calibration()
        self.gaze_mapper = GazeMapper(calibration)

        scale = 24
        margin = 32
        screen_width = 1800
        screen_height = 900
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

        screen_size = (1800, 900)

        self.screen_surface = self.gaze_mapper.add_surface(
            marker_verts,
            screen_size
        )

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.start_async_loop, daemon=True)
        self.thread.start()

        # Start async gaze processing
        asyncio.run_coroutine_threadsafe(self.update_gaze_data(), self.loop)


    def start_async_loop(self):
        """Runs the asyncio event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def update_gaze_data(self):
        """Asynchronously fetch gaze data and process it."""
        while True:
            frame, gaze = await asyncio.to_thread(self.device.receive_matched_scene_video_frame_and_gaze)
            result = await asyncio.to_thread(self.gaze_mapper.process_frame, frame, gaze)

            for surface_gaze in result.mapped_gaze[self.screen_surface.uid]:
                self.newest_sample = [surface_gaze.x, surface_gaze.y]
            
            await asyncio.sleep(0.001)  # Yield control briefly to prevent blocking


    def draw(self):

        self.game.screen.fill((255, 255, 255))
        self.markers.draw(self.game.screen)
        self.draw_gaze_point()



    def draw_gaze_point(self):
        try:
            if self.newest_sample:

                gaze_x = int((1-self.newest_sample[0]) * self.game.screen_width)
                gaze_y = int(self.newest_sample[1] * self.game.screen_height)

                pygame.draw.circle(self.game.screen, (255, 0, 0), (int(gaze_x), int(gaze_y)), 10)

        except KeyError as e:
            print("Surfaces stream not found.")
            print(e)
            return



class Game:
    def __init__(self, screen_width=1800, screen_height=900):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.screen.fill((255, 255, 255))
        pygame.display.set_caption('EEG Eye-Tracking Painter')
        markers = Markers(screen_width=screen_width, screen_height=screen_height)
        self.painter = Painter(self, markers)


    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            self.painter.draw()
            pygame.display.update()  # More efficient than flip()
            clock.tick(60)  # Maintain consistent FPS
        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    game = Game()
    game.run()
