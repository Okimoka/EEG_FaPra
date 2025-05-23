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








    def draw(self):


        frame, gaze = self.device.receive_matched_scene_video_frame_and_gaze()
        result = self.gaze_mapper.process_frame(frame, gaze)

        for surface_gaze in result.mapped_gaze[self.screen_surface.uid]:
            self.newest_sample = [surface_gaze.x, surface_gaze.y]
            print(surface_gaze.x, surface_gaze.y)

        self.game.screen.fill((255, 255, 255))
        self.markers.draw(self.game.screen)
        self.draw_gaze_point()






    def draw_gaze_point(self):
        try:
            #test = self.streams['surfaces'].pull_sample()
            #print(test)SurfaceGaze_0
            ##streamshh = resolve_streams()
            ### Printing information about each stream
            ##for streamm in streamshh:
            ##    print("Stream name:", streamm.name())

            #sample = self.streams['ccs-neon-001_Neon Gaze'].pull_sample()
            #TODO
            #During testing (replaying from xdf), this gap seems to constantly grow
            #Does this also happen during regular recording?
            #print(sample)
            #choose color depending on fixation state
            #print(fixationstate)
            color = (255, 0, 0)

            if self.newest_sample:

                gaze_x = int((1-self.newest_sample[0]) * self.game.screen_width)
                gaze_y = int(self.newest_sample[1] * self.game.screen_height)

                #print(gaze_x, gaze_y)

                if self.prev_x is not None and self.prev_y is not None:
                    smoothed_x = (1 - self.alpha) * self.prev_x + self.alpha * gaze_x
                    smoothed_y = (1 - self.alpha) * self.prev_y + self.alpha * gaze_y
                else:
                    smoothed_x, smoothed_y = gaze_x, gaze_y

                
                self.prev_x, self.prev_y = smoothed_x, smoothed_y


                pygame.draw.circle(self.game.screen, color, (int(smoothed_x), int(smoothed_y)), 10)

        except KeyError as e:
            print("Surfaces stream not found.")
            print(e)
            return



class Game:
    def __init__(self, screen_width=1800, screen_height=900):
        #screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.screen.fill((255, 255, 255)) 
        
        #menu
        self.selected_minigame = "menu"
        self.menu_elements = {}

        #Object to store minigame state
        self.minigame_object = {}

        #misc
        self.current_time = 0


        self.games = {
            "painter": {"name": "Painter", "logo": "logo_draw.png", "state": self.init_painter()}
        }

        #self.games = {
        #    "painter": {"name": "Painter", "logo": "logo_draw.png", "state": self.init_painter()},
        #    "vep": {"name": "VEP Experiment", "logo": "logo_vep.png", "state": {}}
        #}

        #pygame
        pygame.display.set_caption('EEG Eye-Tracking Games')
        pygame.init()
        self.init_menu()


    def init_painter(self):
        markers = Markers(screen_width=self.screen_width, screen_height=self.screen_height)
        return Painter(self, markers)

        
    def init_menu(self):
        #(internal name, display name, logo)

        logo_rects = []
        logo_images = []
        logo_texts = []
        font = pygame.font.Font(None, 36)

        for i, game in enumerate(self.games):
            width, height = 200, 200
            x, y = 100 + i*(width+20), 100

            image = pygame.image.load(self.games[game]["logo"])
            image = pygame.transform.scale(image, (width, height))
            text = font.render(self.games[game]["name"], 1, (0, 0, 0))
            textpos = text.get_rect(centerx=x+width//2, centery=y-10)

            logo_rects.append(pygame.Rect(x, y, width, height))
            logo_images.append(image)
            logo_texts.append((text, textpos))
        
        self.menu_elements = {"logos": logo_images, "rects": logo_rects, "texts": logo_texts, "names": self.games.keys()}


    def draw_menu(self, mouse_clicked):
        mouse_pos = pygame.mouse.get_pos()

        for image, rect, text in zip(self.menu_elements["logos"], self.menu_elements["rects"], self.menu_elements["texts"]):
            x, y = rect.topleft
            self.screen.blit(image, (x, y))
            self.screen.blit(text[0], text[1])

        for image_rect in self.menu_elements["rects"]:
            if image_rect.collidepoint(mouse_pos):
                pygame.draw.rect(self.screen, (255, 0, 0), image_rect, 3)  # Red border, 3 pixels thick

        if mouse_clicked:
            for image_rect, name in zip(self.menu_elements["rects"], self.menu_elements["names"]):
                if image_rect.collidepoint(mouse_pos):
                    self.selected_minigame = name

    def draw_minigame(self, minigame):

        self.games[minigame]["state"].draw()


    def run(self):

        running = True
        #self.current_time = pygame.time.get_ticks()
        
        while running:
            self.current_time = pygame.time.get_ticks()
            mouse_clicked = False

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_clicked = True
                if event.type == pygame.QUIT:
                    running = False
                if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    if(self.selected_minigame == "menu"):
                        running = False
                    else:
                        self.screen.fill((255, 255, 255)) 
                        self.selected_minigame = "menu"

            if(self.selected_minigame == "menu"):
                self.draw_menu(mouse_clicked)
            else:
                self.draw_minigame(self.selected_minigame)

            pygame.display.flip()
            pygame.time.delay(5)
            #pygame.time.delay(100)

        pygame.quit()
        sys.exit()




if __name__ == '__main__':
    game = Game()
    game.run()
