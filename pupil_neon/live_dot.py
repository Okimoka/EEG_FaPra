import random
import cv2
import numpy as np
import pygame
from pygame.locals import *
from pylsl import StreamInlet, resolve_stream, resolve_streams
import sys
import time
import threading




class Markers:
    def __init__(self, pixelSizeArucos=24, border=32, screen_width=1920, screen_height=1080):
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


class LSLStream:
    def __init__(self, name, track_history_seconds=0):

        self.streams = None
        self.inlet = None
        self.sampling_rate = None
        self.max_history_length = None
        self.history = []
        self.track_history_seconds = track_history_seconds

        all_streams = resolve_streams()
        if(name in [stream.name() for stream in all_streams]):
            self.streams = resolve_stream('name', name)
            self.inlet = StreamInlet(self.streams[0])
            self.sampling_rate = self.inlet.info().nominal_srate()
            self.max_history_length = int(self.sampling_rate * self.track_history_seconds)
            print("Connected to stream: " + name)
        else:
            print("Stream " + name + " not found.")
        
    
    def pull_sample(self):
        try:
            sample, timestamp = self.inlet.pull_sample(timeout=0.5)
            if sample:
                if self.track_history_seconds > 0:
                    self.history.append((timestamp, sample))
                    # Remove samples that are older than the specified track_history_seconds
                    while self.history and (timestamp - self.history[0][0]) > self.track_history_seconds:
                        self.history.pop(0)
                return timestamp, sample
            else:
                #print("Stream is not sending data!") todo
                return (0,0)
        except Exception as e:
            #print(e) Todo
            return (0,0)


class Painter:
    def __init__(self, game, markers):
        self.game = game
        self.markers = markers

        self.alpha = 0.2  # Smoothing factor
        self.prev_x = None
        self.prev_y = None
        self.gaze_points = []

        #stream_names = ["ccs-neon-001_Neon Gaze", "Fixations"]
        stream_names = ["SurfaceGaze_0", "Fixations"]
        
        self.streams = {}
        for stream_name in stream_names:
            self.streams[stream_name] = LSLStream(stream_name, 0)

    def draw(self):
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
            fixationstate = self.streams['Fixations'].pull_sample()[1]
            sample = self.streams['SurfaceGaze_0'].pull_sample()[1]
            print(sample)
            #choose color depending on fixation state
            print(fixationstate)
            color = (0, 255, 0) if fixationstate[0] == 1 else (255, 0, 0)

            if sample:
                #gaze_x = sample[0]
                #gaze_y = sample[1]
                
                gaze_x = int(sample[0] * self.game.screen_width)
                gaze_y = int((1 - sample[1]) * self.game.screen_height)
                print(gaze_x, gaze_y)

                if self.prev_x is not None and self.prev_y is not None:
                    smoothed_x = (1 - self.alpha) * self.prev_x + self.alpha * gaze_x
                    smoothed_y = (1 - self.alpha) * self.prev_y + self.alpha * gaze_y
                else:
                    smoothed_x, smoothed_y = gaze_x, gaze_y

                distance_to_last_point = 0
                try:
                    distance_to_last_point = ((smoothed_x - self.prev_x) ** 2 + (smoothed_y - self.prev_y) ** 2) ** 0.5
                    #print(distance_to_last_point)
                except:
                    pass
                    #print("Error")
                
                self.prev_x, self.prev_y = smoothed_x, smoothed_y


                
                self.gaze_points.append((int(smoothed_x), int(smoothed_y)))

                pygame.draw.circle(self.game.screen, color, (int(smoothed_x), int(smoothed_y)), 10)

        except KeyError as e:
            print("Surfaces stream not found.")
            print(e)
            return



class Game:
    def __init__(self, screen_width=1920, screen_height=1080):
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
            #pygame.time.delay(100)

        pygame.quit()
        sys.exit()




if __name__ == '__main__':
    game = Game()
    game.run()
