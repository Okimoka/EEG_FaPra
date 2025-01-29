import random
import cv2
import numpy as np
import pygame
from pygame.locals import *
from pylsl import StreamInlet, resolve_stream, resolve_streams
import sys

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


import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import pylab



class Markers:
    def __init__(self, pixelSizeArucos=20, screen_width=1800, screen_height=900):
        #predefined apriltags from pupil website
        arucos = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        arucos = [np.array(arucos[i], dtype=np.uint8).reshape(8, 8) * 255 for i in range(4)]
        arucos = [np.kron(aruco, np.ones((pixelSizeArucos, pixelSizeArucos))) for aruco in arucos]
        arucos = [np.rot90(aruco, k=1) for aruco in arucos]
        arucos = [np.flipud(aruco) for aruco in arucos]

        #convert marker images to pygame surfaces
        marker_surfaces = [pygame.surfarray.make_surface(np.stack([marker]*3, axis=-1)) for marker in arucos]

        positions = [
            (screen_width - 10*pixelSizeArucos, screen_height - 10*pixelSizeArucos),
            (2*pixelSizeArucos, screen_height - 10*pixelSizeArucos), 
            (screen_width - 10*pixelSizeArucos, 2*pixelSizeArucos),
            (2*pixelSizeArucos, 2*pixelSizeArucos)
        ]

        self.marker_surfaces = marker_surfaces
        self.positions = positions
    
    def draw(self, screen):
        for surf, pos in zip(self.marker_surfaces, self.positions):
            screen.blit(surf, pos)


"""
Eye-Tracking:
LSL "surfaces" stream channels:
0 surface_gaze_x
1 surface_gaze_y
2 surface_confidence
3 surface_on_surf
4 surface_fixations_x
5 surface_fixations_y
6 surface_fixations_confidence
7 surface_fixations_on_surf
8 surface_fixations_duration
9 surface_fixations_dispersion

EEG:
LSL "UnicornEEG_Filtered" stream channels:
0 to 7: EEG channels
"""
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
        
    
    def pull_sample(self, condition=None):
        try:
            sample, timestamp = self.inlet.pull_sample(timeout=0.5)
            if sample:
                if self.track_history_seconds > 0:
                    self.history.append((timestamp, sample, condition))
                    # Remove samples that are older than the specified track_history_seconds
                    while self.history and (timestamp - self.history[0][0]) > self.track_history_seconds:
                        self.history.pop(0)
                return sample
            else:
                #print("Stream is not sending data!") todo
                return (0,0)
        except Exception as e:
            #print(e) Todo
            return (0,0)






class VEPExperiment:
    def __init__(self, game, window, delays, images, start_time):
        self.game = game
        self.window = window
        self.delays = delays
        self.images = images
        self.start_time = start_time
        self.current_img = "checkerboard.png"
        self.current_delay = 0

        self.helper_temp = False #TODO find better solution

        self.init_unfold()

        stream_names = ["UnicornEEG_Filtered"]
        self.streams = {}

        for stream_name in stream_names:
            self.streams[stream_name] = LSLStream(stream_name, window[1]-window[0])


    #def draw_checkerboard(self, size, invert):
    #    # Determine the number of rows and columns based on screen size
    #    rows = self.game.screen.get_height() // size
    #    cols = self.game.screen.get_width() // size
    #    
    #    for row in range(rows):
    #        for col in range(cols):
    #            # Determine the color of the square
    #            if (row + col) % 2 == (1 if invert else 0):
    #                color = (0, 0, 0)  # Black
    #            else:
    #                color = (255, 255, 255)  # White
    #
    #            # Draw the rectangle
    #            rect = (col * size, row * size, size, size)
    #            pygame.draw.rect(self.game.screen, color, rect)


    def draw_image(self, image_path, stretch):
        image = pygame.image.load(image_path)
        screen = self.game.screen
        screen_width, screen_height = screen.get_size()
        
        if stretch:
            image = pygame.transform.scale(image, (screen_width, screen_height))
            screen.blit(image, (0, 0))
        else:
            image_width, image_height = image.get_size()
            
            x = (screen_width - image_width) // 2
            y = (screen_height - image_height) // 2
            
            screen.blit(image, (x, y))
        
        pygame.display.flip()




    def draw(self):

        self.streams["UnicornEEG_Filtered"].pull_sample(self.current_img)
        #print((self.game.current_time, self.start_time, self.game.current_time - self.start_time))
        #print(self.current_delay)

        if (self.game.current_time - self.start_time) >= self.current_delay * 1000:

            if len(self.delays)==1:
                self.delays = []
                self.game.screen.fill((255, 255, 255))
                self.game.selected_minigame = "menu"

            elif not self.helper_temp:
                #self.draw_checkerboard(100, self.inverted)
                self.current_img = self.images.pop(0)
                if(self.current_img != None and (self.current_img.startswith("checkerboard") or self.current_img.startswith("blank"))):
                    self.draw_image(self.current_img, True)
                else:
                    self.draw_image(self.current_img, False)
                pygame.display.flip() 
                self.helper_temp = True

        if (self.game.current_time - self.start_time) >= (self.current_delay * 1000) + self.window[1]*1000:
            if(self.current_delay != 0):
                self.vep_event()
            
            self.helper_temp = False
            self.current_delay = self.delays.pop(0)
            self.start_time = self.game.current_time
            
    def init_unfold(self):
        jlPkg.activate("julia_env")

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


    def draw_plot(self, x, y):
        dpi = 100
        figsize = (4, 4)
        fig = pylab.figure(figsize=figsize, dpi=dpi)
        ax = fig.gca()
        ax.plot(x,y)

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()

        screen_width, screen_height = self.game.screen.get_size()

        size = canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        self.game.screen.blit(surf, (screen_width-dpi*figsize[0],screen_height-dpi*figsize[1]))
        pygame.display.flip()



    def vep_event(self):
        print(str(self.window[1]*1000) + "ms after switch")
        print("VEP event!")
        eeg_data = self.streams["UnicornEEG_Filtered"].history
        srate = 128 #self.streams["UnicornEEG_Filtered"].sampling_rate
        window_size = self.window[1] - self.window[0]
        n_points = int(np.floor(srate * window_size) + 1)+2
        times = np.linspace(self.window[0], self.window[1], n_points)[1:-1]

        #####print(eeg_data)

        latency = [sample[0] for sample in eeg_data]
        data2 = [sample[1][2] for sample in eeg_data]
        data = np.array([[[sample[1][2]/100]*len(times) for sample in eeg_data]]) #use channel 2 for now
        data = data.transpose((0, 2, 1))
        condition = [sample[2] for sample in eeg_data]
        
        #OVERWRITING WITH (correctly shaped) RANDOM DATA
        #Everything very rudimentary, many variable names unfittingly kept from examples
        data2 = np.random.uniform(-1.61, 1.64, (len(eeg_data),))
        data = np.random.uniform(-87, 95, (8,len(times),len(eeg_data)))

        #self.draw_plot(latency,data)
        #compute phase coherence
        #df_string = "DataFrame((condition=String"+str(condition)+",latency=Float64"+str(latency)+"))"


        #
        formula = jl.seval("@formula 0~1+phaseCoh")
        evts_df = jl.DataFrame(face=np.array(condition),phaseCoh=data2)

        #print(evts_df)
        #print("--------------------------------------------------")
        #print(times)
        #print(len(times))
        #print(math.floor(srate * window_size)+1)
        #print("-----------")
        #print(len(data))
        #print(len(latency))
        #print(len(condition))
        #print("-----------")

        #data_np = np.array(data)
        #data.shape == (n_channels, len(times), len(samples))

        m = jl.Unfold.fit(jl.Unfold.UnfoldModel,formula,evts_df,data,times)

        ####len(times) = floor(srate * windowsize)+1

        results_jl = jl.Unfold.coeftable(m)

        results_py = pd.DataFrame({'channel': results_jl.channel,
                                'coefname': results_jl.coefname,
                                'estimate': results_jl.estimate,
                                'time': results_jl.time})

        print(results_py)

        results_ch43 = results_py[results_py.channel == 1]

        #results_jl = Unfold.coeftable(m)
        print("-------------------------RESULT-------------------------")
        print(results_ch43.time)
        print(results_ch43.estimate)

        if(self.current_img.startswith("checkerboard")):
            self.draw_plot(results_ch43.time,results_ch43.estimate)
            #show the plot for 1 second
            pygame.time.delay(1000)
        
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









class Painter:
    def __init__(self, game, markers):
        self.game = game
        self.markers = markers

        self.alpha = 0.2  # Smoothing factor
        self.prev_x = None
        self.prev_y = None
        self.drawing_active = False
        self.gaze_points = []
        self.blink_confidence = 0

        self.last_toggle_time = 0  # Timestamp of the last toggle
        self.last_blink_time = 0   # Timestamp of the last detected blink

        stream_names = ["surfaces", "blinks"]
        self.streams = {}
        for stream_name in stream_names:
            self.streams[stream_name] = LSLStream(stream_name, 0)

    def draw(self):
        self.game.screen.fill((255, 255, 255))
        self.markers.draw(self.game.screen)
        self.draw_gaze_point()
        self.listen_for_blink()
        self.draw_gaze_path()

    def listen_for_blink(self):
        try:
            sample = self.streams['blinks'].pull_sample()
            self.blink_confidence = sample[0]
            if sample and sample[0] >= 0.5:
                current_time = time.time()
                if current_time - self.last_toggle_time > 1:  # Cooldown of 1 second
                    self.toggle_drawing()
                    #sleep
                    self.last_toggle_time = current_time
                    self.gaze_points = self.gaze_points[:-10] #cut off most recent points
        except KeyError:
            print("Blink stream not available.")
            return

    def toggle_drawing(self):
        self.drawing_active = not self.drawing_active
        print("Drawing mode toggled to:", self.drawing_active)
        #if not self.drawing_active:
        #    self.gaze_points = []  # Clear points when stopping drawing

    def draw_gaze_point(self):
        try:
            #test = self.streams['surfaces'].pull_sample()
            #print(test)
            sample = self.streams['surfaces'].pull_sample()
            if sample:
                gaze_x = int(sample[0] * self.game.screen_width)
                gaze_y = int((1 - sample[1]) * self.game.screen_height)
                confidence = sample[2]

                if self.prev_x is not None and self.prev_y is not None:
                    smoothed_x = (1 - self.alpha) * self.prev_x + self.alpha * gaze_x * confidence
                    smoothed_y = (1 - self.alpha) * self.prev_y + self.alpha * gaze_y * confidence
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


                if self.drawing_active and distance_to_last_point < 100:
                    self.gaze_points.append((int(smoothed_x), int(smoothed_y)))

                pygame.draw.circle(self.game.screen, (255, 0, 0), (int(smoothed_x), int(smoothed_y)), 10)

        except KeyError:
            print("Surfaces stream not found.")
            return

    def draw_gaze_path(self):
        if len(self.gaze_points) > 1:
            pygame.draw.lines(self.game.screen, (0, 0, 255), False, self.gaze_points, 2)



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
            "painter": {"name": "Painter", "logo": "logo_draw.png", "state": self.init_painter()},
            "vep": {"name": "Checkerboard", "logo": "logo_vep.png", "state": VEPExperiment(self, [0,0], [], [], 0)}, #init with dummy state, will be overwritten
            "vep2": {"name": "Car/Face", "logo": "logo_car.png", "state": VEPExperiment(self, [0,0], [], [], 0)},
            "vep3": {"name": "Colorblind", "logo": "logo_eye.png", "state": VEPExperiment(self, [0,0], [], [], 0)},
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
    

    def init_vep(self, window, delays, images):
        start_time = pygame.time.get_ticks()
        return VEPExperiment(self, window, delays, images, start_time)


        
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
        

        #checkerboard
        if(minigame == "vep"):
            if len(self.games[minigame]["state"].delays) == 0:
                window = [-0.3, 0.5]
                delays = [round(random.uniform(4, 8)) for _ in range(3)]+[0]
                images = ["checkerboard.png", "checkerboard_inverted.png", "checkerboard.png"]
                self.games[minigame]["state"] = self.init_vep(window, delays, images)

        #car/face
        elif(minigame == "vep2"):
            if len(self.games[minigame]["state"].delays) == 0:
                window = [-0.3, 0.5]
                n = 5
                images = np.random.choice(['face.png', 'car.png'], n)

                delays = np.full(n, 0.5)  #all images shown for 0.5 seconds

                blank_delays = np.random.uniform(2, 5, n)
                blank_images = np.array(['blank.png'] * n)

                final_images = np.empty(n * 2, dtype=object)
                final_delays = np.empty(n * 2, dtype=float)
                final_images[0::2] = images
                final_images[1::2] = blank_images
                final_delays[0::2] = delays
                final_delays[1::2] = blank_delays

                delays = [round(random.uniform(4, 8)) for _ in range(3)]+[0]
                images = ["checkerboard.png", "checkerboard_inverted.png", "checkerboard.png"]
                self.games[minigame]["state"] = self.init_vep(window, [1]+list(final_delays), ["blank.png"]+list(final_images))

        #colorblind
        elif(minigame == "vep3"):
            if len(self.games[minigame]["state"].delays) == 0:
                window = [-0.3, 0.5]
                delays = [0.5]+[100]+[0]
                images = ["blank.png","colorblind.png"]
                self.games[minigame]["state"] = self.init_vep(window, delays, images)

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

