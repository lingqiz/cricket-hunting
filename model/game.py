import pygame
import pathlib, os
import math
from .env import Modulo
from .agent import Agent
PKG_ROOT = pathlib.Path(__file__).parent.resolve()


class ModuloGame():
    def __init__(self):
        # init arena and agent
        self.arena = Modulo()        
        self.agent = Agent(self.arena)

        # init visual 

        # init sound
        pygame.mixer.init()
        file_path = os.path.join(PKG_ROOT, 'chirp.wav')
        self.chirp = pygame.mixer.Sound(file_path)
        self.set_volume()

        # init game
        pygame.init()
        self.screen_size = 2500
        self.screen = pygame.display.set_mode((self.screen_size,
                                               self.screen_size))
        pygame.display.set_caption("Modulo")

        # game variables
        self.circle_radius = 50
        self.circle_color = (50, 150, 245)

    def set_volume(self):
        volume = self.arena.sound_volume(pos=self.agent.get_loc())
        self.chirp.set_volume(volume)

    def play_sound(self):
        self.chirp.play()

    def _draw_mouse(self):
        # Calculate the end point of the radius line
        pos = self.agent.loc
        heading = self.agent.ori
        end_x = pos[0] + self.circle_radius * math.cos(math.radians(heading))
        end_y = pos[1] - self.circle_radius * math.sin(math.radians(heading))

        # Draw open circle and radius line
        pygame.draw.circle(self.screen, self.circle_color,
                           (int(pos[0]), int(pos[1])),
                           self.circle_radius, 4)

        pygame.draw.line(self.screen, self.circle_color,
                         (int(pos[0]), int(pos[1])),
                         (int(end_x), int(end_y)), 4)

    def run_game(self):
        self.running = True
        while self.running:
            # Draw circle
            self._draw_mouse()

            # Update the screen
            pygame.display.flip()

            # Frame rate
            pygame.time.Clock().tick(60)