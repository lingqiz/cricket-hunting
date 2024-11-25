import pygame
import pathlib, os
import math
import numpy as np
from .env import Modulo
from .agent import GameAgent
PKG_ROOT = pathlib.Path(__file__).parent.resolve()


class ModuloGame():
    def __init__(self):
        # init arena and agent
        self.arena = Modulo()
        self.agent = GameAgent(self.arena)

        # init visual

        # init sound
        pygame.mixer.init()
        file_path = os.path.join(PKG_ROOT, 'chirp.wav')
        self.chirp = pygame.mixer.Sound(file_path)
        self.set_volume()

        # init game
        pygame.init()
        self.screen_size = 2000
        self.screen = pygame.display.set_mode((self.screen_size,
                                               self.screen_size))
        pygame.display.set_caption("Modulo")

        # display variables
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.circle_color = (50, 150, 245)
        self.circle_radius = 30

        self.text_color = (50, 150, 245)
        self.font = pygame.font.SysFont(None, 36)

    def set_volume(self):
        volume = self.arena.sound_volume(pos=self.agent.get_loc())
        self.chirp.set_volume(volume)

    def play_sound(self):
        self.chirp.play()

    def _key_press(self):
        '''
        Control mapping
        '''
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.close()

        # Key press handling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            self.agent.turn_left()

        if keys[pygame.K_d]:
            self.agent.turn_right()

        if keys[pygame.K_w]:
            self.agent.move_forward()

        if keys[pygame.K_SPACE]:
            self.running = False

    def _debug_text(self):
        hd = np.rad2deg(self.agent.ori)
        x = int(self.agent.loc[0])
        y = int(self.agent.loc[1])
        heading_text = self.font.render(f'Rotated Heading: {hd:.0f}', True, self.text_color)
        location_text = self.font.render(f'Coordinate: ({x:.0f}, {y:.0f})', True, self.text_color)

        self.screen.blit(heading_text, (10, self.screen_size - 30))
        self.screen.blit(location_text, (10, self.screen_size - 60))

    def _draw_mouse(self):
        # flip y-axis for screen coordinates
        pos = np.copy(self.agent.loc)
        pos[1] = self.screen_size - pos[1]

        # flip orientation along y-axis
        heading = self.agent.ori
        end_x = pos[0] + self.circle_radius * math.cos(heading)
        end_y = pos[1] - self.circle_radius * math.sin(heading)

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
            # Key press
            self._key_press()

            # Draw elements
            self.screen.fill(self.white)
            self._draw_mouse()

            # Debug text
            self._debug_text()

            # Update the screen
            pygame.display.flip()

            # Frame rate
            pygame.time.Clock().tick(60)