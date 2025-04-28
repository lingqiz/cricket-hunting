import pygame
import pathlib, os
import math
import numpy as np
import time
from .env import ModuloSim
from .agent import GameAgent
from utils.data_loader import TILE_RAD_MM
PKG_ROOT = pathlib.Path(__file__).parent.resolve()


class ModuloGame():
    def __init__(self, screen_size=1080, debug=False):
        # init arena and agent
        self.arena = ModuloSim()
        self.agent = GameAgent(self.arena)

        # init sound
        pygame.mixer.init()
        file_path = os.path.join(PKG_ROOT, 'chirp.wav')
        self.chirp = pygame.mixer.Sound(file_path)
        self.set_volume()

        # init game
        pygame.init()
        self.screen_size = screen_size
        self.window_size = screen_size * 0.15
        self.screen = pygame.display.set_mode((self.window_size,
                                               self.window_size))
        pygame.display.set_caption("Modulo")

        # display variables
        self.ref_size = 2300
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.target_color = (240, 122, 122)
        self.circle_color = (50, 150, 245)
        self.tile_color = (76, 181, 5)
        self.circle_radius = 30 / self.ref_size * self.screen_size
        self.mask_radius = 2.0 * self.circle_radius
        self.text_color = (50, 150, 245)
        self.font = pygame.font.SysFont(None, 18)
        self.score_font = pygame.font.SysFont(None, 24)
        self.tile_rad = TILE_RAD_MM / self.ref_size * self.screen_size

        # game variables
        self.running = False
        self.debug = debug
        self.mask = True

        self.stop_threshold = 1.0
        self.stop_time = time.time()

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

                if event.type == pygame.KEYDOWN \
                    and event.key == pygame.K_v:
                    self.mask = not self.mask

        # Key press handling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            self.stop_time = time.time()
            self.agent.turn_left()

        if keys[pygame.K_d]:
            self.stop_time = time.time()
            self.agent.turn_right()

        if keys[pygame.K_w]:
            self.stop_time = time.time()
            self.agent.move_forward()

        if keys[pygame.K_SPACE]:
            self.running = False

    def _check_stop(self):
        stop_duration = time.time() - self.stop_time
        if stop_duration > self.stop_threshold:
            capture = self.arena.check_capture(self.agent.get_loc())

            if capture:
                self.stop_time = time.time() + 10.0

            else:
                self.set_volume()
                self.play_sound()

                # reset timer plus 1 second ISI
                self.stop_time = time.time() + 1.0

    def _debug_text(self):
        hd = np.rad2deg(self.agent.ori)
        x = int(self.agent.loc[0])
        y = int(self.agent.loc[1])

        heading_text = self.font.render(f'Rotated Heading: {hd:.0f}', True, self.text_color)
        location_text = self.font.render(f'Coordinate: ({x:.0f}, {y:.0f})', True, self.text_color)
        timer_text = self.font.render(f'Time: {time.time() - self.stop_time:.2f}', True, self.text_color)

        self.screen.blit(heading_text, (10, self.window_size - 15))
        self.screen.blit(location_text, (10, self.window_size - 30))
        self.screen.blit(timer_text, (10, self.window_size - 45))

    def _score_text(self):
        score = self.arena.target_index
        score_text = self.score_font.render(f'Capture: {score}', True, self.text_color)
        self.screen.blit(score_text, (self.window_size - 100, 10))

    def _draw_mouse(self):
        # center of the viewport
        pos = np.array([self.window_size,
                        self.window_size]) / 2

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

    def _draw_mask(self):
        if not self.mask:
            return

        pos = np.array([self.window_size,
                        self.window_size]) / 2
        heading = math.degrees(self.agent.ori)

        mask = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        mask.fill((0, 0, 0, 255))  # non-transparent mask

        # Draw half circle mask
        half_circle_radius = self.mask_radius
        num_points = 50  # Number of points to approximate the half circle
        points = [(pos[0], pos[1])]
        for i in range(num_points + 1):
            angle = math.radians(heading - 90 + 180 * (i / num_points))
            x = pos[0] + half_circle_radius * math.cos(angle)
            y = pos[1] - half_circle_radius * math.sin(angle)
            points.append((x, y))

        pygame.draw.polygon(mask, (0, 0, 0, 0), points)
        self.screen.blit(mask, (0, 0))

    def _convert_to_screen(self, loc):
        loc = np.copy(loc) / self.ref_size * self.screen_size
        loc[1] = self.screen_size - loc[1]
        return loc

    def _draw_hex(self, center, scr_origin):
        # coordinates are flipped for screen display
        center = self._convert_to_screen(center)

        # Calculate the vertices of the hexagon
        vertices = [
            (
                center[0] + self.tile_rad * math.cos(math.pi / 3 * i) - scr_origin[0],
                center[1] - self.tile_rad * math.sin(math.pi / 3 * i) - scr_origin[1]
            )
            for i in range(6)]

        # Draw the hexagon
        pygame.draw.polygon(self.screen, self.white, vertices, 0)  # Fill with white
        pygame.draw.polygon(self.screen, self.tile_color, vertices, 2)  # Green border

    def _draw_target(self, center, scr_origin):
        # convert coordinates
        center = self._convert_to_screen(center) - scr_origin

        # draw target
        pygame.draw.circle(self.screen, self.target_color, center, 10)

    def _draw_border(self, scr_origin):
        vertices = []
        for i in range(6):
            vertice = self._convert_to_screen(self.arena.vert_bound[:, i]) - scr_origin
            vertices.append((vertice[0], vertice[1]))

        pygame.draw.polygon(self.screen, self.tile_color, vertices, 4)

    def _draw_arena(self):
        # compute view port coordinate origin
        # flip y-axis for screen coordinates
        mouse_loc = self._convert_to_screen(self.agent.loc)
        scr_orign = mouse_loc - np.array([self.window_size,
                                          self.window_size]).reshape([2, -1]) / 2

        for i in range(self.arena.n_tiles):
            self._draw_hex(self.arena.tiles[:, i], scr_orign.squeeze())

        for i in range(self.arena.n_target):
            self._draw_target(self.arena.target[:, i], scr_orign.squeeze())

        self._draw_border(scr_orign.squeeze())

    def run_game(self):
        self.running = True
        while self.running:
            # Key press
            self._key_press()

            # Draw visual elements
            self.screen.fill(self.white)
            self._draw_arena()
            self._draw_mouse()
            self._draw_mask()

            # Stop mechanism
            # include play sound and check capture
            self._check_stop()

            # Draw text
            self._score_text()
            if self.debug:
                self._debug_text()

            # Update the screen
            pygame.display.flip()

            # Frame rate
            pygame.time.Clock().tick(60)