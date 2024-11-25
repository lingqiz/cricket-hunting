import pygame
import pathlib, os
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

    def set_volume(self):
        volume = self.arena.sound_volume(pos=self.agent.get_loc())
        self.chirp.set_volume(volume)

    def play_sound(self):
        self.chirp.play()