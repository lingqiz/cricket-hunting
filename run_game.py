from model.game import ModuloGame
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--screen_size", type=int, default=2560)

args = parser.parse_args()

game = ModuloGame(screen_size=args.screen_size,
                  debug=args.debug)

game.run_game()