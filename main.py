# This entrypoint file to be used in development. Start by reading README.md
from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS import player
from unittest import main
from matplotlib import pyplot as plt
import numpy as np

n_games = 1000

print("Summary against: Quincy (pred)", play(player, quincy, n_games), "\n")
print("Summary against: Abbey (complex pred)", play(player, abbey, n_games), "\n")
print("Summary against: Kris (cyc)", play(player, kris, n_games), "\n")
print("Summary against: Mrugesh (freq)", play(player, mrugesh, n_games), "\n")

# Uncomment line below to play interactively against a bot:
# play(human, abbey, 20, verbose=True)

# Uncomment line below to play against a bot that plays randomly:
# play(human, random_player, 1000)


# Uncomment line below to run unit tests automatically
# main(module='test_module', exit=False)