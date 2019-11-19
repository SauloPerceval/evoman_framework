import numpy
import pandas

import random
import sys
import os
import time
from multiprocessing import Pool

sys.path.insert(0, 'evoman')
from environment import Environment
from neuroevolution.neuroevolutive_player import NeuroEvoPlayer
from neuroevolution.evolution_utils import *


def play_net(net):
    env = Environment(multiplemode="yes",
                      enemies=[1,2,3,4],
                      playermode="ai",
                      player_controller=NeuroEvoPlayer(),
                      enemymode="static",
                      savelogs="no",
                      level=2)
    return env.play(net)


def play_generation(nets):
    start_time = time.time()
    with Pool(5) as p:
        generation_result = list(p.map(play_net, nets[:5]))
    with Pool(5) as p:
        generation_result.extend(list(p.map(play_net, nets[5:])))
    finish_time = time.time()
    print(finish_time - start_time)

    nets_result = list(zip(generation_result, nets))
    return nets_result


if __name__ == '__main__':
    num_generation = 20

    parent_nets = list(NeuralNetwork() for i in range(5))

    # parent_nets = load_generation()

    for i in range(num_generation):
        print(f"gen {i}")
        child_nets = cross(parent_nets)

        child_nets = mutate(child_nets)

        save_generation(parent_nets + child_nets)

        generation_result = play_generation(parent_nets + child_nets)

        print([individual_result[0][0] for individual_result in generation_result])

        parent_nets = tournament(generation_result)
