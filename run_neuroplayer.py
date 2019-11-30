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
    bash_size = 5
    generation_result = list()
    start_time = time.time()
    last_idx = 0
    for idx in range(bash_size, len(nets)+1, bash_size):
        with Pool(bash_size) as p:
            generation_result.extend(list(p.map(play_net, nets[last_idx:idx])))
            last_idx = idx
    finish_time = time.time()
    print(finish_time - start_time)

    nets_result = list(zip(generation_result, nets))
    return nets_result


if __name__ == '__main__':
    num_generation = 20
    initial_population_num = 20

    parent_nets = list(NeuralNetwork() for i in range(initial_population_num))

    # parent_nets = load_generation()

    # first play
    print(f"\ngen 0")
    child_nets = cross(parent_nets, num_childs=initial_population_num)

    child_nets = mutate(child_nets)

    save_generation(parent_nets + child_nets)

    generation_result = play_generation(parent_nets + child_nets)

    print([individual_result[0][0] for individual_result in generation_result])

    parent_nets_w_results = tournament(generation_result, num_next_gen_parents=initial_population_num)

    for i in range(num_generation):
        print(f"\ngen {i+1}")
        child_nets = cross(parent_nets, num_childs=initial_population_num)

        child_nets = mutate(child_nets)

        save_generation(parent_nets + child_nets)

        child_results = play_generation(child_nets)
        generation_result = parent_nets_w_results + child_results

        print([individual_result[0][2] for individual_result in generation_result])

        parent_nets_w_results = tournament(generation_result, num_next_gen_parents=initial_population_num)
