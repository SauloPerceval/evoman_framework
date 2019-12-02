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


def play_net_training(net):
    env = Environment(multiplemode="yes",
                      enemies=[1, 3, 7, 8],
                      playermode="ai",
                      player_controller=NeuroEvoPlayer(),
                      enemymode="static",
                      savelogs="no",
                      level=2)
    return env.play(net)


def play_net_result(net):
    env = Environment(multiplemode="yes",
                      enemies=[1, 2, 3, 4, 5, 6, 7, 8],
                      playermode="ai",
                      player_controller=NeuroEvoPlayer(),
                      enemymode="static",
                      savelogs="yes",
                      experiment_name="neuro_result",
                      level=2)
    return env.play(net)


def play_generation(nets):
    bash_size = 5
    generation_result = list()
    start_time = time.time()
    last_idx = 0
    for idx in range(bash_size, len(nets)+1, bash_size):
        with Pool(bash_size) as p:
            generation_result.extend(list(p.map(play_net_training, nets[last_idx:idx])))
            last_idx = idx
    finish_time = time.time()
    print(finish_time - start_time)

    nets_result = list(zip(generation_result, nets))
    return nets_result


if __name__ == '__main__':
    num_generation = 2
    initial_population_num = 5

    parent_nets = list(NeuralNetwork() for i in range(initial_population_num))

    # parent_nets = load_generation()

    # first play
    print(f"\ngen 0")
    child_nets = cross(parent_nets, num_childs=initial_population_num)

    child_nets = mutate(child_nets)

    save_nets(parent_nets + child_nets)

    generation_result = play_generation(parent_nets + child_nets)

    print([individual_result[0][2] for individual_result in generation_result])

    for i in range(num_generation):
        parent_nets_w_results = tournament(generation_result, num_next_gen_parents=initial_population_num)

        parent_nets = [parent_net_w_result[1] for parent_net_w_result in parent_nets_w_results]

        print(f"\ngen {i+1}")
        child_nets = cross(parent_nets, num_childs=initial_population_num)

        child_nets = mutate(child_nets)

        save_nets(parent_nets + child_nets)

        child_results = play_generation(child_nets)
        generation_result = parent_nets_w_results + child_results

        print([individual_result[0][2] for individual_result in generation_result])

    best_net = max(generation_result, key=lambda result_n_net: result_n_net[0][2])[1]

    save_nets([best_net], archive_name="best_net")

    final_result = play_net_result(best_net)

    print(final_result)
