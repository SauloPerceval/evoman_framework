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
from neuroevolution.neural_network import NeuralNetwork


def play_net(net):
    experiment_name = f'./test_neuro/test_{os.getpid()}'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = Environment(experiment_name=experiment_name,
                      multiplemode="no",
                      enemies=[3],
                      playermode="ai",
                      player_controller=NeuroEvoPlayer(),
                      enemymode="static",
                      level=2)

    return env.play(net)


def cross(parents_nets):
    num_childs = 5
    childs = list()

    for i in range(num_childs):
        alpha = numpy.clip(numpy.random.normal(loc=0.5, scale=0.1), a_min=0, a_max=1)

        first_parent = random.choice(parents_nets)
        second_parent = random.choice([net for net in parent_nets if net != first_parent])

        child_weights = numpy.add(first_parent.weights_to_1D_array()*alpha,
                                  second_parent.weights_to_1D_array()*(1-alpha))
        childs.append(NeuralNetwork(child_weights))

    return childs


def mutate(childs_nets):
    mutant_childs = list()
    for child_net in childs_nets:
        child_weights = child_net.weights_to_1D_array()
        mutation_pos = random.randint(0, len(child_weights)-1)
        # child_weights[mutation_pos] = numpy.random.normal(loc=0, scale=0.1)

        mutant_childs.append(NeuralNetwork(child_weights*0.1*numpy.random.normal(loc=0, scale=0.1)))

    return mutant_childs


def tournament(results_n_nets):
    num_next_gen_parents = 5
    next_gen_parents_nets = list()

    for i in range(num_next_gen_parents):
        first_defiant = random.choice(results_n_nets)
        second_defiant = random.choice([result_net for result_net in results_n_nets if result_net != first_defiant])
        # Winner is determine by the first element on the result of the net (the fitness)
        winner = max(first_defiant, second_defiant, key=lambda defiant: defiant[0][0])

        results_n_nets.remove(winner)
        # Add the winner network to the next gen parents
        next_gen_parents_nets.append(winner[1])

    return next_gen_parents_nets


def save_generation(generation_population):
    pandas.DataFrame(
        numpy.array([individual.weights_to_1D_array() for individual in generation_population])
    ).to_csv("generation.csv")


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

    for i in range(num_generation):
        print(f"gen {i}")
        child_nets = cross(parent_nets)

        child_nets = mutate(child_nets)

        save_generation(parent_nets + child_nets)

        generation_result = play_generation(parent_nets + child_nets)

        print([individual_result[0][0] for individual_result in generation_result])

        parent_nets = tournament(generation_result)
