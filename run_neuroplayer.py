import numpy
import sys
import os
import time
from multiprocessing import Pool

sys.path.insert(0, 'evoman')
from environment import Environment
from neuroevolution.neuroevolutive_player import NeuroEvoPlayer
from neuroevolution.neural_network import NeuralNetwork


def play_net(net):
    experiment_name = 'test_neuro'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = Environment(experiment_name=experiment_name,
                      multiplemode="yes",
                      enemies=[1, 2, 3, 4],
                      playermode="ai",
                      player_controller=NeuroEvoPlayer(),
                      enemymode="static",
                      level=2)

    return env.play(net)


def cross(parents_nets):
    num_childs = 10

    alpha = numpy.random.normal(loc=0.5, scale=0.1)



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


def tournament(nets_and_results):
    pass


if __name__ == '__main__':
    nets = list(NeuralNetwork() for i in range(10))

    generation_result = play_generation(nets)

