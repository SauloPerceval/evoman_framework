import numpy
import random

from neuroevolution.neural_network import NeuralNetwork

mutation_percent = 0.5


def cross(parents_nets, num_childs):
    childs = list()

    for i in range(num_childs):
        first_parent = random.choice(parents_nets)
        second_parent = random.choice([net for net in parents_nets if net != first_parent])

        alpha = numpy.clip(
            numpy.random.normal(loc=0.5, scale=0.1, size=first_parent.weights_to_1D_array().size), a_min=0, a_max=1
        )

        child_weights = numpy.add(first_parent.weights_to_1D_array()*alpha,
                                  second_parent.weights_to_1D_array()*(1-alpha))
        childs.append(NeuralNetwork(child_weights))

    return childs


def mutate(childs_nets):
    childs_to_mutate = random.sample(childs_nets, k=round(len(childs_nets)*mutation_percent))
    [childs_nets.remove(child_to_mutate) for child_to_mutate in childs_to_mutate]
    for child_net_to_mutate in childs_to_mutate:
        child_weights = child_net_to_mutate.weights_to_1D_array()

        childs_nets.append(
            NeuralNetwork(child_weights+numpy.random.normal(loc=0, scale=0.1, size=child_weights.size))
        )

    return childs_nets


def tournament(results_n_nets, num_next_gen_parents):
    next_gen_parents_w_result = list()

    for i in range(num_next_gen_parents):
        first_defiant = random.choice(results_n_nets)
        second_defiant = random.choice([result_net for result_net in results_n_nets if result_net != first_defiant])
        # Winner is determine by the first element on the result of the net (the fitness)
        winner = min(first_defiant, second_defiant, key=lambda defiant: defiant[0][2])

        results_n_nets.remove(winner)
        # Add the winner network to the next gen parents
        next_gen_parents_w_result.append(winner)

    return next_gen_parents_w_result


def save_nets(generation_population, archive_name="generation"):
    generation_weights = [individual.weights_to_1D_array() for individual in generation_population]
    numpy.savetxt(f"{archive_name}.csv", generation_weights, delimiter=",")


def load_generation():
    generation = numpy.loadtxt("generation.csv", delimiter=",")
    generation_nets = [NeuralNetwork(individual) for individual in generation]

    return generation_nets
