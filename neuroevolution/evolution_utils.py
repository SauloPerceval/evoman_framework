import numpy
import random

from neuroevolution.neural_network import NeuralNetwork


def cross(parents_nets):
    num_childs = 5
    childs = list()

    for i in range(num_childs):
        alpha = numpy.clip(numpy.random.normal(loc=0.5, scale=0.1), a_min=0, a_max=1)

        first_parent = random.choice(parents_nets)
        second_parent = random.choice([net for net in parents_nets if net != first_parent])

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

        mutant_childs.append(NeuralNetwork(child_weights+0.2*numpy.random.normal(loc=0, scale=0.1)))

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
    generation_weights = [individual.weights_to_1D_array() for individual in generation_population]
    numpy.savetxt("generation.csv", generation_weights, delimiter=",")


def load_generation():
    generation = numpy.loadtxt("generation.csv", delimiter=",")
    generation_nets = [NeuralNetwork(individual) for individual in generation]

    return generation_nets
