from controller import Controller
from neuroevolution.neural_network import NeuralNetwork


class NeuroEvoPlayer(Controller):

    def control(self, sensors, net: NeuralNetwork):
        output = net.run_network(sensors)

        bin_output = map(lambda x: 1 if x > 0.5 else 0, output)

        left, right, jump, release = bin_output

        shoot = 1

        return [left, right, jump, shoot, release]
