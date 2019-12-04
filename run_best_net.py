import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from neuroevolution.neuroevolutive_player import NeuroEvoPlayer
from neuroevolution.evolution_utils import load_individual


def play_best_net(net):
    env = Environment(multiplemode="yes",
                      enemies=[2, 4, 5, 6],
                      playermode="ai",
                      player_controller=NeuroEvoPlayer(),
                      enemymode="static",
                      savelogs="yes",
                      experiment_name="neuro_result",
                      speed="normal",
                      level=2)
    return env.play(net)


if __name__ == "__main__":
    best_net = load_individual()
    play_best_net(best_net)
