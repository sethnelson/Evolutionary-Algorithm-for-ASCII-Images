from DecomposeImage import decompose
from InitAsciiArray import build_dict

import os
import sys

from matplotlib import pyplot as plt
import numpy as np

from leap_ec import Representation, test_env_var
from leap_ec import ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.real_rep.problems import LangermannProblem

# 864 tiles from moon image
image_name = 'moon.jpg'
tileLuminescenceValues = decompose(image_name, 576, 16, 24)
tileLuminescenceValuesMean = np.mean(tileLuminescenceValues)
print(tileLuminescenceValuesMean)
# index = 0
# for tv in tileLuminescenceValues:
#     print(f"{index}: {tv}")
#     index += 1


ascii_dict = build_dict(16, 24)
# for char, value in ascii_dict.items():
#     print(f"{char}: {value}")

# Implementation of a custom problem
class ASCIIProblem(LangermannProblem):
    def __init__(self):
        super().__init__(maximize=False)
        self.bounds = (0, 1)
        
    def evaluate(self, ind):
        avgLuminescence = sum(ind) / len(ind)
        fitness = abs(tileLuminescenceValuesMean - avgLuminescence)
        return fitness

##############################
# main
##############################
if __name__ == '__main__':
    # Our fitness function will be the Langermann
    # This is defined over a real-valued space, but
    # we can also use it to evaluate integer-valued genomes.
    problem = ASCIIProblem()

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2
    else:
        generations = 1000

    l = 864
    pop_size = 50
    generational_ea(max_generations=generations,pop_size=pop_size,
                    problem=problem,  # Fitness function

                    # Representation
                    representation=Representation(
                        # Initialize a population of integer-vector genomes
                        initialize=create_real_vector(
                            bounds=[problem.bounds] * l)
                    ),

                    # Operator pipeline
                    pipeline=[
                        ops.tournament_selection(k=2),
                        ops.clone,

                        # Apply Gaussian mutation
                        mutate_gaussian(std=1.5, bounds=[problem.bounds]*l,
                                        expected_num_mutations=1),
                        ops.evaluate,
                        ops.pool(size=pop_size),

                        # Some visualization probes so we can watch what happens
                        probe.CartesianPhenotypePlotProbe(
                            xlim=problem.bounds,
                            ylim=problem.bounds,
                            contours=problem),
                        probe.FitnessPlotProbe(),

                        # Collect diversity metrics along with the standard CSV columns
                        probe.AttributesCSVProbe(stream=sys.stdout, do_fitness=True, do_genome=True, best_only=True)
                    ]
            )
    # If we're not in test-harness mode, block until the user closes the app
    if os.environ.get(test_env_var, False) != 'True':
        plt.show()
        
    plt.close('all')