from DecomposeImage import decompose
from InitAsciiArray import build_dict

import os
import sys

from matplotlib import pyplot as plt
import numpy as np

from leap_ec import Representation, test_env_var
from leap_ec.algorithm import generational_ea
from leap_ec.int_rep import create_int_vector
from leap_ec.int_rep.ops import mutate_randint
from leap_ec.decoder import IdentityDecoder
from leap_ec import Individual
from leap_ec.problem import ScalarProblem
from leap_ec import ops, probe

# 864 tiles from moon image
# image_name = 'moon.jpg'
# tileLuminescenceValues = decompose(image_name, 576, 16, 24)
# tileLuminescenceValuesMean = np.mean(tileLuminescenceValues)
# print(tileLuminescenceValuesMean)
# index = 0
# for tv in tileLuminescenceValues:
#     print(f"{index}: {tv}")
#     index += 1

#ascii_dict = build_dict(16, 24)
# for char, value in ascii_dict.items():
#     print(f"{char}: {value}")

# Implementation of a custom problem
class ASCIIProblem(ScalarProblem):
    def __init__(self, image_path):
        super().__init__(maximize=False)
        self.bounds = (32, 126)
        self.ascii_dict = build_dict(16, 24)
        self.target = decompose(image_path, 576, 16, 24)
        #self.target = np.array(decompose(image_path, 576, 16, 24), dtype=float)
        
    def evaluate(self, ind):
        real_vals = self.ascii_to_real(ind)
        diff = np.abs(self.target - real_vals) # piecewise difference
        return diff.mean() # average entropy
    
    def ascii_to_real(self, int_vec):
        rv = []
        for i in int_vec:
            rv.append(self.ascii_dict[i])
        return np.array(rv, dtype=float)

    def print_genome(self, genome):
        cols = 576//16
        chars = ''.join(chr(gene) for gene in genome)

        rows = []
        for i in range(0, len(chars), cols):
            rows.append(chars[i:i + cols])
        return '\n'.join(rows)


##############################
# main
##############################
if __name__ == '__main__':
    # Our fitness function will be the Langermann
    # This is defined over a real-valued space, but
    # we can also use it to evaluate integer-valued genomes.
    problem = ASCIIProblem("moon.jpg")

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2
    else:
        generations = 400

    l = len(problem.target)
    pop_size = 50

    final_pop = generational_ea(
                    max_generations=generations,
                    pop_size=pop_size,
                    problem=problem,  # Fitness function

                    # Representation
                    representation=Representation(
                        # Initialize a population of integer-vector genomes
                        initialize=create_int_vector([problem.bounds] * l),
                        decoder=IdentityDecoder()
                    ),

                    # Operator pipeline
                    pipeline=[
                        ops.tournament_selection(k=2),
                        ops.clone,

                        ops.UniformCrossover(p_swap=0.2, p_xover=0.9),
                        # Apply randomized mutation
                        mutate_randint(bounds=[problem.bounds]*l,
                                        probability= 0.005),
                        ops.evaluate,
                        ops.pool(size=pop_size),

                        # Some visualization probes so we can watch what happens
                        probe.FitnessPlotProbe(),

                        # Collect diversity metrics along with the standard CSV columns
                        #! would be nice to print every 100th best sample here
                        #probe.AttributesCSVProbe(stream=sys.stdout, do_fitness=True, do_genome=True, best_only=True)
                    ]
            )
    # If we're not in test-harness mode, block until the user closes the app
    if os.environ.get(test_env_var, False) != 'True':
        plt.show()
        
    plt.close('all')
    best = max(final_pop)
    print(f"Fitness: {best.fitness}")
    print(problem.print_genome(best.genome))