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
from scipy.spatial import distance

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
    def __init__(self, image_path, n_n, tile_w, tile_h):
        super().__init__(maximize=False)
        self.n = n_n
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.bounds = (32, 126)                             # ASCII char set
        self.ascii_dict_l, self.ascii_dict_s = build_dict(tile_w, tile_h)                # tile size
        self.luminescence, self.structure = decompose(image_path, n_n, tile_w, tile_h)   # image nxn size, tile size
        #self.target = np.array(decompose(image_path, 576, 16, 24), dtype=float)
        
    def evaluate(self, ind):
        # real_vals = self.ascii_to_real(ind)
        # diff_l = np.abs(self.luminescence - real_vals) # piecewise difference
        i = 0
        total_distance = 0
        for key in ind:
            val = self.ascii_dict_s[key]
            total_distance += distance.hamming(val.flatten(), self.structure[i].flatten())
            i += 1
        diff_s = total_distance / len(ind)
        print(diff_s)
        # return diff_l.mean() # average entropy
        return diff_s
    
    def ascii_to_real(self, int_vec):
        rv = []
        for i in int_vec:
            rv.append(self.ascii_dict_l[i])
        return np.array(rv, dtype=float)

    def print_genome(self, genome):
        cols = self.n//self.tile_w
        chars = ''.join(chr(gene) for gene in genome)

        rows = []
        for i in range(0, len(chars), cols):
            rows.append(chars[i:i + cols])
        return '\n'.join(rows)


##############################
# main
##############################
if __name__ == '__main__':
    problem = ASCIIProblem("moon.jpg", 576, 16, 24)

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2
    else:
        generations = 1500

    l = len(problem.luminescence)
    pop_size = 100

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
                        ops.tournament_selection(k=5),
                        ops.clone,

                        ops.UniformCrossover(p_xover=0.8),
                        # Apply randomized mutation
                        mutate_randint(bounds=[problem.bounds]*l,
                                        probability= 0.01),
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