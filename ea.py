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

# ascii_ramp_codes = [
#     36, 64, 66, 37, 56, 38, 87, 77, 35, 42,
#     111, 97, 104, 107, 98, 100, 112, 113, 119, 109,
#     90, 79, 48, 81, 76, 67, 74, 85, 89, 88,
#     122, 99, 118, 117, 110, 120, 114, 106, 102, 116,
#     47, 92, 124, 40, 41, 49, 123, 125, 91, 93,
#     63, 45, 95, 43, 126, 60, 62, 105, 33, 32,
#     108, 73, 59, 58, 44, 34, 94, 39, 96, 41
# ]

# ascii_ramp_codes = [
#     64, 35, 87, 36, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
#     63, 33, 97, 98, 99, 59, 58, 43, 61, 45, 44, 46, 95
# ]

ascii_ramp_codes = [
    64, 37, 35, 42, 43, 61, 45, 58, 46
]

img_file = "helmet3.jpg"

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
        self.bounds = (0, 8)                             # ASCII char set
        self.ascii_dict = build_dict(tile_w, tile_h)                # tile size
        self.target = decompose(image_path, n_n, tile_w, tile_h)   # image nxn size, tile size
        #self.target = np.array(decompose(image_path, 576, 16, 24), dtype=float)
        
    def evaluate(self, ind):
        real_vals = self.ascii_to_real(ind)
        diff = np.abs(self.target - real_vals) # piecewise difference
        return diff.mean() # average entropy
    
    def ascii_to_real(self, int_vec):
        rv = []
        for i in int_vec:
            actual_ascii_char = ascii_ramp_codes[i]
            rv.append(self.ascii_dict[actual_ascii_char])
        return np.array(rv, dtype=float)

    def print_genome(self, genome):
        cols = self.n//self.tile_w
        converted_genes = []
        for gene in genome:
            converted_genes.append(ascii_ramp_codes[int(gene)])
        chars = ''.join(chr(code) for code in converted_genes)

        rows = []
        for i in range(0, len(chars), cols):
            rows.append(chars[i:i + cols])
        return '\n'.join(rows)


##############################
# main
##############################
if __name__ == '__main__':
    problem = ASCIIProblem("helmet3.jpg", 576, 16, 24)

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2
    else:
        generations = 200

    l = len(problem.target)
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

                        ops.UniformCrossover(p_swap=0.2, p_xover=0.9),
                        # Apply randomized mutation
                        mutate_randint(bounds=[problem.bounds]*l,
                                        probability= 0.03),
                        ops.evaluate,
                        ops.pool(size=pop_size),

                        # Some visualization probes so we can watch what happens
                        probe.FitnessPlotProbe(),

                        # Collect diversity metrics along with the standard CSV columns
                        #! would be nice to print every 100th best sample here
                        probe.AttributesCSVProbe(stream=sys.stdout, do_fitness=True, best_only=True)
                    ]
            )
    # If we're not in test-harness mode, block until the user closes the app
    if os.environ.get(test_env_var, False) != 'True':
        plt.show()
        
    plt.close('all')
    best = max(final_pop)
    print(f"Fitness: {best.fitness}")
    # print(problem.print_genome(best.genome))
    with open(f"{img_file}_ascii.txt", "w") as f:
        f.write(problem.print_genome(best.genome))