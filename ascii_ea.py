from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import random
import os
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.measure import block_reduce
from skimage.filters import sobel

from leap_ec import Representation, ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.problem import ScalarProblem

IMAGE_NAME = "images/moon.jpg"
FONT_PATH = "consola.ttf"
FONT_SIZE = 10
MAX_SIZE = 500

MAX_GENERATIONS = 2000
POP_SIZE        = 50
PRINT_EVERY_N   = 50

MSE_WEIGHT = 0.2
SSIM_WEIGHT = 0.4
REGION_WEIGHT = 0.4
assert abs(MSE_WEIGHT + SSIM_WEIGHT + REGION_WEIGHT - 1.0) < 1e-9, \
    "Fitness weights must sum to 1.0"

REGION_POOL_TILES = 5

CHARS = (
    ' .\'`^",:;Il!i><~+_-?][}{1)(|\\/'
    'tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*'
    '#MW&8%B@$'
)

def load_and_resize(path, max_size):
    img = ImageOps.grayscale(Image.open(path))
    img.thumbnail((max_size, max_size))
    return img

def get_bg_fg(image_array):
    corners = [image_array[0, 0], image_array[0, -1], image_array[-1, 0], image_array[-1, -1]]
    if np.mean(corners) > 128:
        return 255, 0    #Light background, dark text
    else:
        return 0, 255    #Dark background, light text

def get_tile_size(font):
    ascent, descent = font.getmetrics()
    tile_h = ascent + descent
    tile_w = int(font.getlength('M'))
    return tile_w, tile_h

def crop_to_grid(image_array, tile_w, tile_h):
    h, w = image_array.shape
    h = (h // tile_h) * tile_h
    w = (w // tile_w) * tile_w
    return image_array[:h, :w]

def build_char_templates(font, tile_w, tile_h, bg, fg):
    """Pre-render every character as a numpy array."""
    templates = {}
    for char in CHARS:
        img = Image.new('L', (tile_w, tile_h), bg)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, fg, font=font)
        templates[char] = np.array(img, dtype=np.float32)
    return templates

def greedy_match(image_array, templates, tile_w, tile_h):
    """For every tile, pick the character with the lowest MSE."""
    h, w = image_array.shape
    rows = h // tile_h
    cols = w // tile_w

    char_list = list(templates.keys())
    template_stack = np.stack([templates[c] for c in char_list], axis=0)

    result = []
    for row in range(rows):
        row_chars = []
        for col in range(cols):
            r0, r1 = row * tile_h, (row + 1) * tile_h
            c0, c1 = col * tile_w, (col + 1) * tile_w
            tile = image_array[r0:r1, c0:c1].astype(np.float32)
            diff = template_stack - tile[np.newaxis, :, :]
            mse = np.mean(diff ** 2, axis=(1, 2))
            row_chars.append(char_list[np.argmin(mse)])
        result.append(row_chars)
        if row % 10 == 0:
            print(f"  Greedy row {row}/{rows}")

    return result  #list[list[char]]

def render_grid(char_grid, font, tile_w, tile_h, bg, fg):
    """Render a 2D character grid into a PIL image."""
    rows = len(char_grid)
    cols = len(char_grid[0])
    img = Image.new('L', (cols * tile_w, rows * tile_h), bg)
    draw = ImageDraw.Draw(img)
    for row_idx, row in enumerate(char_grid):
        for col_idx, char in enumerate(row):
            draw.text((col_idx * tile_w, row_idx * tile_h), char, fg, font=font)
    return img

class EvalData:
    def __init__(self, target_image, font, tile_w, tile_h, cols, bg, fg):
        self.target_image = target_image
        self.font = font
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.cols = cols
        self.bg = bg
        self.fg = fg

def genome_to_image_array(genome, eval_data):
    ed = eval_data
    cols = ed.cols
    rows = len(genome) // cols
    img = Image.new('L', (cols * ed.tile_w, rows * ed.tile_h), ed.bg)
    draw = ImageDraw.Draw(img)
    for row_idx in range(rows):
        for col_idx in range(cols):
            char = CHARS[int(genome[row_idx * cols + col_idx]) % len(CHARS)]
            draw.text((col_idx * ed.tile_w, row_idx * ed.tile_h), char, ed.fg, font=ed.font)
    return np.array(img)

def edge_continuity_loss(rendered, target):
    """
    Compare Sobel edge maps of the rendered and target images.
    Rewards the EA for aligning edges *across* tile boundaries —
    something greedy cannot see because it evaluates each tile alone.
    Normalised to [0, 1] by the worst-case gradient magnitude.
    """

    r_edges = sobel(rendered.astype(np.float32))
    t_edges = sobel(target.astype(np.float32))
    
    #Sobel on uint8 has a max per-pixel value of ~1440 (255 * 4*sqrt(2)).
    MAX_SOBEL_MSE = 1440.0 ** 2
    return np.mean((r_edges - t_edges) ** 2) / MAX_SOBEL_MSE

def region_brightness_loss(rendered, target, tile_w, tile_h, pool_tiles):
    """
    Downsample both images into coarse blocks of (pool_tiles * tile) pixels
    and compare average brightness in each block.
    Rewards the EA for correcting regions that are collectively too bright
    or too dark — a global error that tile-level MSE never penalises.
    Normalised to [0, 1].
    """
    block_h = tile_h * pool_tiles
    block_w = tile_w * pool_tiles
    
    #Trim to exact multiple so block_reduce doesn't pad.
    h, w = rendered.shape
    h_trim = (h // block_h) * block_h
    w_trim = (w // block_w) * block_w
    r_pool = block_reduce(rendered[:h_trim, :w_trim], (block_h, block_w), np.mean)
    t_pool = block_reduce(target[:h_trim, :w_trim], (block_h, block_w), np.mean)
    return np.mean((r_pool - t_pool) ** 2) / (255.0 ** 2)

class ASCIIArtProblem(ScalarProblem):
    """
    Fitness = weighted blend of three terms, each targeting a different scale:

      MSE_WEIGHT    — per-pixel accuracy       (local,  what greedy optimises)
      SSIM_WEIGHT   — edge / structural detail  (mid,    across tile boundaries)
      REGION_WEIGHT — regional brightness       (global, NxN tile block averages)

    All three are normalised to [0, 1] so the weights are interpretable.
    Keep MSE_WEIGHT low: greedy already handles local accuracy well.
    The EA's job is to improve the mid- and global-scale terms.
    """

    def __init__(self, eval_data):
        super().__init__(maximize=False)
        self.eval_data = eval_data

    def evaluate(self, genome):
        rendered = genome_to_image_array(genome, self.eval_data)
        target = self.eval_data.target_image
        ed = self.eval_data

        #Local: per-pixel accuracy.
        mse = (np.mean((rendered.astype(np.float32) - target.astype(np.float32)) ** 2) / (255.0 ** 2))

        #Mid: structural coherence across tile boundaries (via SSIM).
        ssim_loss = (1.0 - ssim(rendered, target, data_range=255)) / 2.0

        #Global: region-level brightness matching.
        region = region_brightness_loss(rendered, target, ed.tile_w, ed.tile_h, REGION_POOL_TILES)

        return MSE_WEIGHT * mse + SSIM_WEIGHT * ssim_loss + REGION_WEIGHT * region

def create_ascii_initializer(greedy_genome, ascii_length, perturbation_ratio=0.02):
    """
    Every individual starts from the greedy solution with a small random
    perturbation (~2 % of positions) so the population has genetic diversity
    without straying far from the near-optimal starting point.
    """
    def initializer():
        genome = greedy_genome.copy()
        n_perturb = max(1, int(ascii_length * perturbation_ratio))
        for pos in random.sample(range(ascii_length), n_perturb):
            genome[pos] = random.randint(0, len(CHARS) - 1)
        return genome
    return initializer

def row_crossover(cols):
    """
    Swap whole rows rather than arbitrary genome slices.
    This keeps spatial structure intact — a row from parent A is always
    combined with complete rows from parent B, never a partial row.
    """
    def _crossover(next_individual):
        while True:
            parent1 = next(next_individual)
            parent2 = next(next_individual)

            g1 = parent1.genome.copy()
            g2 = parent2.genome.copy()

            total_rows = len(g1) // cols
            crossover_row = random.randint(1, total_rows - 1)
            cut = crossover_row * cols

            child1 = np.concatenate([g1[:cut], g2[cut:]])
            child2 = np.concatenate([g2[:cut], g1[cut:]])

            parent1.genome = child1
            parent2.genome = child2
            parent1.fitness = None
            parent2.fitness = None

            yield parent1
            yield parent2

    return _crossover

#Shared stagnation state. Ipdated by the probe, read by the mutator.
stagnation_counter = [0]
last_fitness = [float('inf')]


def mutate_region(eval_data, templates):
    """
    Memetic mutation: explore tone globally, let greedy decide characters locally.

    Instead of randomising character indices directly, we:
      1. Pick a random rectangular patch of tiles.
      2. Apply a random brightness offset to the target pixels in that patch.
      3. Re-run greedy locally on the shifted patch to pick the best characters
         given that new tone interpretation.

    The EA therefore searches over *brightness interpretations* of regions
    (a global decision), while greedy handles character selection within each
    interpretation (a local decision). This is the classic memetic pattern:
    global search drives a local optimiser.

    Patch size and brightness range scale up with stagnation to escape
    local optima.
    """
    cols = eval_data.cols
    char_list = list(templates.keys())
    t_stack = np.stack([templates[c] for c in char_list], axis=0)

    def _mutate(next_individual):
        while True:
            individual = next(next_individual)
            genome = individual.genome.copy()
            total_rows = len(genome) // cols

            #Scale aggressiveness with stagnation.
            if stagnation_counter[0] > 20:
                max_patch_rows = max(2, total_rows // 4)
                max_patch_cols = max(2, cols // 4)
                brightness_range = 60
            else:
                max_patch_rows = max(1, total_rows // 8)
                max_patch_cols = max(1, cols // 8)
                brightness_range = 30

            patch_rows = random.randint(1, max_patch_rows)
            patch_cols = random.randint(1, max_patch_cols)
            start_row = random.randint(0, total_rows - patch_rows)
            start_col = random.randint(0, cols - patch_cols)
            brightness_shift = random.randint(-brightness_range, brightness_range)

            for r in range(start_row, start_row + patch_rows):
                for c in range(start_col, start_col + patch_cols):
                    r0 = r * eval_data.tile_h
                    r1 = r0 + eval_data.tile_h
                    c0 = c * eval_data.tile_w
                    c1 = c0 + eval_data.tile_w

                    #Shift the target tile's brightness and re-run greedy.
                    tile = eval_data.target_image[r0:r1, c0:c1].astype(np.float32)
                    tile = np.clip(tile + brightness_shift, 0, 255)

                    diff = t_stack - tile[np.newaxis, :, :]
                    mse = np.mean(diff ** 2, axis=(1, 2))
                    genome[r * cols + c] = np.argmin(mse)

            individual.genome = genome
            individual.fitness = None
            yield individual

    return _mutate

def make_image_saving_probe(eval_data, dir_name, every_n=50):
    generation = [0]

    def probe_fn(population):
        best = min(population, key=lambda ind: ind.fitness)

        #Update stagnation tracking (shared with mutator).
        if abs(best.fitness - last_fitness[0]) < 1e-6:
            stagnation_counter[0] += 1
        else:
            stagnation_counter[0] = 0
        last_fitness[0] = best.fitness

        if generation[0] % every_n == 0:
            print(f"Generation {generation[0]:4d}: "
                  f"fitness = {best.fitness:.6f}, "
                  f"stagnation = {stagnation_counter[0]}")
            img = Image.fromarray(genome_to_image_array(best.genome, eval_data))
            img.save(f"{dir_name}/result_after_{generation[0]}_iterations.jpeg")

        generation[0] += 1
        return population

    return probe_fn

if __name__ == "__main__":
    #Load image and font.
    print("Loading image and font...")
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    tile_w, tile_h = get_tile_size(font)
    img_array = np.array(load_and_resize(IMAGE_NAME, MAX_SIZE))
    bg, fg = get_bg_fg(img_array)
    img_array = crop_to_grid(img_array, tile_w, tile_h)
    h, w = img_array.shape
    cols = w // tile_w
    print(f"  Image: {w}x{h}px  |  Tile: {tile_w}x{tile_h}px  |  "
          f"Grid: {cols} cols x {h // tile_h} rows  |  "
          f"Background: {'white' if bg == 255 else 'black'}")

    #Greedy initialization.
    print("Running greedy initialization...")
    templates = build_char_templates(font, tile_w, tile_h, bg, fg)
    char_grid = greedy_match(img_array, templates, tile_w, tile_h)
    flat_chars = [c for row in char_grid for c in row]
    greedy_genome = np.array([CHARS.index(c) if c in CHARS else 0 for c in flat_chars], dtype=int)
    ascii_length = len(greedy_genome)
    print(f"  Greedy genome length: {ascii_length} characters")

    greedy_img = render_grid(char_grid, font, tile_w, tile_h, bg, fg)
    greedy_img.save("greedy_result.png")
    print("  Saved greedy_result.png")

    #EA refinement.
    eval_data = EvalData(img_array, font, tile_w, tile_h, cols, bg, fg)
    problem = ASCIIArtProblem(eval_data)

    greedy_arr = np.array(greedy_img)
    greedy_mse = np.mean((greedy_arr.astype(np.float32) - img_array.astype(np.float32)) ** 2) / (255.0 ** 2)
    greedy_ssim = ssim(greedy_arr, img_array, data_range=255)
    greedy_region = region_brightness_loss(greedy_arr, img_array, tile_w, tile_h, REGION_POOL_TILES)
    greedy_fitness = (MSE_WEIGHT * greedy_mse + SSIM_WEIGHT * (1.0 - greedy_ssim) / 2.0 + REGION_WEIGHT * greedy_region)
    print(f"  Greedy MSE={greedy_mse:.4f}  SSIM={greedy_ssim:.4f}  "
          f"Region={greedy_region:.4f}  fitness={greedy_fitness:.6f}")
    print("Starting EA refinement...")

    DIR_NAME = f"./{IMAGE_NAME}_{POP_SIZE}_{MAX_GENERATIONS}"
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    else:
        raise Exception(f"Directory {DIR_NAME} exists — rename it to avoid data loss")

    final_population = generational_ea(
        max_generations=MAX_GENERATIONS,
        pop_size=POP_SIZE,
        problem=problem,
        representation=Representation(initialize=create_ascii_initializer(greedy_genome, ascii_length, perturbation_ratio=0.02)),
        pipeline=[
            ops.tournament_selection(k=5),
            ops.clone,
            row_crossover(cols),              # spatially aware crossover
            mutate_region(eval_data, templates),  # memetic: tone search + greedy chars
            ops.evaluate,
            ops.pool(size=POP_SIZE),
            make_image_saving_probe(eval_data, DIR_NAME, PRINT_EVERY_N),
            probe.AttributesCSVProbe(
                stream=open(f"{DIR_NAME}/fitness_log.csv", "w"),
                do_fitness=True,
                best_only=True),
        ]
    )

    #Save final result.
    best = min(final_population, key=lambda ind: ind.fitness)
    print(f"\nFinal best fitness : {best.fitness:.6f}  "
          f"(greedy was {greedy_fitness:.6f})")

    final_img = Image.fromarray(genome_to_image_array(best.genome, eval_data))
    final_img.save(f"{DIR_NAME}/result.png")

    text = [CHARS[int(i) % len(CHARS)] for i in best.genome]
    with open(f"{DIR_NAME}/result.txt", 'w', encoding='utf-8') as f:
        for row_start in range(0, ascii_length, cols):
            f.write(''.join(text[row_start:row_start + cols]) + '\n')

    print(f"Saved final result to {DIR_NAME}/")
    plt.show()
    plt.close('all')