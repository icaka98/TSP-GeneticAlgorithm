from math import sqrt, cos, pi
import matplotlib.pyplot as plt
import random
import pickle


# Point on the unit circle (represented by an angle [degrees])
class Point:
    def __init__(self, angle, id):
        self.id = id
        self.angle = angle

    # Calculates the distance between 2 points on the unit circle
    def dist(self, point):
        diff = abs(point.angle - self.angle)
        diff = min(diff, 360.0 - diff)
        return sqrt(2.0 - 2.0 * cos(diff / 180.0 * pi))

    def __repr__(self):
        return "(" + str(self.id) + ': ' + str(self.angle) + ")"


# Calculates the fitness value for an individual
def fitness(path):
    distance = 0.0

    for i in range(len(path)):
        loc = path[i]
        dest = path[i + 1] if i + 1 < len(path) else path[0]

        distance += loc.dist(dest)

    return 1.0 / distance if distance > .0 else float('inf')


# Initialize the population
def init_population(size, points):
    population = []

    for i in range(size):
        population.append(random.sample(points, len(points)))

    return population


# Order population by fitness
def order_by_fitness(population):
    return sorted(population, key=lambda x: fitness(x), reverse=True)


# Selection procedure for the Genetic Algorithm
def selection(sorted_population, elite_size):
    selection_pool = sorted_population[:elite_size]

    return selection_pool


# Custom Partially Mapped Crossover (CPMX) implementation
def partially_mapped_crossover(pathA, pathB):
    child1 = [None for _ in range(len(pathA))]

    posA = int(random.random() * len(pathA))
    posB = int(random.random() * len(pathA))

    startPos = min(posA, posB)
    endPos = max(posA, posB)

    for i in range(startPos, endPos):
        child1[i] = pathA[i]

    to_add = []
    for i in range(len(pathB)):
        if pathB[i] not in child1:
            to_add.append(pathB[i])

    idx = 0
    for i in range(len(pathA)):
        if startPos <= i < endPos:
            continue

        child1[i] = to_add[idx]
        idx += 1

    return child1, list(reversed(child1))


# Produce the next population
def next_population(mating_pool, elite_size, population_size, type):
    children = mating_pool

    for _ in range(int((population_size - elite_size) / 2.0)):
        rand_parents = random.sample(mating_pool, 2)

        rand_parents[0] = mating_pool[0]
        rand_parents[1] = mating_pool[1]

        if type == 1:
            child1, child2 = modified_cycle_crossover(rand_parents[0], rand_parents[1])
        elif type == 2:
            child1, child2 = order_crossover(rand_parents[0], rand_parents[1])
        else:
            child1, child2 = partially_mapped_crossover(rand_parents[0], rand_parents[1])

        children.append(child1)
        children.append(child2)

    return children


# Mutate an individual by swapping two points
def mutate(path, mutation_rate):
    for i in range(len(path)):
        if random.random() < mutation_rate:
            to_swap = int(random.random() * len(path))

            pointA = path[i]
            pointB = path[to_swap]
            path[i] = pointB
            path[to_swap] = pointA

    return path


# Population can have individuals that mutate based on the mutation rate
def mutate_population(population, mutation_rate):
    mutated_population = []

    for individual in population:
        mutated_individual = mutate(individual, mutation_rate)
        mutated_population.append(mutated_individual)

    return mutated_population


# Compute the next generation of the Genetic Algorithm
def get_next_generation(generation, elite_size, mutation_rate, type):
    best_fitness_population = order_by_fitness(generation)
    mating_pool = selection(best_fitness_population, elite_size)
    children = next_population(mating_pool, elite_size, len(best_fitness_population), type)
    next_generation = mutate_population(children, mutation_rate)

    return next_generation


# Main function for executing the Genetic Algorithm
def genetic_algorithm(id, population, population_size, elite_size, mutation_rate, num_gens, cross_type):
    population = init_population(population_size, population)
    # print('Start fitness: ' + str(fitness(order_by_fitness(population)[0])))
    fitness_track = []
    best_fitness = float('-inf')
    best_path = None

    for i in range(num_gens):
        population = get_next_generation(population, elite_size, mutation_rate, cross_type)
        # print(1, fitness(order_by_fitness(population)[0]), order_by_fitness(population)[0])
        # print(2, fitness(order_by_fitness(population)[-1]), order_by_fitness(population)[-1])
        # print(3, len(population))

        current_path = order_by_fitness(population)[0]
        current_fitness = fitness(current_path)
        fitness_track.append(1.0 / current_fitness)

        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_path = current_path

    # print('Final fitness: ' + str(fitness(order_by_fitness(population)[0])))

    plt.plot(fitness_track)
    plt.ylabel('Distance')
    plt.xlabel('Generation')

    plt.savefig('image_results/result' + str(id) + '.png')

    plt.show()

    return best_fitness, best_path  # order_by_fitness(population)[0]


# Generate a new experiment with N points
def generate_experiment(N):
    population_angles = random.sample(range(0, 361), N)

    with open('experiments/exp_' + str(N) + '.txt', 'wb') as file:
        pickle.dump(population_angles, file)


# Load the experiment with N points
def load_experiment(N):
    with open('experiments/exp_' + str(N) + '.txt', 'rb') as file:
        population_angles = pickle.load(file)

    points = []

    idx = 0
    for angle in population_angles:
        points.append(Point(angle, idx))
        idx += 1

    return points


# Find point by id
def find_by_id(individual, id):
    idx = 0
    for point in individual:
        if point.id == id:
            return point, idx
        idx += 1

    return None, 0


# Modified Cycle Crossover (CX2) implementation
def modified_cycle_crossover(parent1, parent2, ref_parent1=None, ref_parent2=None):
    child1 = []
    child2 = []

    if ref_parent1 is None:
        ref_parent1 = parent1
    if ref_parent2 is None:
        ref_parent2 = parent2

    # print(parent2)
    prev_to_add = None
    for i in range(len(parent1)):
        if i == 0:
            to_add1 = parent2[0]
        else:
            same_num_p1, num_idx_p1 = find_by_id(ref_parent1, prev_to_add.id)
            to_add1 = ref_parent2[num_idx_p1]

        same_num_p1, num_idx_p1 = find_by_id(ref_parent1, to_add1.id)

        # print(1, num_idx_p1)
        same_id_p2 = ref_parent2[num_idx_p1]

        same_num_p1, num_idx_p1 = find_by_id(ref_parent1, same_id_p2.id)
        # print(2, num_idx_p1)
        to_add2 = ref_parent2[num_idx_p1]

        child1.append(to_add1)
        child2.append(to_add2)

        prev_to_add = to_add2

        # print('to add:', to_add1, to_add2)
        # print('child 1: ', child1)
        # print('child 2: ', child2)

        # Check for cycle
        if to_add2.id == parent1[0].id and i != (len(parent1) - 1):
            rec_parent1 = list(filter(lambda x: x.id not in list(map(lambda y: int(y.id), child2)), parent1))
            rec_parent2 = list(filter(lambda x: x.id not in list(map(lambda y: int(y.id), child1)), parent2))

            # print(10, rec_parent1)
            # print(20, rec_parent2)

            next_child1, next_child2 = modified_cycle_crossover(
                rec_parent1, rec_parent2,
                ref_parent1=ref_parent1, ref_parent2=ref_parent2)

            child1 += next_child1
            child2 += next_child2

            break

    return child1, child2


# Checks if an individual solution contains a point on the unit circle
def contains(child, point):
    for pt in child:
        if pt is not None and pt.id == point.id:
            return True

    return False


# Order Crossover (OX) implementation
def order_crossover(parent1, parent2):
    child1 = [None for x in range(len(parent1))]
    child2 = [None for x in range(len(parent1))]

    posA = int(random.random() * len(parent1))
    posB = int(random.random() * len(parent1))

    startPos = min(posA, posB)
    endPos = max(posA, posB)

    child1[startPos:endPos] = parent1[startPos:endPos]
    child2[startPos:endPos] = parent2[startPos:endPos]

    to_add1 = parent2[endPos:] + parent2[:endPos]
    to_add2 = parent1[endPos:] + parent1[:endPos]

    iter_range = list(range(endPos, len(parent1))) + list(range(0, startPos))

    to_add1 = list(filter(lambda x: not contains(child1, x), to_add1))
    to_add2 = list(filter(lambda x: not contains(child2, x), to_add2))

    idx1 = 0
    for i in iter_range:
        nxt = to_add1[idx1]
        child1[i] = nxt
        idx1 += 1

    idx1 = 0
    for i in iter_range:
        nxt = to_add2[idx1]
        child2[i] = nxt
        idx1 += 1

    return child1, child2


# Check if solution is optimal
def is_solution_optimal(solution):
    if solution is None:
        return False

    num_lower = 0
    num_higher = 0

    # clockwise check
    for i in range(len(solution) - 1):
        if solution[i].angle > solution[i + 1].angle:
            num_lower += 1

    optimal_clockwise = num_lower in [0, 1]

    # anticlockwise check
    for i in range(len(solution) - 1):
        if solution[i].angle < solution[i + 1].angle:
            num_higher += 1

    optimal_anticlockwise = num_higher in [0, 1]

    return optimal_clockwise or optimal_anticlockwise


# Get the optimal fitness value
def get_optimal_fitness(problem):
    problem_sorted = sorted(problem, key=lambda x: x.angle)
    return fitness(problem_sorted)


if __name__ == '__main__':
    K = 5  # Number of runs for each experiment

    file = open("results.txt", "w")

    # Custom experiment
    Ns = [20, 30, 45, 60]
    mutation_rates = [.01, .075]
    population_sizes = [200]
    elite_sizes = [.25]
    crossover_types = [1, 2, 3]
    generation_sizes = [200, 300, 500]

    # Hard but simplified setup
    # Ns = [60]
    # mutation_rates = [.075]
    # population_sizes = [200]
    # elite_sizes = [.25]
    # crossover_types = [1, 2, 3]
    # generation_sizes = [200, 300, 500]

    # Easy setup
    # Ns = [20]
    # mutation_rates = [.01]
    # population_sizes = [200]
    # elite_sizes = [.25]
    # crossover_types = [3]
    # generation_sizes = [200]

    ID = 63
    results = list()

    idx = 0
    max_idx = len(Ns) * len(mutation_rates) * len(population_sizes) * len(elite_sizes) \
             * len(crossover_types) * len(generation_sizes) * K

    for N in Ns:
        test_points = load_experiment(N)

        optimal_fitness = get_optimal_fitness(test_points)

        for mut_rate in mutation_rates:
            for pop_size in population_sizes:
                for elite_size in elite_sizes:
                    elite_size = int(pop_size * elite_size)

                    for cros_type in crossover_types:
                        for gen_size in generation_sizes:
                            best_fitness = float('-inf')
                            worst_fitness = float('inf')
                            sum_fitness = 0.0

                            final_solution = None

                            for i in range(K):
                                final_fitness, final_solution = genetic_algorithm(str(N) + '_' + str(ID) + '_' + str(i),
                                                                   test_points, pop_size,
                                                                   elite_size, mut_rate,
                                                                   gen_size, cros_type)

                                if final_fitness > best_fitness:
                                    best_fitness = final_fitness

                                if final_fitness < worst_fitness:
                                    worst_fitness = final_fitness

                                sum_fitness += final_fitness

                                print('Step ', idx, ' of ', max_idx)
                                print(N, i, final_fitness)
                                idx += 1

                            avg_fitness = sum_fitness / K

                            if cros_type == 1:
                                str_cros_type = 'CX2'
                            elif cros_type == 2:
                                str_cros_type = 'OX'
                            else:
                                str_cros_type = 'PMX'

                            is_optimal_sol = 'YES' if is_solution_optimal(final_solution) else 'NO'

                            # Print no population size and elite size
                            result = [ID, N, round(optimal_fitness, 4), str_cros_type, mut_rate,
                                      gen_size, round(worst_fitness, 4), round(avg_fitness, 4),
                                      round(best_fitness, 4), is_optimal_sol]

                            # Print everything
                            # result = [ID, N, round(optimal_fitness, 4), cros_type, mut_rate, pop_size,
                            #           elite_size, gen_size, round(worst_fitness, 4), round(avg_fitness, 4),
                            #           round(best_fitness, 4), is_optimal_sol]

                            ID += 1

                            result = list(map(lambda x: str(x), result))

                            file.write(' & '.join(result) + '\n')

    file.close()
