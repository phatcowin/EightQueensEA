import random
import numpy as np
import matplotlib.pyplot as plt

class GeneticNQueens:
    def __init__(self, N=8, population_size=100, mutation_rate=0.1, generations=1000):
        self.N = N
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = [self.random_board() for _ in range(population_size)]
        self.history = []  # Store fitness values over generations

    def random_board(self):
        """Generate a random board representation"""
        return [random.randint(0, self.N - 1) for _ in range(self.N)]

    def fitness(self, board):
        """Calculate the number of attacking queen pairs (lower is better)"""
        conflicts = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if board[i] == board[j] or abs(board[i] - board[j]) == j - i:
                    conflicts += 1
        return conflicts

    def tournament_selection(self):
        """Select the best individual from a random subset of the population"""
        tournament = random.sample(self.population, k=5)
        return min(tournament, key=self.fitness)

    def crossover(self, parent1, parent2):
        """Perform crossover by swapping a random part of two parents"""
        cut = random.randint(1, self.N - 2)
        child = parent1[:cut] + parent2[cut:]
        return child

    def mutate(self, board):
        """Randomly mutate a board by changing the position of one queen"""
        if random.random() < self.mutation_rate:
            row = random.randint(0, self.N - 1)
            board[row] = random.randint(0, self.N - 1)
        return board

    def evolve(self):
        """Run the genetic algorithm to find a solution"""
        for generation in range(self.generations):
            self.population = sorted(self.population, key=self.fitness)
            best_fitness = self.fitness(self.population[0])
            worst_fitness = self.fitness(self.population[-1])
            avg_fitness = np.mean([self.fitness(ind) for ind in self.population])
            
            # Store fitness history
            self.history.append((generation, best_fitness, avg_fitness, worst_fitness))
            
            # If global optimum (fitness=0) is found, stop
            if best_fitness == 0:
                print(f"Solution found in generation {generation}")
                return self.population[0]

            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.tournament_selection(), self.tournament_selection()
                child1, child2 = self.crossover(parent1, parent2), self.crossover(parent2, parent1)
                new_population.extend([self.mutate(child1), self.mutate(child2)])

            self.population = new_population

        print("No perfect solution found.")
        return None

    def print_board(self, board):
        """Print the board with queens"""
        if board:
            for row in range(self.N):
                line = ['.'] * self.N
                line[board[row]] = 'Q'
                print(" ".join(line))
            print("\n" + "-" * (2 * self.N - 1))

    def plot_fitness(self):
        """Scatter plot of best, worst, and average fitness over generations"""
        data = np.array(self.history)
        generations = data[:, 0]

        plt.figure(figsize=(10, 5))
        plt.scatter(generations, data[:, 1], label="Best Fitness (Global Optimum)", color="green", alpha=0.6)
        plt.scatter(generations, data[:, 2], label="Average Fitness", color="blue", alpha=0.6)
        plt.scatter(generations, data[:, 3], label="Worst Fitness (Local Optima)", color="red", alpha=0.6)

        plt.xlabel("Generations")
        plt.ylabel("Fitness (Lower is Better)")
        plt.title(f"Evolution of Fitness Over {len(self.history)} Generations (Scatter Plot)")
        plt.legend()
        plt.grid()
        plt.show()

# Run the Genetic Algorithm with Visualization
solver = GeneticNQueens()
solution = solver.evolve()
if solution:
    solver.print_board(solution)

# Plot fitness evolution
solver.plot_fitness()
