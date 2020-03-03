#!/usr/bin/env python3
"""
Name:  Agim Allkanjari (agim@tufts.edu)
Student ID: 1333940
Name:  Jiayi (Frank) Ma (jiayi.ma@tufts.edu)
Student ID: 1238671
Assignment 4: Genetic Algorithms
"""
import time
import numpy as np
import random


"""
1. Define the problem as a genetic algorithm
2. Provide the genome for the problem
3. Define all the fringe operations
4. Cull your population by 50% at every generation
"""

class Box:
    label = ""
    weight = 0 
    importance = 0

    def __init__(self, label, weight, importance):
        """
        Creates a new Box Object

        Args
        ----
        label : string
            Boxe label or identifier

        weight: int
            Box Weight
            
        importance: int
            Importance of box

        """
        self.label = label
        self.weight = weight
        self.importance = importance


    def get_weight(self):
        return self.get_weight


    def get_importance(self):
        return self.importance

    def __str__(self):
        """
        Box info printing
        """
        return "Box Label: {}, Weight: {}, Importance: {}".format(self.label, self.weight, self.importance)



class BackPack:
    size = 0

    def __init__(self, size):
        """
        Creates a new BackPack Object that is used to hold boxes

        Args
        ----
        size: int
            Max weight it can hold

        """
        self.size = size


# Initialize BoxObjects
all_boxes = np.array([
    Box("7", 30, 4),
    Box("3", 60, 8),
    Box("1", 20, 6),
    Box("4", 90, 7),
    Box("5", 50, 6),
    Box("6", 70, 9),
    Box("2", 30, 5),
])

# Create the backpack with max size 120
backPack = BackPack(120)


class GA:
    """
    Genetic Algorithms Class
    """
    cull_rate = 0.5
    max_time  = None
    max_iter  = None
    mutation_prob = 0.01
    convergence_limit = None

    all_boxes_weight = 0
    all_boxes_importance = 0

    def __init__(self, cull_rate = 0.5, max_iter=None, max_time=None, mutation_prob = 0.01, convergence_limit=None):
        self.cull_rate = cull_rate
        self.max_time  = max_time
        self.max_iter  = max_iter
        self.mutation_prob = mutation_prob
        self.convergence_limit = convergence_limit

        # Calculate all boxes weight and importance
        for box in all_boxes:
            self.all_boxes_importance += box.importance
            self.all_boxes_weight += box.weight


    def generate_random(self):
        """
        Generates a random individual as [True, False.....True], meaning that boxes at indexs of value: True are possible indiviudals
        The size of indiviudal is the the size of boxes
    
        Returns
        -------
        population: array
            Returns the indivudal as a vector of True,False values
        """
        vector = []
        for i in range(len(all_boxes)):
            vector.append(random.choice([True, False]))
        return vector


    def random_population(self, count=10):
        """
        Creates a population with random individuals
    
        Args:
        -------
        count: int
            Number of individuals in population
        
        Returns
        -------
        population: array
            Returns population with random indiduals
        """
        population = []
        for i in range(count):
            individual = self.generate_random()
           
            population.append(individual)
        
        return population
    

    def order_by_fitness(self, population):
        """
        Orders population by fitness function
    
        Args:
        -------
        popuplation: array
            An array of individuals
        
        Returns
        -------
        ordered: array
            Returns ordered population according to the above method
        high_score: float
            Returns the highest score in population
        """
        ordered = []

        # Calculate score of each individual in population
        for i, individual in enumerate(population):
            importance, weight, score = self.fitness(all_boxes[individual])
            ordered.append((individual, importance, weight, score))

        # Order by highest score
        ordered.sort(key=lambda x: x[3], reverse=True)
        return [item[0] for item in ordered], ordered[0][3]


    def cull_population(self, population):
        """
        Checks the fitness function for an item

        Args:
        -------
        population: array
            An array of ordered individuals
        
        Returns
        -------
        culled_population: array
            A new array of individuals reduced by cull rate

        """
        n = int(len(population)*(1-self.cull_rate))
        return population[:n]


    def fitness(self, selection):
        """
        Checks the fitness function for a selection individual

        Args:
        -------
        selection: array
            A possible list of boxes
        
        Returns
        -------
        importance: int
            Total importance of all boxes in selection
        weight : int
            Total weight of all boxes in selection
        score : int
            Score of boxes in selection

        """
        importance, weight, score = 0, 0, 0
        for box in selection:
            importance += box.importance
            weight += box.weight

        # Calculate score as importance - penalty from weight
        penalty = self.all_boxes_weight * max(0, weight - backPack.size)/500
        score = importance - penalty 
        
        return importance, weight, score


    def mutation(self, x):
        """
        A Mutation fringe operation: given a candidate, return a slightly different candidate
        
        Args:
        -------
        x: array
            An individual
        
        Returns
        -------
        x: array
            The new individual with a random item mutated
        """
        idx = random.randint(0, len(x)-1)
        x[idx] = not x[idx]

        return x


    def crossover(self, x, y):
        """
        A Crossover fringe operation: given two candidates, produce one that has elements of each
        """
        n = len(x)
        c = random.randint(1, n)
        crossover_1 = x[:c] + y[c:]
        crossover_2 = y[:c] + x[c:]
        return crossover_1, crossover_2


    def run(self, population):
        start_time = time.time()
        iterations = 0

        # Store best run
        best_run    = None
        best_score  = 0
        best_run_iteration = 0

        while True:
            #Order population by fitness
            population, score = self.order_by_fitness(population)

            #Store best run
            if best_run is None or score > best_score:
                best_score  = score
                best_run    = population[0]
                best_run_iteration = iterations
            
            # Cull population and keep the top 50% in new population
            new_population = self.cull_population(population)

            # Genetic Algorithm
            size = len(new_population)
            for i in range(size):
                # Sample 2 individuals from new population
                x, y = random.sample(new_population, 2) # x and y

                # Crossover the two individuals
                crossed_1, crossed_2 = self.crossover(x, y)

                # Mutate with low probability
                if (np.random.uniform(0, 1) < self.mutation_prob):
                    crossed_1 = self.mutation(crossed_1)

                # Only append one child from the crossover 
                new_population.append(crossed_1)
            
            # Replace old population with new one
            population = new_population

            # Increment number of iterations
            iterations += 1
                
            # Check for convergence
            if self.convergence_limit is not None and iterations > best_run_iteration + self.convergence_limit:
                break

            # Check for max iterations
            if self.max_iter is not None and iterations >= self.max_iter:
                break
            
            # Check for max time
            if self.max_time is not None and time.time() >= start_time+self.max_time:
                break

        return best_run, iterations


def main():
    print('************************************')
    print('***       Genetic Algorithm      ***')
    print('************************************')
    print("Initial Boxes:")
    for box in all_boxes:
        print(box)

    print("")
    print("BackPack Size:", backPack.size)

    # Global variables
    runs = 10  # Random for each case    
    population_size = 2**5 #Initial Population size
    convergence_limit = 200 # Desired convergence limit, if the best_run does not improve
    mutation_prob = 0.1 # Mutation probability
    max_time = .5 # Maximum execution time
    max_iter = 1000 # Maximum number of iterations

    # Run multiple times with maximum of 0.5 seconds of runtime
    for i in range(runs):
        # Record initial Time
        start_time = time.time()

        # Run Genetic Algorithm
        ga = GA(max_time=max_time, mutation_prob=mutation_prob, convergence_limit=convergence_limit)
        print("")
        print('***  Genetic Algorithm Run {:2d} of {}  with maximum run time of {} seconds  ***'.format(i+1, runs, max_time))
        population = ga.random_population(population_size)

        # Run Genetic Algorithm
        solution, iterations = ga.run(population)

        # Print Run Time
        print("--- Run Time: %0.5f seconds ---" % (time.time() - start_time))
        print("--- Iterations: %d iterations ---" % iterations)

        # Print Solution
        print("--- Solution ---")
        for box in all_boxes[solution]:
            print(box)
        weight, importance, score = ga.fitness(all_boxes[solution])
        print("Score: {}".format(score))


    # Run multiple times with maximum number of iterations
    for i in range(runs):
        # Record initial Time
        start_time = time.time()

        # Run Genetic Algorithm
        ga = GA(max_iter=max_iter,mutation_prob=mutation_prob, convergence_limit=convergence_limit)
        print("")
        print('***  Genetic Algorithm Run {:2d} of {}  with maximum of {} iterations  ***'.format(i+1, runs, max_iter))
        population = ga.random_population(population_size)

        # Run Genetic Algorithm
        solution, iterations =  ga.run(population)

        # Print Run Time
        print("--- Run Time: %0.5f seconds ---" % (time.time() - start_time))
        print("--- Iterations: %d iterations ---" % iterations)

        # Print Solution
        print("--- Solution ---")
        for box in all_boxes[solution]:
            print(box)
        weight, importance, score = ga.fitness(all_boxes[solution])
        print("Score: {}".format(score))


if __name__ == "__main__":
    main()
