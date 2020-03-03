# Genetic Algorithms

`ga.py` returns a solution to a knapsack problem specified in `GA.pdf`
using a genetic algorithm. The knapsack problem is framed as such: there is
a backpack with a fixed size and a set of boxes with different sizes and
importance values. The goal of the program is to achieve the highest total
importance possible while remaining under the size limit of the backpack. 

## Language and libraries:

Written in: Python 3 using the following libraries: random, numpy, time

## Directory Structure:
~~~~
   |--ga.py
   |—-README.md
~~~~
## Execution instructions:
Change the directory to where the it is located (e.g: $ cd GA)

Run the following command: `$ python ga.py`

By executing the command above, the system will run GA Algorithm on 20
random runs of random populations of size 32. The global variables are
specified below.

~~~~
   # Global variables
   runs = 10  # Random for each case    
   population_size = 2**5 # Initial Population size
   convergence_limit = 200 # Desired convergence limit, if the est_run does not improve
   mutation_prob = 0.1 # Mutation probability
   max_time = .5 # Maximum execution time
   max_iter = 1000 # Maximum number of iterations
~~~~

## Design Choices
### Chromosome
Each chromosome is designed as a binary array of length 7. Each element of
this binary array represents whether or not a given box is included in the
bag.

### Fitness Function
For each chromosome, a fitness score is calculated. To arrive at the final
fitness score, a weight penalty is first calculated. For chromosomes with a
total weight of under 120, the weight penalty is always 0. For those with a
total weight of over 120, the weight penalty is a fraction of how much it
exceeds 120. 

### Culling
At the start of each generation, 50% of the least fit individuals from the
previous generation are culled. The top 50% from the previous generation is
kept. This choice was made since discarding the entirety of the previous
generation caused the population to fluctuate too violently. 

### Fringe Operations
Both mutations and crossover operations are implemented. A mutation rate of
10% was chosen.

## Outcomes
In the case where the algorithm does not produce the optimal solution (20),
it means that it is stuck at a local maximum (e.g. 19). 


## Collaborators and resources:
- Lecture Slides

- Discussions with Prof Santini

- https://stackoverflow.com/questions/339007/nicest-way-to-pad-zeroes-to-a-string

- https://docs.python.org/2/library/random.htm

- numpy documentation

- http://www.optiwater.com/optiga/ga.html


## Authors
Jiayi (Frank) Ma (jiayi.ma@tufts.edu) Agim Allkanjari (agim@tufts.edu)