import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import math
import operator

plt.ion()
plt.figure(figsize=(10,5))

def plotTSP(generation, path, points, path_distance, save, num_iters=1):

	"""
	generation: The generation number to display
	path: List of lists with the different orders in which the nodes are visited
	points: coordinates for the different nodes
	path_distance: the distance to display in the figure
	save: True if saving to final_route.png, False otherwise
	num_iters: number of paths that are in the path list

	SOURCE: https://gist.github.com/payoung/6087046

	"""
	### MOD: Brian Bennett

	plt.suptitle("Tennessee Traveling Salesman - Generation " + str(generation) + \
				 "\nPath Length: " + str(path_distance))
	### END MOD

	# Unpack the primary TSP path and transform it into a list of ordered
	# coordinates

	x = []; y = []
	for i in path:
		x.append(points[i][0])
		y.append(points[i][1])

	plt.cla()

	plt.plot(x, y, 'ko')

	# Set a scale for the arrow heads (there should be a reasonable default for this)
	a_scale = 3.5
	# Draw the older paths, if provided
	if num_iters > 1:

		for i in range(1, num_iters):

			# Transform the old paths into a list of coordinates
			xi = []; yi = [];
			for j in paths[i]:
				xi.append(points[j][0])
				yi.append(points[j][1])

			plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]),
					head_width = a_scale, color = 'r',
					length_includes_head = True, ls = 'dashed',
					width = 0.001/float(num_iters))
			for i in range(0, len(x) - 1):
				plt.arrow(xi[i], yi[i], (xi[i+1] - xi[i]), (yi[i+1] - yi[i]),
						head_width = a_scale, color = 'r', length_includes_head = True,
						ls = 'dashed', width = 0.001/float(num_iters))

	# Draw the primary path for the TSP problem
	plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale,
			color ='b', length_includes_head=True)
	for i in range(0,len(x)-1):
		plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
				color = 'b', length_includes_head = True)

	if save:
		plt.savefig("final_route.png")

	plt.pause(1)

class GeneticSearch:
	"""
		Class: GeneticSearch
	"""
	def __init__(self, origin, generations, points, cities, population_size, mutation_rate):
		self.population = None
		self.points = points
		self.cities = cities
		self.chromosome_size = len(self.points)
		self.generations = generations
		self.population_size = population_size
		self.mutation_rate = mutation_rate
		self.origin = origin
		self.origin_index = self.points.index(self.origin)
		self.values = []


	def print_population(self, generation, chromosomes):
		index = 0
		print("===== GENERATION %d" % generation)
		for chromosome in self.population:
			print ("Index %5d , Fitness %0.4f : %s" % (index,chromosome[1], ''.join(str(chromosome[0]))))
			index = index + 1
			if index > chromosomes:
				break


	def initialize_population(self):

		self.population = []

		# TODO: This code generates a random initial population.
		#       You may adjust this code in any way that you believe would help.
		for i in range(self.population_size):

			individual = [x for x in range(self.chromosome_size)]
			random.shuffle(individual)

			# Move the origin_index to the front of the path
			individual.remove(self.origin_index)
			individual = [self.origin_index] + individual

			fitness = self.fitnessfcn(individual)

			# Prevent duplicate individuals in the initial population
			while [individual,fitness] in self.population:
				individual = [x for x in range(self.chromosome_size)]
				random.shuffle(individual)

				individual.remove(self.origin_index)
				individual = [self.origin_index] + individual

				fitness = self.fitnessfcn(individual)

			# POPULATION NODES are in the form [chromosome, fitness]
			self.population.append([individual,fitness])

		# Sort the population in descending order
		# -- "Maximize the objective function"
		self.population.sort(key=operator.itemgetter(1),reverse=True)


	def straight_line_distance(self, p1, p2):
		'''
			Return the Euclidian Distance between p1 and p2
		'''
		# Calculate the straight-line distance between p1 and p2 (both in the form (x,y))
		# p[0] = x and p[1] = y
		sld = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
		return sld


	def route_distance(self, individual):
		'''
			Determine the distance for the entire route
		'''

		distance = 0
		
		for i in range(len(individual)-1):
			p1 = self.points[individual[i]]
			p2 = self.points[individual[i+1]]
			distance += self.straight_line_distance(p1, p2)

		# need to return to origin city at the end of TSP
		# p1 = the last city and p2 = the first city
		p1 = self.points[individual[-1]]
		p2 = self.points[individual[0]]
		distance += self.straight_line_distance(p1,p2)

		return distance


	def fitnessfcn(self, individual):
		'''
			Return the negative route distance so it can be maximized.
		'''
		return -self.route_distance(individual)


	def select_parents(self):
		'''
			Selects two parents from the population and returns them as a list
		'''

		# random selection strategy
		parent1 = self.population[random.randint(0,self.population_size-1)][0]
		parent2 = self.population[random.randint(0,self.population_size-1)][0]

		# If the same parent is chosen, pick another
		while str(parent1) == str(parent2):
			parent2 = self.population[random.randint(0,self.population_size-1)][0]
		return parent1, parent2


	def reproduce(self,parent1,parent2):
		'''
			Reproduce using parent1 and parent2 and a crossover
			 strategy.
		'''

		child1 = []
		child2 = []

		# in case the parents are different sizes
		max_index = min(len(parent1), len(parent2))
		i = random.randint(0,max_index-1)
		j = random.randint(0,max_index-1)

		while i == j:
			i = random.randint(0,max_index-1)
		
		# place the first crossover point based on the lower index
		gene_index_start, gene_index_end = (i,j) if i<j else (j,i)

		child1 = parent1[0:gene_index_start] + parent2[gene_index_start:gene_index_end] + parent1[gene_index_end:]
		child2 = parent2[0:gene_index_start] + parent1[gene_index_start:gene_index_end] + parent2[gene_index_end:]

		return child1, child2


	def mutate(self,child):
		'''
			Mutation Strategy
		'''
		# swap two cities (not the origin city)
		i, j = random.sample(range(1, len(child)), 2)

		while i == j:
			j = random.sample(range(1, len(child)),2)

		child[i], child[j] = child[j], child[i]
		return child


	def print_result(self):
		'''
			Displays the resulting route in the console.
		'''
		individual = self.population[0][0]
		fitness = self.population[0][1]

		print(" Final Route in %d Generations" % self.generations)
		print(" Final Distance : %5.3f\n" % -fitness)

		counter = 1

		for index in individual:
			print ("%2d. %s" % (counter, self.cities[index]))
			counter += 1

		print ("%2d. %s" % (counter, self.cities[self.origin_index]))


	def run(self):
		'''
			Run the genetic algorithm. Note that this method initializes the
			 first population.
		'''
		generations = 0

		# TODO: Update Initialization
		self.initialize_population()

		last_fitness = 0
		fitness_counter = 0

		while generations <= self.generations:
			new_population = []
			parent1 = []
			parent2 = []

			while len(new_population) < self.population_size:

				# TODO: Update selection
				parent1, parent2 = self.select_parents()
				# TODO: Update reproduction
				child1, child2 = self.reproduce(parent1,parent2)

				# TODO: Update Mutation
				# Generate a random number, and only mutate if the number
				#  is below the mutation rate.
				if (random.random() < self.mutation_rate):
					child1 = self.mutate(child1)
				if (random.random() < self.mutation_rate):
					child2 = self.mutate(child2)

				fitness1 = self.fitnessfcn(child1)
				fitness2 = self.fitnessfcn(child2)

				new_population.append([child1,fitness1])
				new_population.append([child2,fitness2])

			generations = generations + 1

			# Sort the new population in descending order
			new_population.sort(key=operator.itemgetter(1),reverse=True)

			self.population = new_population

			# TODO: Change display rate as needed. Set by 1000 as default.
			if generations % 100 == 0 or generations >= self.generations:
				print("Generation: %d" % generations,"Fitness: %f" % self.population[0][1])
				if generations == self.generations:
					plotTSP(generations, self.population[0][0], self.points, self.population[0][1],True)
				else:
					plotTSP(generations, self.population[0][0], self.points, self.population[0][1],False)

			self.values.append(self.population[0][1])

		self.print_result()


if __name__ == '__main__':

	city_coordinates = "coordinates.txt"
	city_names = "cities.txt"
	start_city = "Johnson City, TN"
	locations = list(np.loadtxt(city_coordinates))
	cities = [line.rstrip('\n') for line in open(city_names)]
	points = []
	paths = []
	start_city_index = [i for i in range(len(cities)) if cities[i] == start_city][0]

	loc_x = [x for x,y in locations]
	loc_y = [y for x,y in locations]
	loc_c = ["black" for _ in range(len(locations))]

	for i in range(0, len(loc_x)):
		points.append((loc_x[i], loc_y[i]))

	#origin, generations, points, population_size, mutation_rate
	origin = (locations[start_city_index][0],locations[start_city_index][1])

	# TODO: Adjust parameters as needed
	# Parameters: 1. origin location,
	#             2. number of generations,
	#             3. locations as a list of tuples,
	#             4. list of city names,
	#             5. number of individuals in each generation,
	#             6. mutation rate
	gs = GeneticSearch(origin, 3000, points, cities, 100, 0.5)
	gs.run()

	x = input("Press Enter to Exit...")
	plt.close()
