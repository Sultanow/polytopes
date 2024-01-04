import math
import numpy as np
from sympy.geometry import Point
import matplotlib.pyplot as plt
# from polytope import polytope
# from polytope.polytope import enumerate_integral_points, qhull, box2poly
from scipy.spatial import ConvexHull
import random
from random import randint
import copy
from fractions import Fraction
import cdd
from polytope.polytope import qhull, enumerate_integral_points, ABS_TOL
import sys


class Population:
    def __init__(self, number_of_generations, survival_rate, crossover_type, crossover_parent_choosing_type, chrom_list, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point):
        #self.generation_0 = Generation(chrom_list, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point)
        
        self.number_of_generations = number_of_generations
        self.all_generations = []
        
        self.all_generations.append(Generation(chrom_list, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point))
        
        for curr_generation_number in range(0, number_of_generations-1):
            if curr_generation_number % 10 == 0:
                print("curr Generation count: " + str(curr_generation_number))
                
            self.copy_prev_Generation_chrom_list = copy.deepcopy(self.all_generations[curr_generation_number].get_chrom_list())
            self.copy_prev_Generation_chrom_list = self.copy_prev_Generation_chrom_list[:int(len(self.copy_prev_Generation_chrom_list)/2)]
            parents = []
            while len(self.copy_prev_Generation_chrom_list) < number_of_generations: #len(self.all_generations[0].get_chrom_list()):
                print(len(self.copy_prev_Generation_chrom_list))
                parents = random.sample(self.copy_prev_Generation_chrom_list, 2)
                print("null")
                crossover_chromosom = Chromosom.apply_crossover(parents[0], parents[1], crossover_type)
                #print("eins")
                if crossover_chromosom.get_is_valid():
                    self.copy_prev_Generation_chrom_list.append(crossover_chromosom)
                    #print("zwei")
                
            self.copy_prev_Generation_chrom_list.sort(key=lambda x: x.fitness, reverse=True)
            self.all_generations.append(Generation(self.copy_prev_Generation_chrom_list, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point))


    def get_all_generations(self):
        return self.all_generations


class Generation:
    num_points_per_polytope = 0
    num_chroms_per_Generation = 0
    num_dimensions = 0 # 3
    min_value_per_point = 0  # -3
    max_value_per_point = 0  # 3
    
    
    def __init__(self, chrom_list, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point):
        
        Generation.num_points_per_polytope = num_points_per_polytope
        Generation.num_chroms_per_Generation = num_chroms_per_Generation
        Generation.num_dimensions = num_dimensions # 3
        Generation.min_value_per_point = min_value_per_point  # -3
        Generation.max_value_per_point = max_value_per_point  # 3
        
        if chrom_list == None:
            self.chrom_list = self.gen_first_generation(num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point)
        else:
            self.chrom_list = chrom_list

        self.generation_fitness_list = self.set_fitness_value_list(self.chrom_list)

    def set_fitness_value_list(self, chrom_list):
        fitness = []
        for chrom in chrom_list:
            fitness.append(chrom.get_fitness())

        #print(fitness)
        return(fitness)

    def get_chrom_list(self):
        return self.chrom_list
    
    def get_dimension():
        return Generation.num_dimensions

    def get_min_value_per_point():
        return Generation.min_value_per_point
    
    def get_max_value_per_point():
        return Generation.max_value_per_point

    def get_fitness_values_list(self):
        return self.generation_fitness_list

    def gen_first_generation(self, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point):
        #num_points_per_polytope = 5
        #num_chroms_per_Generation = 10
        #num_dimensions = num_dimensions

        polytope_points = []
        generation_polytopes = []

        #for j in range(0, num_chroms_per_Generation):
        while len(generation_polytopes) < num_chroms_per_Generation:
            #chromosome_test_points=[]
            polytope_points = []
            for i in range(0,num_points_per_polytope):
                polytope_points.append(random.sample(range(min_value_per_point, max_value_per_point), num_dimensions))
            chromosome_polytope_points = np.array(polytope_points)
            #print(polytope_points)
            chromosom_test = Chromosom(chromosome_polytope_points)

            if chromosom_test.get_is_valid():
                generation_polytopes.append(chromosom_test)
        
        generation_polytopes.sort(key=lambda x: x.fitness, reverse=True)
        #print("generation done")
        return generation_polytopes

class Chromosom:
    chrom_count = 0
    reflexive_politopes = {}

    def __init__(self, points):
        chromosome_polytope_points = points
        #print("pre try")
        #print(Generation.num_dimensions)
        self.is_valid = False

        print("before try")
        #result = None
        #while result is None:
        try:
            print("in try")
            internal_points = []
            #print("###############################################")
            Chromosom.chrom_count = Chromosom.chrom_count + 1
            #print(Chromosom.chrom_count)
            
            self.gen_list = chromosome_polytope_points
            print("before fitness")
            self.fitness = self.calc_fitness(chromosome_polytope_points)
            #print(True)
            self.is_valid = True
            print("after fitness")
            if self.fitness == 0: 
                print("in if")
                print(True)
                polytope_volume = ConvexHull(chromosome_polytope_points).volume #qhull(chromosome_polytope_points).volume
                Chromosom.reflexive_politopes[polytope_volume] = ConvexHull(self.gen_list)
            #result = 1
        except:
            (print("except"))
            self.is_valid = False
    
    def get_is_valid(self):
        return self.is_valid
    
    def get_gen_list(self):
        return self.gen_list
    
    def get_fitness(self):
        return self.fitness

    def get_gen_list(self):
        return self.gen_list

    #code based on https://github.com/stephane-caron/pypoman/blob/master/pypoman/duality.py
    def compute_distances(self, vertices: np.ndarray) -> np.ndarray:
        #print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
        
        V = np.vstack(vertices)
        t = np.ones((V.shape[0], 1))  # first column is 1 for vertices
        tV = np.hstack([t, V])
        mat = cdd.Matrix(tV, number_type="fraction")
        mat.rep_type = cdd.RepType.GENERATOR
        P = cdd.Polyhedron(mat)
        bA = np.array(P.get_inequalities())
        if bA.shape == (0,):  # bA == []
            return bA
        # the polyhedron is given by b + A x >= 0 where bA = [b|A]
        b = bA[:, 0]
        A = -np.array(bA[:, 1:])

        for i, row in enumerate(A):
            denominators = []
            has_fraction = False
            for j in row:
                if type(j) == Fraction:
                    has_fraction = True
                    denominators.append(j.denominator)    
            if has_fraction:
                b[i] *= np.lcm.reduce(denominators)
        return b

    def calc_fitness(self, vertices):
        print("in fitness 1")
        print(qhull(vertices))
        integral_points = enumerate_integral_points(qhull(vertices))
        print("1")
        integral_points = integral_points.transpose()
        print("2")
        integral_points = integral_points.astype(int)
        print("3")
        ip_count = len(integral_points)
        print("in fitness 2")
        #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        #print(np.array(vertices))
        distances = self.compute_distances(np.array(vertices))
        result = 0
        if ip_count > 1:
            result -= 1
        for d in distances:
            result -= abs(d-1)
        
        return result

    def apply_crossover(chromosom_1, chromosom_2, crossover_type):
        print("crossover")
        if chromosom_1.get_fitness() > chromosom_2.get_fitness():
            chromosom_1_points = chromosom_1.get_gen_list()
            chromosom_2_points = chromosom_2.get_gen_list()
        else:
            chromosom_2_points = chromosom_1.get_gen_list()
            chromosom_1_points = chromosom_2.get_gen_list()
        
        crossover_type = crossover_type
        print("crossover 1")
        new_chromosom_gen_list = []
        if crossover_type == 1:
            
            if len(chromosom_1_points)>len(chromosom_2_points):
                length_new_chromosom = len(chromosom_2_points)
                random_crossover_point = random.randint(1,len(chromosom_2_points)-1)
            else:
                length_new_chromosom = len(chromosom_2_points)
                random_crossover_point = random.randint(1,len(chromosom_2_points)-1)
            print("crossover 2")
            #new_gen = []
            for i in range(0,length_new_chromosom):
                
                if random.random() < 0.8:
                    if i < random_crossover_point:
                        new_gen = []
                        for j in chromosom_1_points[i]:
                            new_gen.append(j)#[chromosom_1_points[0],chromosom_1_points[1]])
                    else:
                        new_gen = []
                        for j in chromosom_2_points[i]:
                            new_gen.append(j)

                    new_chromosom_gen_list.append(new_gen)
                    #print("random_gen_test")
                else:
                    new_chromosom_gen_list.append(random.sample(range(Generation.min_value_per_point, Generation.max_value_per_point), Generation.num_dimensions))
                    #print("random_gen_test")
            print("crossover 3")
        print(new_chromosom_gen_list)
        new_chromosom = Chromosom(np.array(new_chromosom_gen_list))
        print("crossover 4")
        return new_chromosom
    

#def main():
    
number_of_generations = 100
survival_rate = None
crossover_type = 1
crossover_parent_choosing_type = None
chrom_list = None
num_points_per_polytope = 7
num_chroms_per_Generation = 50
num_dimensions = 5
min_value_per_point = -5
max_value_per_point = 5

test_population = Population(number_of_generations, survival_rate, crossover_type, crossover_parent_choosing_type, chrom_list, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point)


