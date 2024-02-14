# %%
#pip install polytope


# %%
#pip install pypoman

# %%
# pip install polytope
# pip install pypoman

# import math
# import numpy as np
# from sympy.geometry import Point
# import matplotlib.pyplot as plt
# from polytope import polytope
# from polytope.polytope import enumerate_integral_points, qhull, box2poly
# from scipy.spatial import ConvexHull

# %% [markdown]
# 

# %% [markdown]
# Let's construct a reflexive 2D polytope (taken from the book "The Calabi–Yau Landscape", page 44)

# %%
#V = np.array([[0, -1], [-1, 0], [-1, 2], [1, 0], [1, -1]])


# %% [markdown]
# Pypoman provides a form $0\le -Ax+b$

# %%


# #print(compute_polytope_vertices(A, b))
# print(ConvexHull(V).npoints)

# %% [markdown]
# # GA TEST

# %%
# timeout to catch error of qhull, enumerate_integral_points, ABS_TOL infinity loop
# https://stackoverflow.com/questions/21827874/timeout-a-function-windows

from threading import Thread
import functools

#V= np.array([[-4, 3, 1, -3, 2], [1, 4, -4, -2, -3], [-1,  0,  2, -2,  1], [ 4,  0, -3, -2, -4], [ 2, -1,  0, -2,  1], [-1,  2,  0, -2,  1], [ 3, -3,  4,  1, -2]])

def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print ('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

# %%


# %%
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
from itertools import combinations
import random
from numpy.random import choice



def apply_crossover_parallel(chrom_tuple):
    # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # print("test")
    # #new_chromosom = chrom_tuple
    # print(chrom_tuple)
    # print(chrom_tuple[0].get_chromosom_string())
    # print(chrom_tuple[1].get_chromosom_string())
    # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    chromosom_1 = chrom_tuple[0]
    chromosom_2 = chrom_tuple[1]
    crossover_type = 1# crossover_type

    new_chromosom_gen_list = []
    if crossover_type == 1: ##############to be removed


        if chromosom_1.get_fitness() < chromosom_2.get_fitness():
            chromosom_1_points = chromosom_1.get_gen_list()
            chromosom_2_points = chromosom_2.get_gen_list()
        else:
            chromosom_2_points = chromosom_1.get_gen_list()
            chromosom_1_points = chromosom_2.get_gen_list()

        
        # if len(chromosom_1_points)>len(chromosom_2_points):
        #     length_new_chromosom = len(chromosom_1_points)
        #     random_crossover_point = random.randint(1,len(chromosom_2_points)-1)
        # else:
        #     length_new_chromosom = len(chromosom_2_points)
        #     random_crossover_point = random.randint(1,len(chromosom_2_points)-1)

        new_gen = []
        length_new_chromosom = len(chromosom_1_points)
        random_crossover_point = random.randint(1,len(chromosom_2_points)-1)


        for i in range(0,length_new_chromosom):
            
            if random.random() < 0.8:
                if i < random_crossover_point:
                    new_gen = []
                    #print(chromosom_2_points[i])
                    #new_gen_2 = []
                    # for j in chromosom_1_points[i]:
                    #     #print(j)
                    # if random.random() < 0.8:
                    new_gen = list(chromosom_1_points[i])#[chromosom_1_points[0],chromosom_1_points[1]])
                    # else:
                    #     #j = randint(Generation.min_value_per_point, Generation.max_value_per_point)
                    #     new_gen = [random.randrange(Generation.min_value_per_point, Generation.max_value_per_point, 1) for i in range(Generation.num_dimensions)]

                else:
                    new_gen = []
                    
                    # if random.random() < 0.8:
                    new_gen = list(chromosom_2_points[i])#[chromosom_1_points[0],chromosom_1_points[1]])
                    # else:
                        #j = randint(Generation.min_value_per_point, Generation.max_value_per_point)
                    #   new_gen = [random.randrange(Generation.min_value_per_point, Generation.max_value_per_point, 1) for i in range(Generation.num_dimensions)]
                    # for j in chromosom_2_points[i]:

                        
                    #     if random.random() < 0.8:
                    #         new_gen.append(j)#[chromosom_1_points[0],chromosom_1_points[1]])
                            
                    #     else:
                    #         j = randint(Generation.min_value_per_point, Generation.max_value_per_point)
                    #         new_gen.append(j)

                new_chromosom_gen_list.append(new_gen)
                #print("random_gen_test")
            else:
                new_gen = list(chromosom_1_points[i])#random.sample(range(Generation.min_value_per_point, Generation.max_value_per_point), Generation.num_dimensions)
                #print(new_gen)
                new_chromosom_gen_list.append(new_gen)
                #print("random_gen_test")

    #new_chromosom_gen_list.sort(key=lambda x: int(x[0]))###################################################################################################################
        #print(new_chromosom_gen_list)
        #print("----------------------------------------------------")      
        #print(new_chromosom_gen_list)
        
        new_chromosom = Chromosom(np.array(new_chromosom_gen_list))
        #print(new_chromosom)

    return new_chromosom






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
        #print(1)
        if chrom_list == None:

            self.chrom_list = self.gen_first_generation(num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point)
        else:
            self.chrom_list = chrom_list
        #print("two")
        self.generation_fitness_list = self.set_fitness_value_list(self.chrom_list)
        Population.fitness_evolution.append(self.generation_fitness_list[0])

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
        counter = 0
        #for j in range(0, num_chroms_per_Generation):
        while (len(generation_polytopes)) < num_chroms_per_Generation:
            #chromosome_test_points=[]
            #print(len(generation_polytopes)+len(polytope_points))
            polytope_points = []
            
            # if len(generation_polytopes) == 0:
            #     polytope_points = [[0,-1,1,-1,1],[0,0,0,1,-2],[1,-1,0,0,0],[-1,1,1,1,0],[0,0,1,1,-1],[1,0,-1,-1,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,-2,-1,0]]

            # # #     #polytope_points = [[-2, 3, 1, -3, 0], [1,-2,-1, 2, 0], [ 4,-4, 1, 2,-2], [ 1,2,3,0,-1], [-3,3,-1,-2,2], [ 1,2,3,-3,-1], [-2,-1,-3,0,2]]
            # # #     #polytope_points = [[-1,0,1,1,3],[3,4,-3,9,-5],[-1,-8,1,-15,3],[0,0,1,1,0],[-1,0,1,-7,3],[-1,-1,-1,-3,1]]
            # # #     polytope_points = [[  2,   0,  -2,   2,   3],[  1,  -2,  -2,  -4,   3],[ -4,  14,   1, -11,   0],[  0,   0,   0,   0,   0],[ 11,  -6,   0,  -2,   1],[  4,   1,  -9,  -8,  10],[  1, -12,   1, -12,   0],[  0,   0,  -1,  -6,   2],[ -7, -14,  -2,  -3,   3]]
            # elif(len(generation_polytopes) == 1):
            #     polytope_points = [[ -2, 0, 0, 1,-1], [1, 0, 0,-2,-2], [0, 0, 0, 1, 1], [ -2,-1, 0,-2,-1], [ -3, 2,-1, 5,-2], [1, 0, 0,-1, 0], [0,-1, 1, 1, 3], [6,-1, 1,-5, 1], [-12, 2,-2, 7,-7]]
            # # #     polytope_points = [[10,  18,  -2,  -9,   3], [  9,   4,  -2, -10,   3], [ 29,  28,   1, -10,   0], [  0,   0,   0,   0,   0,], [ -1,  -6,   0,  18,   1], [ 19,  19,  -9,  -4,  10], [ -4, -23,   1, -24,   0], [ 14,  11,  -1,  -7,   2], [-27,  29,  -2, -21,   3]]
            # # # # # #     polytope_points = [[-1,0,2,5,3],[3,4,-3,9,-5],[-1,-10,1,-7,3],[-1,0,0,-11,1],[-1,0,1,1,-1],[-1,0,1,-7,3]]
            # # # # # #     polytope_points = [[4, -2,  3,  2, -3], [ 1, -2, -1,  2,  0], [ 4, -4,  1,  2, -2], [ 1,  2,  3,  0, -1], [-3,  3, -1, -2,  2], [ 1,  2,  3, -3, -1,], [-2, -4, -5,  2,  1]]
            # # # # #     polytope_points = [[-1,0,0,0,-1],[1,0,1,-2,1],[-1,1,0,2,0],[-1,0,-1,3,0],[2,-1,1,-3,1],[-1,1,-1,2,-1],[-1,0,-1,2,0],[1,-1,0,-2,0],[-1,0,-1,1,-1]]

            # # elif(len(generation_polytopes) == 2):
            # #     polytope_points = [[ -2,   0,   0,   1,  -1],[  1,   0,   0,  -2,  -2],[  0,   0,   0,   1,   1],[ -2,  -2,   0,  -3,  -1],[ -3,   3,  -1,   6,  -2],[  1,   0,   0,  -1,   0],[  0,  -1,   1,   1,   3],[  6,  -1,   1,  -5,   1],[-12,   2,  -2,   7,  -7]]
            # # #     polytope_points = [[-1,0,0,0,-1],[1,0,1,0,1],[-1,1,0,2,0], [-1,-1,-1,-1,1], [-1,1,-1,2,-1],[1,-1,0,-1,0],[-1,0,-1,1,0],[2,-1,1,-2,1],[-1,1,0,1,-1]]

            # # # elif(len(generation_polytopes)==3):
            # # #     polytope_points = [[-1,  1,  0,  0,  0], [ 1, -1,  0, -1,  0],[-2,  0,  0,  1,  1],[ 0,  0,  0,  0, -1],[-3,  1, -1,  4, -2],[ 1,  0,  0, -1,  0],[ 0,  0,  0,  1,  1],[ 2, -2,  1, -4,  3],[-6,  2, -1,  6, -1]]

            # # # elif(len(generation_polytopes)==4):
            # # #     polytope_points = [[ -2, 0, 0, 1,-1], [1, 0, 0,-2,-2], [0, 0, 0, 1, 1], [ -2,-1, 0,-2,-1], [ -3, 2,-1, 5,-2], [1, 0, 0,-1, 0], [0,-1, 1, 1, 3], [6,-1, 1,-5, 1], [-12, 2,-2, 7,-7]]

            # else:

            counter += 1
            for i in range(0,num_points_per_polytope):
                polytope_points.append([random.randrange(min_value_per_point, max_value_per_point, 1) for i in range(num_dimensions)])#random.sample(range(min_value_per_point, max_value_per_point), num_dimensions))

            #print(polytope_points)
                
            #polytope_points.sort(key=lambda x: int(x[0])) ################################################################################
            
            chromosome_polytope_points = np.array(polytope_points)
            #print(polytope_points)
            chromosom_test = Chromosom(chromosome_polytope_points)
            #print(chromosom_test.get_is_valid())
            if chromosom_test.get_is_valid():
                generation_polytopes.append(chromosom_test)
        
        generation_polytopes.sort(key=lambda x: x.fitness, reverse=True)
        #print("generation done")
        return generation_polytopes
    
    def normalize(min, max, value):
        
        #new_value = (max-value)/max
        
        normalized_value = (value-min)/(max-min)
        return normalized_value

    def adjust_fitness_values(fitness_value, total_sum):
        propability = (fitness_value / total_sum)

        return float(propability)

class Chromosom:
    chrom_count = 0
    reflexive_politopes = {}
    all_chromosomes = {}

    def __init__(self, points):
        chromosome_polytope_points = points
        #print("pre try")
        #print(Generation.num_dimensions)
        self.is_valid = False
        self.chromosom_string = np.array2string(chromosome_polytope_points).replace("\n", ",")

        if self.chromosom_string in Chromosom.all_chromosomes.keys():
            self.fitness = Chromosom.all_chromosomes[self.chromosom_string]
            Chromosom.chrom_count = Chromosom.chrom_count + 1
            self.gen_list = chromosome_polytope_points
            self.is_valid = True
        else:
            #result = None
            #while result is None:
            try:
                internal_points = []
                #print("###############################################")
                Chromosom.chrom_count = Chromosom.chrom_count + 1
                #print(Chromosom.chrom_count)
                
                self.gen_list = chromosome_polytope_points
                self.fitness = self.calc_fitness(chromosome_polytope_points)
                #print(True)
                self.is_valid = True


                if self.fitness == 0 and self.chromosom_string not in Chromosom.reflexive_politopes.keys(): 
                    print(True)
                    print(repr(self.gen_list))
                    #if np.min(self.gen_list) < 0:
                    self.fitness = np.max(self.gen_list) - np.min(self.gen_list)
                    
                    
                    #self.fitness = np.sum(np.absolute(self.gen_list))
                    
                    
                        #self.fitness = np.max(self.gen_list) + (np.min(self.gen_list) *-1)
                    # elif np.min(self.gen_list) > 0:
                    #     self.fitness = np.max(self.gen_list) - np.min(self.gen_list)   
                    # elif np.min(self.gen_list)==0:
                    #     self.fitness = np.max(self.gen_list)
                    #polytope_volume = ConvexHull(chromosome_polytope_points).volume #qhull(chromosome_polytope_points).volume
                    #Chromosom.reflexive_politopes[polytope_volume] = [ConvexHull(self.gen_list), self.gen_list]
                    Chromosom.reflexive_politopes[self.chromosom_string] = [repr(ConvexHull(self.gen_list)), self.gen_list]
                    #print(Chromosom.reflexive_politopes.keys())
                        
                    Chromosom.all_chromosomes[self.chromosom_string] = self.fitness

                #result = 1
            except:
                self.is_valid = False
    
    def get_chromosom_string(self):
        return self.chromosom_string

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

    @timeout(10)
    def calc_fitness(self, vertices):
        
        # integral_points = enumerate_integral_points(qhull(vertices))
        # integral_points = integral_points.transpose()
        # integral_points = integral_points.astype(int)
        # ip_count = len(integral_points)
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        # print(np.array(vertices))
        distances = self.compute_distances(np.array(vertices))
        result = 0
        # if ip_count > 1:
        #     result -= 1
        for d in distances:
            result -= abs(d-1)
        
        return int(result)

    def apply_crossover(chromosom_1, chromosom_2, crossover_type):
        #print("crossover")
        # chromosom_1 =
        # chromosom_2
        crossover_type = crossover_type

        new_chromosom_gen_list = []
        if crossover_type == 1: ##############to be removed

            if chromosom_1.get_fitness() > chromosom_2.get_fitness():
                chromosom_1_points = chromosom_1.get_gen_list()
                chromosom_2_points = chromosom_2.get_gen_list()
            else:
                chromosom_2_points = chromosom_1.get_gen_list()
                chromosom_1_points = chromosom_2.get_gen_list()

            
            if len(chromosom_1_points)>len(chromosom_2_points):
                length_new_chromosom = len(chromosom_2_points)
                random_crossover_point = random.randint(1,len(chromosom_2_points)-1)
            else:
                length_new_chromosom = len(chromosom_2_points)
                random_crossover_point = random.randint(1,len(chromosom_2_points)-1)

            #new_gen = []
            
            
            for i in range(0,length_new_chromosom):
                
                if random.random() < 0.8:
                    if i < random_crossover_point:
                        new_gen = []
                        new_gen_2 = []
                        for j in chromosom_1_points[i]:
                            
                            if random.random() < 0.8:
                                new_gen.append(j)#[chromosom_1_points[0],chromosom_1_points[1]])
                            else:
                                j = randint(Generation.min_value_per_point, Generation.max_value_per_point)
                                new_gen.append(j)

                    else:
                        new_gen = []
                        for j in chromosom_2_points[i]:

                            
                            if random.random() < 0.8:
                                new_gen.append(j)#[chromosom_1_points[0],chromosom_1_points[1]])
                            else:
                                j = randint(Generation.min_value_per_point, Generation.max_value_per_point)
                                new_gen.append(j)

                    new_chromosom_gen_list.append(new_gen)
                    #print("random_gen_test")
                else:
                    #for i in range(0,Generation.num_points_per_polytope):
                    new_chromosom_gen_list.append([random.randrange(Generation.min_value_per_point, Generation.max_value_per_point, 1) for i in range(Generation.num_dimensions)])#random.sample(range(Generation.min_value_per_point, Generation.max_value_per_point), Generation.num_dimensions))
                    #print("random_gen_test")

        #new_chromosom_gen_list.sort(key=lambda x: int(x[0]))###################################################################################################################

            new_chromosom = Chromosom(np.array(new_chromosom_gen_list))

            return new_chromosom

    # def apply_crossover_parallel(chrom_tuple):
    #     print("test")
    #     #new_chromosom = chrom_tuple
    #     print("crossover")
    #     chromosom_1 =chrom_tuple[0]
    #     chromosom_2 = chrom_tuple[1]
    #     crossover_type = 1# crossover_type

    #     new_chromosom_gen_list = []
    #     if crossover_type == 1: ##############to be removed

    #         if chromosom_1.get_fitness() > chromosom_2.get_fitness():
    #             chromosom_1_points = chromosom_1.get_gen_list()
    #             chromosom_2_points = chromosom_2.get_gen_list()
    #         else:
    #             chromosom_2_points = chromosom_1.get_gen_list()
    #             chromosom_1_points = chromosom_2.get_gen_list()

            
    #         if len(chromosom_1_points)>len(chromosom_2_points):
    #             length_new_chromosom = len(chromosom_2_points)
    #             random_crossover_point = random.randint(1,len(chromosom_2_points)-1)
    #         else:
    #             length_new_chromosom = len(chromosom_2_points)
    #             random_crossover_point = random.randint(1,len(chromosom_2_points)-1)

    #         #new_gen = []
            
            
    #         for i in range(0,length_new_chromosom):
                
    #             if random.random() < 0.8:
    #                 if i < random_crossover_point:
    #                     new_gen = []
    #                     new_gen_2 = []
    #                     for j in chromosom_1_points[i]:
    #                         print(j)
    #                         if random.random() < 0.8:
    #                             new_gen.append(j)#[chromosom_1_points[0],chromosom_1_points[1]])
    #                         else:
    #                             j = randint(Generation.min_value_per_point, Generation.max_value_per_point)
    #                             new_gen.append(j)

    #                 else:
    #                     new_gen = []
    #                     for j in chromosom_2_points[i]:

                            
    #                         if random.random() < 0.8:
    #                             new_gen.append(j)#[chromosom_1_points[0],chromosom_1_points[1]])
                                
    #                         else:
    #                             j = randint(Generation.min_value_per_point, Generation.max_value_per_point)
    #                             new_gen.append(j)

    #                 new_chromosom_gen_list.append(new_gen)
    #                 #print("random_gen_test")
    #             else:
                    
    #                 new_chromosom_gen_list.append(random.sample(range(Generation.min_value_per_point, Generation.max_value_per_point), Generation.num_dimensions))
    #                 #print("random_gen_test")

    #     #new_chromosom_gen_list.sort(key=lambda x: int(x[0]))###################################################################################################################
    #         #print(new_chromosom_gen_list)
    #         new_chromosom = Chromosom(np.array(new_chromosom_gen_list))
    #         print(new_chromosom)

    #     return new_chromosom





    def mut_one(gen_list):

        chrom_points = gen_list
        random_point = random.randint(0 ,Generation.num_points_per_polytope)
        random_axis = random.randint(0,Generation.num_dimensions)
        random_value = random.random()

        if random_value < 0.1:
            chrom_points[random_point-1][random_axis-1] = random.randint(Generation.min_value_per_point, Generation.max_value_per_point) #(-3, 3)
        elif random_value > 0.55:
        #if random.random() > 0.5:
            chrom_points[random_point-1][random_axis-1] = chrom_points[random_point-1][random_axis-1] -1 #random.randint(Generation.min_value_per_point, Generation.max_value_per_point) #(-3, 3)
        else:
            chrom_points[random_point-1][random_axis-1] = chrom_points[random_point-1][random_axis-1] +1

        return chrom_points



    def mut_two(gen_list):
        random_value = random.random()
        chrom_points = gen_list
        random_point = random.randint(0 ,Generation.num_points_per_polytope)
        random_axis = random.randint(0,Generation.num_dimensions)
        # if random.random()<0.1:
        #     chrom_points[random_point-1][random_axis-1] = random.randint(Generation.min_value_per_point, Generation.max_value_per_point) #(-3, 3)
        #elif random.random() > 0.55:
        if random_value > 0.5:
            chrom_points[random_point-1][random_axis-1] = chrom_points[random_point-1][random_axis-1] -1 #random.randint(Generation.min_value_per_point, Generation.max_value_per_point) #(-3, 3)
        else:
            chrom_points[random_point-1][random_axis-1] = chrom_points[random_point-1][random_axis-1] +1

        return chrom_points


    def mut_three(gen_list):
        random_value = random.random()
        chrom_points = gen_list
        random_point = random.randint(0 ,Generation.num_points_per_polytope)
        random_axis = random.randint(0,Generation.num_dimensions)
        # if random.random()<0.1:
        #     chrom_points[random_point-1][random_axis-1] = random.randint(Generation.min_value_per_point, Generation.max_value_per_point) #(-3, 3)
        #elif random.random() > 0.55:
        if random_value <= 0.25:
            chrom_points[random_point-1][random_axis-1] = chrom_points[random_point-1][random_axis-1] -1 #random.randint(Generation.min_value_per_point, Generation.max_value_per_point) #(-3, 3)
        elif random_value > 0.25 and random_value<= 0.5:
            chrom_points[random_point-1][random_axis-1] = chrom_points[random_point-1][random_axis-1] +1
        elif random_value > 0.5 and random_value<= 0.75:
            chrom_points[random_point-1][random_axis-1] = chrom_points[random_point-1][random_axis-1] -2

        elif random_value > 0.75:
            chrom_points[random_point-1][random_axis-1] = chrom_points[random_point-1][random_axis-1] +2

        return chrom_points




    def mutate_Chromosom(self):
        
        chrom_points = copy.deepcopy(self.get_gen_list())
        
        #chrom_points[random_point-1][random_axis-1] = random.randint(Generation.min_value_per_point, Generation.max_value_per_point) #(-3, 3)
        
        for i in range(0,random.randint(1,Generation.num_points_per_polytope)):
            # random_point = random.randint(0 ,Generation.num_points_per_polytope)
            # random_axis = random.randint(0,Generation.num_dimensions)
            # # if random.random()<0.1:
            # #     chrom_points[random_point-1][random_axis-1] = random.randint(Generation.min_value_per_point, Generation.max_value_per_point) #(-3, 3)
            # #elif random.random() > 0.55:
            # if random.random() > 0.5:
            #     chrom_points[random_point-1][random_axis-1] = chrom_points[random_point-1][random_axis-1] -1 #random.randint(Generation.min_value_per_point, Generation.max_value_per_point) #(-3, 3)
            # else:
            #     chrom_points[random_point-1][random_axis-1] = chrom_points[random_point-1][random_axis-1] +1
            chrom_points = Chromosom.mut_one(chrom_points)



        new_chromosom = Chromosom(chrom_points)
        #print(new_chromosom.get_chromosom_string())
        #print("done")
        return new_chromosom

    #def apply_crossover_return_2_children(chromosom_1, chromosom_2, crossover_type):


# 





class Population:
    reduction_intervall = 10
    fitness_evolution = []
    def __init__(self, number_of_generations, survival_rate, crossover_type, crossover_parent_choosing_type, chrom_list, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point):
        #self.generation_0 = Generation(chrom_list, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point)
        #print(True)
        self.number_of_generations = number_of_generations
        self.all_generations = []
        


        self.all_generations.append(Generation(chrom_list, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point))
        #print(True)
        for curr_generation_number in range(0, number_of_generations-1):
            #print(True)
            if curr_generation_number % 100 == 0:
                print("curr Generation count: " + str(curr_generation_number))
                print(self.all_generations[curr_generation_number-1].get_chrom_list()[0].get_fitness())
                #print(repr(self.all_generations[curr_generation_number-1].get_chrom_list()[0].get_chromosom_string()))
            
            new_gen_list = []
            if curr_generation_number % (number_of_generations/Population.reduction_intervall) == 0 and curr_generation_number != 0:
                copy_prev_Generation_chrom_list = copy.deepcopy(self.all_generations[curr_generation_number].get_chrom_list())

                new_gen_list.append(copy.deepcopy(copy_prev_Generation_chrom_list[0]))
                #new_gen_list.append(copy.deepcopy(copy_prev_Generation_chrom_list[0]))
                for i in range(0, len(copy_prev_Generation_chrom_list)-1):
                    if copy_prev_Generation_chrom_list[i].get_chromosom_string() != copy_prev_Generation_chrom_list[i+1].get_chromosom_string():
                        new_gen_list.append(copy.deepcopy(copy_prev_Generation_chrom_list[i+1]))


                
                while len(new_gen_list) < num_chroms_per_Generation:
                    #chromosome_test_points=[]
                    polytope_points = []

                    for i in range(0,num_points_per_polytope):
                        polytope_points.append([random.randrange(min_value_per_point, max_value_per_point, 1) for i in range(num_dimensions)])#random.sample(range(min_value_per_point, max_value_per_point), num_dimensions))
                    
                    #polytope_points.sort(key=lambda x: int(x[0]))###########################################################################################
                    
                    chromosome_polytope_points = np.array(polytope_points)
                    #print(polytope_points)
                    chromosom_test = Chromosom(chromosome_polytope_points)

                    if chromosom_test.get_is_valid():
                        new_gen_list.append(chromosom_test)
                
                new_gen_list.sort(key=lambda x: x.fitness, reverse=True)
                
                print("REDUCTION PERFORMED!!!")
                
                #XXXXXX     .sort(key=lambda x: x.fitness, reverse=True)
                self.all_generations.append(Generation(new_gen_list, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point))


            else:    
                keeper_rate=0.1

                copy_prev_Generation_chrom_list = copy.deepcopy(self.all_generations[curr_generation_number].get_chrom_list())
                
                new_copy_prev_Generation_chrom_list = []
                for i in range(0,int(Generation.num_chroms_per_Generation*keeper_rate)):
                    new_copy_prev_Generation_chrom_list.append(copy_prev_Generation_chrom_list[i])
                

                fitness_list_of_prev_generation = self.all_generations[curr_generation_number].get_fitness_values_list()
                fitness_max = max(fitness_list_of_prev_generation)
                fitness_min = min(fitness_list_of_prev_generation) 
                
                #fitness_list_of_prev_generation = [((fitness_max-x)/fitness_max) for x in fitness_list_of_prev_generation]
                fitness_list_of_prev_generation = [Generation.normalize(min(fitness_list_of_prev_generation), max(fitness_list_of_prev_generation), i) for i in fitness_list_of_prev_generation]
               
                fitness_sum = sum(fitness_list_of_prev_generation)
                
                
                fitness_list_of_prev_generation = [Generation.adjust_fitness_values(i, fitness_sum) for i in fitness_list_of_prev_generation]
                
#                propabilities = list(range((len(copy_prev_Generation_chrom_list)*2), 0, -2))
#                sum_propabilities = sum(propabilities)
                #print(propabilities)
                #fitness_list_of_prev_generation = [Generation.adjust_fitness_values(i,sum_propabilities) for i in propabilities]
                #print(fitness_list_of_prev_generation)
                chromosomes_to_chose = int(int(Generation.num_chroms_per_Generation/2)-(Generation.num_chroms_per_Generation*keeper_rate))
                #print(propabilities)
                #copy_prev_Generation_chrom_list = random.choices()
                copy_prev_Generation_chrom_list = list(choice(copy_prev_Generation_chrom_list, size=chromosomes_to_chose,replace=False, p=fitness_list_of_prev_generation))

                copy_prev_Generation_chrom_list.extend(new_copy_prev_Generation_chrom_list)
                #print(len(copy_prev_Generation_chrom_list))
                # copy_prev_Generation_chrom_list = copy_prev_Generation_chrom_list[:int(len(copy_prev_Generation_chrom_list)/2)]
                
                #starting_length = len(self.all_generations[curr_generation_number].get_chrom_list())
                #parents_pool = list(combinations(copy_prev_Generation_chrom_list, 2))
                
                random.shuffle(copy_prev_Generation_chrom_list)
                #print(copy_prev_Generation_chrom_list)
                parents = copy_prev_Generation_chrom_list
                # if len(parents)%2 !=0:
                #     parents.pop()
                print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
                parents_pool = []
                for i in range(0, int(len(parents)/2)):
                    parents_tuple_list = [parents[i], parents[(len(parents)-i-1)]]
                    parents_pool.append(tuple(parents_tuple_list))
                    
                
                # parents1 = parents[:len(parents)//2]
                # print(len(parents1))
                # parents2 = parents[len(parents)//2:]
                # print(len(parents2))
                # parents_pool = list(zip(parents1, parents2))
                    

                # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                # print(parents_pool[1])
                # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")





#################################################################                

                # for parent_list_element in parents_pool:
                #     #print(parent_list_element)
                #     crossover_chromosom = Chromosom.apply_crossover(parent_list_element[0], parent_list_element[1], crossover_type)
                #     if crossover_chromosom.get_is_valid():
                #         new_gen_list.append(crossover_chromosom)


                #print(parents_pool)
###############################################################
                #from multiprocessing import Pool
                import multiprocessing

                PROCESSES = 6
                
                with multiprocessing.Pool(PROCESSES) as pool:
                    params = parents_pool
                    #print(p for p in params)
                    results = pool.map(apply_crossover_parallel, parents_pool) #[pool.map(apply_crossover_parallel, p) for p in params] #

                #print(results)
                for i in results:
                    if i.get_is_valid():
                        #print(i.get())
                        new_gen_list.append(i)                    

##

                # number_of_processes = 8
                # print(number_of_processes)
                # with Pool(number_of_processes) as p:
                #     results = p.map(self.apply_crossover_parallel, parents_pool)
                # print(results)



##

                # result = []
                # print(1)
                # pool = Pool(processes=2)
                # print(2)
                # res = pool.map_async(self.apply_crossover_parallel, parents_pool)
                # print(3)
                # result.append(res)
                # print(4)
                # print(res)
                # pool.close()
                # print(5)
                # pool.join()
                # print(result)
                # print(6)

################################################################




                parents = []
                while len(new_gen_list) < num_chroms_per_Generation/2: #len(self.all_generations[0].get_chrom_list()):
                    parents = random.sample(copy_prev_Generation_chrom_list, 2)
                    
                    crossover_chromosom = Chromosom.apply_crossover(parents[0], parents[1], crossover_type)
                    
                    if crossover_chromosom.get_is_valid():
                        new_gen_list.append(crossover_chromosom)
                    
                #try:
                num_to_mutate = random.randint(1,int(len(copy_prev_Generation_chrom_list)*0.3))

                number_sample_to_mutate = random.sample(range(0, len(copy_prev_Generation_chrom_list)), num_to_mutate)
                #copy_prev_Generation_chrom_list[:num_to_mutate]
                
                mut_counter = 0
                #mutate best two 
                mutated_chomosom1 = copy_prev_Generation_chrom_list[0].mutate_Chromosom()                
                mutated_chomosom2 = copy_prev_Generation_chrom_list[1].mutate_Chromosom()  

                # mutated_chomosom1 = mutated_chomosom1.mutate_Chromosom()
                # mutated_chomosom2 = mutated_chomosom2.mutate_Chromosom()

                if mutated_chomosom1.get_is_valid():
                    copy_prev_Generation_chrom_list.append(mutated_chomosom1)
                elif (mutated_chomosom2.get_is_valid()):   
                    copy_prev_Generation_chrom_list.append(mutated_chomosom2)
                
                mut_counter = 2

                for i in number_sample_to_mutate:
                    mutated_chomosom = copy_prev_Generation_chrom_list[i]
                    mutated_chomosom = mutated_chomosom.mutate_Chromosom()


                    #if mutated_chomosom.get_fitness() > copy_prev_Generation_chrom_list[i].get_fitness():
                    if mutated_chomosom.get_is_valid():
                        copy_prev_Generation_chrom_list.append(mutated_chomosom)
                    
                        mut_counter = mut_counter + 1



                # except:
                #     pass
                #print("next-Gen xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                #print(starting_length)
                #print("new_gen_list")
                #print(len(new_gen_list))
                #print("copy_prev_Generation_chrom_list")
                #print(len(copy_prev_Generation_chrom_list))
                copy_prev_Generation_chrom_list.extend(new_gen_list)
                copy_prev_Generation_chrom_list.sort(key=lambda x: x.fitness, reverse=True)
                #print("mut_counter")
                #print(mut_counter)
                #print(len(copy_prev_Generation_chrom_list))
                copy_prev_Generation_chrom_list = copy_prev_Generation_chrom_list[:-mut_counter]
                #print(len(copy_prev_Generation_chrom_list))

                self.all_generations.append(Generation(copy_prev_Generation_chrom_list, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point))



    def get_all_generations(self):
        return self.all_generations




# # %%
# # Population Test:

# # test_population = Population(number_of_generations, survival_rate, crossover_type, crossover_parent_choosing_type, chrom_list, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point)
# Population.reduction_intervall = 5 # Generations / reduction intervall

# number_of_generations = 10000
# survival_rate = None
# crossover_type = 1
# crossover_parent_choosing_type = None
# chrom_list = None
# num_points_per_polytope = 13
# num_chroms_per_Generation = 50
# num_dimensions = 5
# min_value_per_point = -3
# max_value_per_point = 3

# test_population = Population(number_of_generations, survival_rate, crossover_type, crossover_parent_choosing_type, chrom_list, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point)

def main():
    # Population Test:

    # test_population = Population(number_of_generations, survival_rate, crossover_type, crossover_parent_choosing_type, chrom_list, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point)
    Population.reduction_intervall = 5 # Generations / reduction intervall

    number_of_generations = 10000
    survival_rate = None
    crossover_type = 1
    crossover_parent_choosing_type = None
    chrom_list = None
    num_points_per_polytope = 13
    num_chroms_per_Generation = 50
    num_dimensions = 5
    min_value_per_point = -3
    max_value_per_point = 3

    test_population = Population(number_of_generations, survival_rate, crossover_type, crossover_parent_choosing_type, chrom_list, num_points_per_polytope, num_chroms_per_Generation, num_dimensions, min_value_per_point, max_value_per_point)



if __name__ == "__main__":
    main()