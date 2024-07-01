#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../palp/Global.h"
#include "../palp/LG.h"
#include "../palp/Rat.h"
#include "../palp/Subpoly.h"
#include "Definitions.h"



FILE *inFILE, *outFILE;


int main() {
// Seed the random number generator
    srand(time(NULL));

    // Example values for initialization
    int number_of_generations = 10;
    double survival_rate = 0.3;
    int crossover_type = 1;
    int crossover_parent_choosing_type = 1;
    int num_points_per_polytope = 30;
    int num_chroms_per_generation = POP_SIZE;
    int num_dimensions = 5;
    int min_value_per_point = -10;
    int max_value_per_point = 10;

    // Create initial chromosome list (assuming a function to initialize this list exists)
    Polytope chrom_list[POP_SIZE];
    // Create the generation
    Generation gen;

    // Initialize population
    Population pop;
    pop.current_generation = gen;
    initialize_population(&pop, number_of_generations, survival_rate, crossover_type, crossover_parent_choosing_type, chrom_list, num_points_per_polytope, num_chroms_per_generation, num_dimensions, min_value_per_point, max_value_per_point);
// Print fitness values for the first generation
    printf("Fitness values for the initial generation:\n");
    for (int i = 0; i < num_chroms_per_generation; i++) {
        printf("Polytope %d: %f\n", i, pop.current_generation.generation_fitness_list[i]);
    }
    return 0;
 //// Seed the random number generator
 //   srand(time(NULL));



 //   // Example values
 //   int num_points_per_polytope = 30;
 //   int num_chroms_per_generation = POP_SIZE;
 //   int num_dimensions = 5;
 //   int min_value_per_point = -10;
 //   int max_value_per_point = 10;


 //   

 //   return 0;

 //   
 //  
}

void print_matrix(int matrix[ROWS][COLS]) {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%ld ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
