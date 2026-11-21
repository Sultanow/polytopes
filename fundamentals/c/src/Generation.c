#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


#include "../palp/Global.h"
#include "../palp/LG.h"
#include "../palp/Rat.h"
#include "../palp/Subpoly.h"
#include "Definitions.h"


void create_generation(Generation* gen, Polytope* polytope_list, int num_points_per_polytope, int num_chroms_per_generation, int num_dimensions, int min_value_per_point, int max_value_per_point) {
    gen->num_points_per_polytope = num_points_per_polytope;
    gen->num_chroms_per_generation = num_chroms_per_generation;
    gen->num_dimensions = num_dimensions;
    gen->min_value_per_point = min_value_per_point;
    gen->max_value_per_point = max_value_per_point;

    if (polytope_list == NULL) {
        gen_first_generation(gen, num_points_per_polytope, num_chroms_per_generation, num_dimensions);
    } else {
        for (int i = 0; i < POP_SIZE; i++) {
            gen->polytope_list[i] = polytope_list[i];
        }
    }

    set_fitness_value_list(gen);
    // Population.fitness_evolution.append(gen->generation_fitness_list[0]);  // Handle this appropriately
}


void set_fitness_value_list(Generation* gen) {
    for (int i = 0; i < POP_SIZE; i++) {
        gen->generation_fitness_list[i] = fitness_function(gen->polytope_list[i].polyrepr);
    }
}

void gen_first_generation(Generation* gen, int num_points_per_polytope, int num_chroms_per_generation, int num_dimensions) {
    int counter = 0;
    gen->num_chroms_per_generation = num_chroms_per_generation;

    while (counter < num_chroms_per_generation) {
        int polytope_points[ROWS][COLS] = {0};

        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                polytope_points[i][j] = (rand() % (M + N + 1)) - N;
            }
        }

        if (is_full_rank(polytope_points, num_dimensions)) {
            Polytope chromosom_test;
            // Set the matrix and other properties of chromosom_test
            for (int i = 0; i < ROWS; i++) {
                for (int j = 0; j < COLS; j++) {
                    chromosom_test.polyrepr[i][j] = polytope_points[i][j];
                }
            }
            gen->polytope_list[counter] = chromosom_test;
            gen->polytope_list[counter].fitness = fitness_function(chromosom_test.polyrepr);
            counter++;
        }
    }


// Sort the generation_polytopes by fitness in descending order
    qsort(gen->polytope_list, num_chroms_per_generation, sizeof(Polytope), compare);
}

bool is_full_rank(int matrix[ROWS][COLS], int num_dimensions) {
    int temp_matrix[ROWS][COLS];
    int i, j, k;

    // Copy the matrix to a temporary matrix
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            temp_matrix[i][j] = matrix[i][j];
        }
    }

    // Perform Gaussian elimination
    for (i = 0; i < num_dimensions; i++) {
        // Find the pivot element
        int pivot_row = i;
        for (j = i + 1; j < ROWS; j++) {
            if (abs(temp_matrix[j][i]) > abs(temp_matrix[pivot_row][i])) {
                pivot_row = j;
            }
        }

        // Swap the pivot row with the current row
        if (pivot_row != i) {
            for (j = 0; j < COLS; j++) {
                int temp = temp_matrix[i][j];
                temp_matrix[i][j] = temp_matrix[pivot_row][j];
                temp_matrix[pivot_row][j] = temp;
            }
        }

        // Check if the pivot element is zero
        if (temp_matrix[i][i] == 0) {
            return false;
        }

        // Normalize the pivot row
        for (j = i + 1; j < COLS; j++) {
            temp_matrix[i][j] /= temp_matrix[i][i];
        }
        temp_matrix[i][i] = 1;

        // Eliminate the other rows
        for (k = 0; k < ROWS; k++) {
            if (k != i) {
                int factor = temp_matrix[k][i];
                for (j = i; j < COLS; j++) {
                    temp_matrix[k][j] -= factor * temp_matrix[i][j];
                }
                temp_matrix[k][i] = 0;
            }
        }
    }

    // If we reached here, the matrix is of full rank
    return true;
}



float normalize(float min, float max, float value) {
    return (value - min) / (max - min);
}

float adjust_fitness_values(float fitness_value, float total_sum) {
    return fitness_value / total_sum;
}















void crossover(long parents[POP_SIZE][ROWS][COLS], long offspring[POP_SIZE][ROWS][COLS]) {
    for (int i = 0; i < POP_SIZE; i += 2) {
        if ((double)rand() / RAND_MAX < CROSS_RATE) {
            int crossover_point = rand() % (ROWS * COLS);
            for (int j = 0; j < ROWS; j++) {
                for (int k = 0; k < COLS; k++) {
                    int index = j * COLS + k;
                    if (index < crossover_point) {
                        offspring[i][j][k] = parents[i][j][k];
                        offspring[i + 1][j][k] = parents[i + 1][j][k];
                    } else {
                        offspring[i][j][k] = parents[i + 1][j][k];
                        offspring[i + 1][j][k] = parents[i][j][k];
                    }
                }
            }
        } else {
            for (int j = 0; j < ROWS; j++) {
                for (int k = 0; k < COLS; k++) {
                    offspring[i][j][k] = parents[i][j][k];
                    offspring[i + 1][j][k] = parents[i + 1][j][k];
                }
            }
        }
    }
}

void mutate(long offspring[POP_SIZE][ROWS][COLS]) {
    for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < ROWS; j++) {
            for (int k = 0; k < COLS; k++) {
                if ((double)rand() / RAND_MAX < MUTATE_RATE) {
                    offspring[i][j][k] = rand() % 21 - 10; // Random mutation to a value between -10 and 10
                }
            }
        }
    }
}

