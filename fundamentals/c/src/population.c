#include <stdlib.h>
#include <time.h>
#include <math.h>


#include "../palp/Global.h"
#include "../palp/LG.h"
#include "../palp/Rat.h"
#include "../palp/Subpoly.h"
#include "Definitions.h"

// Function to initialize the population
void initialize_population(Population* pop, int number_of_generations, double survival_rate, int crossover_type, int crossover_parent_choosing_type, Polytope* chrom_list, int num_points_per_polytope, int num_chroms_per_generation, int num_dimensions, int min_value_per_point, int max_value_per_point) {
    pop->number_of_generations = number_of_generations;
    pop->survival_rate = survival_rate;
    pop->crossover_type = crossover_type;
    pop->crossover_parent_choosing_type = crossover_parent_choosing_type;


    // Initialize the first generation
    create_generation(&pop->current_generation,NULL, num_points_per_polytope, num_chroms_per_generation, num_dimensions, min_value_per_point, max_value_per_point);

    // Print the initial generation's fitness values
    printf("Initial Generation fitness values:\n");
    for (int i = 0; i < num_chroms_per_generation; i++) {
        printf("Polytope %d: %f\n", i, pop->current_generation.generation_fitness_list[i]);
    }

    // Generate subsequent generations
    for (int curr_generation_number = 1; curr_generation_number < number_of_generations; curr_generation_number++) {
        // Print generation progress
   

        // Generate new generation
        generate_new_generation(pop);

        // Print the current generation's fitness values
        printf("Generation %d fitness values:\n", curr_generation_number);
        for (int i = 0; i < num_chroms_per_generation; i++) {
            printf("Polytope %d: %f\n", i, pop->current_generation.generation_fitness_list[i]);
        }
    }
}

void generate_new_generation(Population* pop) {
    int num_points_per_polytope = pop->current_generation.num_points_per_polytope;
    int num_chroms_per_generation = pop->current_generation.num_chroms_per_generation;
    int num_dimensions = pop->current_generation.num_dimensions;
    int min_value_per_point = pop->current_generation.min_value_per_point;
    int max_value_per_point = pop->current_generation.max_value_per_point;
    // Swap current and previous generations
    pop->previous_generation = pop->current_generation;

    Polytope new_gen_list[POP_SIZE];
    int new_gen_count = 0;
    
    // Keep the fittest 30%
    int num_to_keep = (int)(0.3 * num_chroms_per_generation);

    for (int i = 0; i < num_to_keep; i++) {
        new_gen_list[new_gen_count++] = pop->previous_generation.polytope_list[i];
    }

    // Mutate the rest and fill up the new generation
    while (new_gen_count < num_chroms_per_generation) {
        int idx = rand() % num_to_keep;
        Polytope mutated_chromosom = mutate_chromosom(new_gen_list[idx]);
        if (is_full_rank(mutated_chromosom.polyrepr, num_dimensions)) {
            mutated_chromosom.fitness = fitness_function(mutated_chromosom.polyrepr);
            new_gen_list[new_gen_count++] = mutated_chromosom;
        }
    }

    // Copy new generation to current generation
    for (int i = 0; i < num_chroms_per_generation; i++) {
        pop->current_generation.polytope_list[i] = new_gen_list[i];
    }
    // Update the fitness values list after sorting
    set_fitness_value_list(&pop->current_generation);
}
//// Function to initialize the population
//void initialize_population(Population* pop, int number_of_generations, double survival_rate, int crossover_type, int crossover_parent_choosing_type, Polytope* chrom_list, int num_points_per_polytope, int num_chroms_per_generation, int num_dimensions, int min_value_per_point, int max_value_per_point) {
//    pop->number_of_generations = number_of_generations;
//    pop->survival_rate = survival_rate;
//    pop->crossover_type = crossover_type;
//    pop->crossover_parent_choosing_type = crossover_parent_choosing_type;
//    pop->all_generations = (Generation*)malloc(number_of_generations * sizeof(Generation));
//
//    // Initialize the first generation
//    create_generation(&pop->all_generations[0], chrom_list, num_points_per_polytope, num_chroms_per_generation, num_dimensions, min_value_per_point, max_value_per_point);
//
//    // Loop to create subsequent generations
//    for (int curr_generation_number = 1; curr_generation_number < number_of_generations; curr_generation_number++) {
//        // Print generation progress
//        if (curr_generation_number % 100 == 0) {
//            printf("Current Generation count: %d\n", curr_generation_number);
//            printf("Top fitness: %f\n", pop->all_generations[curr_generation_number - 1].generation_fitness_list[0]);
//        }
//
//        // Generate new generation
//        generate_new_generation(pop, curr_generation_number);
//    }
//}
//

//void generate_new_generation(Population* pop, int curr_generation_number) {
//    int num_points_per_polytope = pop->all_generations[0].num_points_per_polytope;
//    int num_chroms_per_generation = pop->all_generations[0].num_chroms_per_generation;
//    int num_dimensions = pop->all_generations[0].num_dimensions;
//    int min_value_per_point = pop->all_generations[0].min_value_per_point;
//    int max_value_per_point = pop->all_generations[0].max_value_per_point;
//
//    Generation* curr_generation = &pop->all_generations[curr_generation_number];
//    Generation* prev_generation = &pop->all_generations[curr_generation_number - 1];
//    
//    Polytope new_gen_list[POP_SIZE];
//    int new_gen_count = 0;
//    
//    // Keep the fittest 30%
//    int num_to_keep = (int)(0.3 * num_chroms_per_generation);
//    for (int i = 0; i < num_to_keep; i++) {
//        new_gen_list[new_gen_count++] = prev_generation->polytope_list[i];
//    }
//
//    // Mutate the rest and fill up the new generation
//    while (new_gen_count < num_chroms_per_generation) {
//        int idx = rand() % num_to_keep;
//        Polytope mutated_chromosom = mutate_chromosom(new_gen_list[idx]);
//        if (is_full_rank(mutated_chromosom.polyrepr, num_dimensions)) {
//            mutated_chromosom.fitness = fitness_function(mutated_chromosom.polyrepr);
//            new_gen_list[new_gen_count++] = mutated_chromosom;
//        }
//    }
//
//    // Copy new generation to current generation
//    for (int i = 0; i < num_chroms_per_generation; i++) {
//        curr_generation->polytope_list[i] = new_gen_list[i];
//    }
//
//    // Update the fitness values list after sorting
//    set_fitness_value_list(curr_generation);
//}

Polytope mutate_chromosom(Polytope chrom) {
    // Mutate one random element of the matrix
    int row = rand() % ROWS;
    int col = rand() % COLS;
    chrom.polyrepr[row][col] = (rand() % (M + N + 1)) - N;
    return chrom;
}




//void create_generation(Generation* gen, Polytope* polytope_list, int num_points_per_polytope, int num_chroms_per_generation, int num_dimensions, int min_value_per_point, int max_value_per_point) {
//    gen->num_points_per_polytope = num_points_per_polytope;
//    gen->num_chroms_per_generation = num_chroms_per_generation;
//    gen->num_dimensions = num_dimensions;
//    gen->min_value_per_point = min_value_per_point;
//    gen->max_value_per_point = max_value_per_point;
//
//    if (polytope_list == NULL) {
//        gen_first_generation(gen, num_points_per_polytope, num_chroms_per_generation, num_dimensions);
//    } else {
//        for (int i = 0; i < POP_SIZE; i++) {
//            gen->polytope_list[i] = polytope_list[i];
//        }
//    }
//
//    set_fitness_value_list(gen);
//}
//
//
//
//void generate_new_generation(Population* pop, int curr_generation_number) {
//    int num_points_per_polytope = pop->all_generations[0].num_points_per_polytope;
//    int num_chroms_per_generation = pop->all_generations[0].num_chroms_per_generation;
//    int num_dimensions = pop->all_generations[0].num_dimensions;
//    int min_value_per_point = pop->all_generations[0].min_value_per_point;
//    int max_value_per_point = pop->all_generations[0].max_value_per_point;
//
//    Generation* curr_generation = &pop->all_generations[curr_generation_number];
//    Generation* prev_generation = &pop->all_generations[curr_generation_number - 1];
//    
//    Polytope new_gen_list[POP_SIZE];
//    int new_gen_count = 0;
//    
//    // Check if reduction interval condition is met
//    if (curr_generation_number % (pop->number_of_generations / 10) == 0 && curr_generation_number != 0) {
//        for (int i = 0; i < num_chroms_per_generation; i++) {
//            new_gen_list[i] = prev_generation->polytope_list[i];
//        }
//        new_gen_list[0] = prev_generation->polytope_list[0];
//        new_gen_count = 1;
//
//        for (int i = 1; i < num_chroms_per_generation - 1; i++) {
//            if (memcmp(prev_generation->polytope_list[i].polyrepr, prev_generation->polytope_list[i+1].polyrepr, sizeof(int) * ROWS * COLS) != 0) {
//                new_gen_list[new_gen_count] = prev_generation->polytope_list[i+1];
//                new_gen_count++;
//            }
//        }
//
//        while (new_gen_count < num_chroms_per_generation) {
//            Polytope new_polytope;
//            for (int i = 0; i < ROWS; i++) {
//                for (int j = 0; j < COLS; j++) {
//                    new_polytope.polyrepr[i][j] = (rand() % (max_value_per_point - min_value_per_point + 1)) + min_value_per_point;
//                }
//            }
//
//            if (is_full_rank(new_polytope.polyrepr, num_dimensions)) {
//                new_polytope.fitness = fitness_function(new_polytope.polyrepr);
//                new_gen_list[new_gen_count] = new_polytope;
//                new_gen_count++;
//            }
//        }
//
//        qsort(new_gen_list, num_chroms_per_generation, sizeof(Polytope), compare_fitness);
//        printf("REDUCTION PERFORMED!!!\n");
//    } else {
//        int keeper_rate = 0.1 * num_chroms_per_generation;
//        Polytope keeper_list[keeper_rate];
//        memcpy(keeper_list, prev_generation->polytope_list, keeper_rate * sizeof(Polytope));
//        
//        float fitness_list[POP_SIZE];
//        float total_fitness = 0.0;
//
//        for (int i = 0; i < num_chroms_per_generation; i++) {
//            fitness_list[i] = normalize(prev_generation->generation_fitness_list[0], prev_generation->generation_fitness_list[num_chroms_per_generation-1], prev_generation->generation_fitness_list[i]);
//            total_fitness += fitness_list[i];
//        }
//
//        for (int i = 0; i < num_chroms_per_generation; i++) {
//            fitness_list[i] = adjust_fitness_values(fitness_list[i], total_fitness);
//        }
//
//        Polytope selected_list[num_chroms_per_generation / 2];
//        for (int i = 0; i < num_chroms_per_generation / 2 - keeper_rate; i++) {
//            int index = 0;
//            float rand_val = (float)rand() / RAND_MAX;
//            while (rand_val > 0) {
//                rand_val -= fitness_list[index];
//                index++;
//            }
//            selected_list[i] = prev_generation->polytope_list[index - 1];
//        }
//
//        memcpy(selected_list + (num_chroms_per_generation / 2 - keeper_rate), keeper_list, keeper_rate * sizeof(Polytope));
//        for (int i = 0; i < num_chroms_per_generation / 2; i++) {
//            int idx = rand() % (num_chroms_per_generation / 2);
//            Polytope temp = selected_list[i];
//            selected_list[i] = selected_list[idx];
//            selected_list[idx] = temp;
//        }
//
//        Polytope parents_pool[num_chroms_per_generation / 2][2];
//        for (int i = 0; i < num_chroms_per_generation / 2; i++) {
//            parents_pool[i][0] = selected_list[i];
//            parents_pool[i][1] = selected_list[num_chroms_per_generation / 2 - i - 1];
//        }
//
//        for (int i = 0; i < num_chroms_per_generation / 2; i++) {
//            Polytope crossover_chromosom = apply_crossover(parents_pool[i][0], parents_pool[i][1], pop->crossover_type);
//            if (is_full_rank(crossover_chromosom.polyrepr, num_dimensions)) {
//                crossover_chromosom.fitness = fitness_function(crossover_chromosom.polyrepr);
//                new_gen_list[new_gen_count] = crossover_chromosom;
//                new_gen_count++;
//            }
//        }
//
//        while (new_gen_count < num_chroms_per_generation) {
//            int idx1 = rand() % (num_chroms_per_generation / 2);
//            int idx2 = rand() % (num_chroms_per_generation / 2);
//            Polytope crossover_chromosom = apply_crossover(selected_list[idx1], selected_list[idx2], pop->crossover_type);
//            if (is_full_rank(crossover_chromosom.polyrepr, num_dimensions)) {
//                crossover_chromosom.fitness = fitness_function(crossover_chromosom.polyrepr);
//                new_gen_list[new_gen_count] = crossover_chromosom;
//                new_gen_count++;
//            }
//        }
//
//        int num_to_mutate = (rand() % ((int)(0.3 * num_chroms_per_generation))) + 1;
//        for (int i = 0; i < num_to_mutate; i++) {
//            int idx = rand() % num_chroms_per_generation;
//            Polytope mutated_chromosom = mutate_chromosom(new_gen_list[idx]);
//            if (is_full_rank(mutated_chromosom.polyrepr, num_dimensions)) {
//                mutated_chromosom.fitness = fitness_function(mutated_chromosom.polyrepr);
//                new_gen_list[idx] = mutated_chromosom;
//            }
//        }
//
//        qsort(new_gen_list, num_chroms_per_generation, sizeof(Polytope), compare_fitness);
//        for (int i = 0; i < num_chroms_per_generation; i++) {
//            curr_generation->polytope_list[i] = new_gen_list[i];
//        }
//    }
//
//    // Update the fitness values list after sorting
//    set_fitness_value_list(curr_generation);
//}
//




//void initialize_population(long population[POP_SIZE][ROWS][COLS]) {
//    for (int i = 0; i < POP_SIZE; i++) {         
//      generateFullRankMatrix(population[i], N, M, ROWS, COLS);
//    }
//}

//void evaluate_population(long population[POP_SIZE][ROWS][COLS], double fitness[POP_SIZE]) {
//    for (int i = 0; i < POP_SIZE; i++) {
//        fitness[i] = fitness_function(population[i]);
//    }
//}

void select_fittest(long population[POP_SIZE][ROWS][COLS], long fittest_population[(int)(POP_SIZE * SELECT_RATE)][ROWS][COLS], double fitness[POP_SIZE]) {
    int indices[POP_SIZE];
    for (int i = 0; i < POP_SIZE; i++) {
        indices[i] = i;
    }

    // Sort indices based on fitness in descending order
    qsort(indices, POP_SIZE, sizeof(int), compare);

    // Select the fittest individuals
    for (int i = 0; i < (int)(POP_SIZE * SELECT_RATE); i++) {
        for (int j = 0; j < ROWS; j++) {
            for (int k = 0; k < COLS; k++) {
                fittest_population[i][j][k] = population[indices[i]][j][k];
            }
        }
    }
}
void select_parents(long population[POP_SIZE][ROWS][COLS], long parents[POP_SIZE][ROWS][COLS], double fitness[POP_SIZE]) {
    // Simple roulette wheel selection
    double total_fitness = 0.0;
    for (int i = 0; i < POP_SIZE; i++) {
        total_fitness += fitness[i];
    }

    for (int i = 0; i < POP_SIZE; i++) {
        double rand_val = ((double)rand() / RAND_MAX) * total_fitness;
        double running_sum = 0.0;
        for (int j = 0; j < POP_SIZE; j++) {
            running_sum += fitness[j];
            if (running_sum > rand_val) {
                for (int k = 0; k < ROWS; k++) {
                    for (int l = 0; l < COLS; l++) {
                        parents[i][k][l] = population[j][k][l];
                    }
                }
                break;
            }
        }
    }
}

void copy_population(long source[POP_SIZE][ROWS][COLS], long destination[POP_SIZE][ROWS][COLS], int num) {
    for (int i = 0; i < num; i++) {
        for (int j = 0; j < ROWS; j++) {
            for (int k = 0; k < COLS; k++) {
                destination[i][j][k] = source[i][j][k];
            }
        }
    }
}

void mutate_and_repopulate(long fittest_population[(int)(POP_SIZE * SELECT_RATE)][ROWS][COLS], long population[POP_SIZE][ROWS][COLS]) {
    int num_fittest = (int)(POP_SIZE * SELECT_RATE);
    // Copy the fittest individuals directly into the new population
    copy_population(fittest_population, population, num_fittest);

    // Fill the rest of the population by mutating the fittest individuals
    for (int i = num_fittest; i < POP_SIZE; i++) {
        int parent_index = rand() % num_fittest;
        for (int j = 0; j < ROWS; j++) {
            for (int k = 0; k < COLS; k++) {
                population[i][j][k] = fittest_population[parent_index][j][k];
                if ((double)rand() / RAND_MAX < MUTATE_RATE) {
                    population[i][j][k] = rand() % 21 - 10; // Random mutation to a value between -10 and 10
                }
            }
        }
    }
}



int compare(const void* a, const void* b) {
    Polytope* polytopeA = (Polytope*)a;
    Polytope* polytopeB = (Polytope*)b;

    if (polytopeA->fitness > polytopeB->fitness) return -1; // For descending order
    if (polytopeA->fitness < polytopeB->fitness) return 1;
    return 0;
}
