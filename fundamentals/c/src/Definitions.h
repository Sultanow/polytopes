#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
// Define constants
#define POP_SIZE 10       // Population size
#define ROWS 6             // Number of rows in the matrix
#define COLS 5           // Number of columns in the matrix (m)
#define IP_WEIGHT 1
#define DIST_WEIGHT 1
#define MAX_GEN 10000       // Maximum number of generations
#define CROSS_RATE 0.4    // Crossover rate
#define MUTATE_RATE 0.01   // Mutation rate
#define MAXNVERTS 6

#define N 10  // Size of coefficients for which we randomly start
#define M 10

#define SELECT_RATE 0.30 



/*
*  ## Structs for genetic algorithm ##
*/

typedef struct {
    int len;
    int polyrepr[ROWS][COLS]; 
    float fitness; 
    int isReflexive;
    float vol;
} Polytope;

typedef struct {
    int num_points_per_polytope;
    int num_chroms_per_generation;
    int num_dimensions;
    int max_value_per_point;
    int min_value_per_point;
    Polytope polytope_list[POP_SIZE];  // Fixed-length array of Polytopes
    float generation_fitness_list[POP_SIZE];  // Fixed-length array of fitness values
} Generation;


typedef struct {
  int reduction_interval;
  int fitness_evolution[POP_SIZE];
  int number_of_generations;
  int survival_rate;
  int crossover_type;
  int crossover_parent_choosing_type;
  int num_points_per_polytope;
  int num_polys_per_gen;
  int num_dimensions;
  int min_value_per_point;
  int max_value_per_point;
  Generation current_generation;
  Generation previous_generation;
} Population;



/*  new shi*/
void create_generation(Generation* gen, Polytope* polytope_list, int num_points_per_polytope, int num_chroms_per_generation, int num_dimensions, int min_value_per_point, int max_value_per_point);
void set_fitness_value_list(Generation* gen);
void gen_first_generation(Generation* gen, int num_points_per_polytope, int num_chroms_per_generation, int num_dimensions);
bool is_full_rank(int matrix[ROWS][COLS], int num_dimensions);
float normalize(float min, float max, float value);
float adjust_fitness_values(float fitness_value, float total_sum);
float fitness_function(int matrix[ROWS][COLS]);
void create_generation(Generation* gen, Polytope* polytope_list, int num_points_per_polytope, int num_chroms_per_generation, int num_dimensions, int min_value_per_point, int max_value_per_point);
void gen_first_generation(Generation* gen, int num_points_per_polytope, int num_chroms_per_generation, int num_dimensions);
//void generate_new_generation(Population* pop, int curr_generation_number);
Polytope mutate_chromosom(Polytope chrom);
void initialize_population(Population* pop, int number_of_generations, double survival_rate, int crossover_type, int crossover_parent_choosing_type, Polytope* chrom_list, int num_points_per_polytope, int num_chroms_per_generation, int num_dimensions, int min_value_per_point, int max_value_per_point);
void generate_new_generation(Population* pop);

void print_matrix(int matrix[ROWS][COLS]);


/* old shi */

void generate_random_full_rank_matrix(int A[ROWS][COLS]);
// Function prototypes

long randomLong(int n, int m);
void swapRows(long matrix[ROWS][COLS], int row1, int row2, int cols);
void swapRows(long matrix[ROWS][COLS], int row1, int row2, int cols);
void generateFullRankMatrix(long matrix[ROWS][COLS], int n, int m, int rows, int cols);
int dimensionOfPolytope(int R, int C, int matrix[R][C]);
int rankOfMatrix(int R, int C, int mat[R][C]);
//void initialize_population(long population[POP_SIZE][ROWS][COLS]);
void evaluate_population(long population[POP_SIZE][ROWS][COLS], double fitness[POP_SIZE]);
void select_parents(long population[POP_SIZE][ROWS][COLS], long parents[POP_SIZE][ROWS][COLS], double fitness[POP_SIZE]);
void crossover(long parents[POP_SIZE][ROWS][COLS], long offspring[POP_SIZE][ROWS][COLS]);
void mutate(long offspring[POP_SIZE][ROWS][COLS]);
void copy_population(long source[POP_SIZE][ROWS][COLS], long destination[POP_SIZE][ROWS][COLS], int num);
double get_best_fitness(double fitness[POP_SIZE], int *best_index);

int compare(const void *a, const void *b);

void select_fittest(long population[POP_SIZE][ROWS][COLS], long fittest_population[(int)(POP_SIZE * SELECT_RATE)][ROWS][COLS], double fitness[POP_SIZE]);
void mutate_and_repopulate(long fittest_population[(int)(POP_SIZE * SELECT_RATE)][ROWS][COLS], long population[POP_SIZE][ROWS][COLS]);
