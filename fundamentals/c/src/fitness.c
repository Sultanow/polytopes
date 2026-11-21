#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#include "../palp/Global.h"
#include "../palp/LG.h"
#include "../palp/Rat.h"
#include "../palp/Subpoly.h"
#include "Definitions.h"

float adjust_fitness_value(double fitness_value, double tototal_sum);


float adjust_fitness_value(double fitness_value, double tototal_sum){
  float propability;
  propability = (fitness_value/tototal_sum);
  return propability;
}

float fitness_function(int matrix[ROWS][COLS]) {
    int dim;
    float score;
    int i, j, IP;
    float totalDist, avDist;
    VertexNumList V;
    EqList *E = (EqList *) malloc(sizeof(EqList));
    PolyPointList *_P = (PolyPointList *) malloc(sizeof(PolyPointList));
    PairMat *PM = (PairMat *) malloc(sizeof(PairMat));
    BaHo BH;
    FaceInfo *FI = (FaceInfo *) malloc(sizeof(FaceInfo));

    // define the polytope dimension
    _P->n = POLYDIM; 

    // define the number of points
    _P->np = ROWS; 

    for(i = 0; i < ROWS; i++) {
        for(j = 0; j < COLS; j++) {
            _P->x[i][j] = (Long) matrix[i][j];
        }
    }

    // find the bounding hyperplane equations of the polytope
    IP = Find_Equations(_P, &V, E); 

    // compute the fitness score
    score = 0.0;
    // penalty for the IP property
    if(IP_WEIGHT > 0) score += IP_WEIGHT * (IP - 1);
    // penalty for the distance of facets from the origin
    totalDist = 0.0;
    for(i = 0; i < E->ne; i++) {
        totalDist += llabs(E->e[i].c - 1);
    }
    avDist = totalDist / E->ne;
    score += -DIST_WEIGHT * avDist / (31 * POLYDIM);

    free(E);
    free(_P);
    free(PM);
    free(FI);

    return score; 
}

double get_best_fitness(double fitness[POP_SIZE], int *best_index) {
    double best_fitness = fitness[0];
    *best_index = 0;
    for (int i = 0; i < POP_SIZE; i++) {
        if (fitness[i] > best_fitness) {
            best_fitness = fitness[i];
            *best_index = i;
        }
    }
    //printf("%d \n", *best_index);
   // printf("%f ", best_fitness);
    return best_fitness;
}

