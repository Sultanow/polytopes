#include <stdlib.h>
#include <time.h>
#include <math.h>


#include "../palp/Global.h"
#include "../palp/LG.h"
#include "../palp/Rat.h"
#include "../palp/Subpoly.h"
#include "Definitions.h"

// Function to generate a random long integer in the range [-n, m]
long randomLong(int n, int m) {
    return rand() % (m + n + 1) - n;
}


// Function to swap two rows
void swapRows(long matrix[ROWS][COLS], int row1, int row2, int cols) {
    for (int i = 0; i < cols; i++) {
        long temp = matrix[row1][i];
        matrix[row1][i] = matrix[row2][i];
        matrix[row2][i] = temp;
    }
}

// Function to check if the matrix is full rank
int isFullRank(long matrix[ROWS][COLS], int rows, int cols) {
    for (int i = 0; i < cols; i++) {
        if (matrix[i][i] == 0) {
            int swapRow = -1;
            for (int j = i + 1; j < rows; j++) {
                if (matrix[j][i] != 0) {
                    swapRow = j;
                    break;
                }
            }
            if (swapRow == -1) {
                return 0;
            }
            swapRows(matrix, i, swapRow, cols);
        }
        for (int j = i + 1; j < rows; j++) {
            double ratio = (double)matrix[j][i] / matrix[i][i];
            for (int k = 0; k < cols; k++) {
                matrix[j][k] -= ratio * matrix[i][k];
            }
        }
    }
    return 1;
}

void generateFullRankMatrix(long matrix[ROWS][COLS], int n, int m, int rows, int cols) {
    do {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = randomLong(n, m);
            }
        }
    } while (!isFullRank(matrix, rows, cols));
}


