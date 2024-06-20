#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

/*
For whole matrix A,
    A.stride = A.width
but for A's submatrix Asub
    Asub.stride = A.stride
    Asub.width  < A.width
Matrices are stored in row-major order:
    M(row, col) = *(M.elements + row * M.stride + col)
*/
typedef struct {
    int    width;
    int    height;
    size_t size;
    int    stride;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 4

// Get a matrix element
__device__ __host__ inline float GetElement(const Matrix A, int r, int c) { return A.elements[r * A.stride + c]; }

// Set a matrix element
__device__ __host__ inline void SetElement(Matrix A, int r, int c, float value) { A.elements[r * A.stride + c] = value; }

// C_{r,c} = \sum_{k=0}^{w-1} A_{r,k} B_{k,c}
__device__ __host__ inline float MatMultElement(const Matrix A, const Matrix B, int r, int c)
{
    float C_rc = 0;
    for (int k = 0; k < A.width; k++) {
        C_rc += A.elements[r * A.stride + k] * B.elements[k * B.stride + c];
    }
    return C_rc;
}

__device__ __host__ void MatInit(Matrix* X, int height, int width);

void MatInitHost(Matrix* X, int height, int width);

void MatInitGPU(Matrix* X, int height, int width);

/*
Get the blockSize x blockSize sub-matrix Asub of A that is
located col sub-matrices to the right and row sub-matrices down
from the upper-left corner of A
*/
__device__ __host__ Matrix GetSubMatrix(Matrix A, int R, int C, int blockSize);

// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMultGPU(const Matrix A, const Matrix B, Matrix C, bool optimized);

void MatMultHost(const Matrix A, const Matrix B, Matrix C);

void MatPrint(Matrix A, const char name[]);

bool MatEqual(Matrix A, Matrix B, float tol);
