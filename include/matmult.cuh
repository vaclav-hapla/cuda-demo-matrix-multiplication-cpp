#pragma once

#include <assert.h>
#include <cuda_runtime.h>

#include "matmult.h"

// Thread block size
#define BLOCK_SIZE 4

// Initialize inner matrix fields without allocating its elements
__device__ __host__ inline void MatInit(Mat* X, int height, int width)
{
    X->height = height;
    X->width  = width;
    X->stride = width;
    X->size   = height * width * sizeof(float);

    X->elements            = NULL;
    X->elements_malloc     = NULL;
    X->elements_cudaMalloc = NULL;
}

// Copy a matrix element
__device__ __host__ inline void MatCopyElement(const Mat* A, Mat* B, int r, int c) { MatSetElement(B, r, c, MatGetElement(A, r, c)); }

// Get the blockSize x blockSize submatrix Asub of A,
// located C submatrices to the right and R submatrices down
// from the upper-left corner of A.
__device__ __host__ inline void MatGetSubMatrix(Mat* A, int R, int C, int blockSize, Mat* Asub)
{
    assert(Asub->elements_cudaMalloc == NULL && Asub->elements_malloc == NULL);
    MatInit(Asub, blockSize, blockSize);
    Asub->stride   = A->stride;
    Asub->elements = &(A->elements[R * A->stride * blockSize + C * blockSize]);
}

// Compute a single element C(r,c) of the matrix-matrix product C = A * B
// C_{r,c} = \sum_{k=0}^{w-1} A_{r,k} B_{k,c}
__device__ __host__ inline float MatMultElement(const Mat* A, const Mat* B, int r, int c)
{
    float C_rc = 0;
    for (int k = 0; k < A->width; k++) {
        C_rc += A->elements[r * A->stride + k] * B->elements[k * B->stride + c];
    }
    return C_rc;
}
