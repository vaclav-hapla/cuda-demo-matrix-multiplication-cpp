#pragma once

#include <stdbool.h>
#include <stddef.h>

// Dense matrix structure.
// Matrices are stored in row-major order:
//     M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int    width;
    int    height;
    size_t size;
    int    stride;
    float* elements;
    float* elements_malloc;
    float* elements_cudaMalloc;
} Mat;

// Get a matrix element
// We use a macro because it can be reused in a kernel without the need to redefine it with __device__
#define MatGetElement(A, r, c) ((A)->elements[(r) * (A)->stride + (c)])

// Set a matrix element
// We use a macro because it can be reused in a kernel without the need to redefine it with __device__
#define MatSetElement(A, r, c, v) ((A)->elements[(r) * (A)->stride + (c)] = (v))

// Create a matrix without allocating its elements
Mat* MatCreateEmpty(int height, int width);

// Create a matrix with elements allocated on the host
Mat* MatCreateHost(int height, int width);

// Create a matrix with elements allocated on the GPU
Mat* MatCreateGPU(int height, int width);

// Deallocate a matrix including its elements
void MatFree(Mat** X);

// Matrix multiplication A * B = C with naive or optimized GPU kernel
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMultGPU(const Mat* A, const Mat* B, Mat* C, bool optimized);

// Matrix multiplication A * B = C on host
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMultHost(const Mat* A, const Mat* B, Mat* C);

// Print matrix elements
void MatPrint(Mat* A, const char name[]);

// Compare two matrices up to a given tolerance
bool MatEqual(Mat* A, Mat* B, float tol);
