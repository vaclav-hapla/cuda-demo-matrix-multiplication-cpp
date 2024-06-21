#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "matmult.cuh"

Mat* MatCreateEmpty(int height, int width)
{
    Mat* X = (Mat*)malloc(sizeof(Mat));
    MatInit(X, height, width);
    return X;
}

Mat* MatCreateHost(int height, int width)
{
    Mat* X = MatCreateEmpty(height, width);

    X->elements = (float*)malloc(X->size);
    memset(X->elements, 0, X->size);
    X->elements_malloc = X->elements;
    return X;
}

Mat* MatCreateGPU(int height, int width)
{
    Mat* X = MatCreateEmpty(height, width);

    cudaMalloc(&X->elements, X->size);
    cudaMemset(X->elements, 0, X->size);
    X->elements_cudaMalloc = X->elements;
    return X;
}

void MatFree(Mat** X)
{
    if (!X || !*X)
        return;
    if ((*X)->elements_malloc)
        free((*X)->elements_malloc);
    if ((*X)->elements_cudaMalloc)
        cudaFree((*X)->elements_cudaMalloc);
    free(*X);
    *X = NULL;
}

// Matrix multiplication kernel called by MatMultGPU() - basic version
__global__ void MatMult_naive(Mat A, Mat B, Mat C)
{
    // Each thread computes one element of C
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    MatSetElement(&C, r, c, MatMultElement(&A, &B, r, c));
}

// Matrix multiplication kernel called by MatMultGPU() - optimized version
// Should be run this way:
// size_t sharedMemSize = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float); // Total size for As and Bs
// MatMulKernel<<<gridDim, blockDim, sharedMemSize>>>(A, B, C);
__global__ void MatMult_optimized(Mat MatA, Mat MatB, Mat MatC)
{
    extern __shared__ char sharedMemory[];

    int R = blockIdx.y;
    int C = blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;
    int w = blockDim.x;
    int W = MatA.width / w;

    float* Asubs = (float*)sharedMemory;
    float* Bsubs = (float*)&(sharedMemory[w * w * sizeof(float)]);

    Mat Asub, Bsub, Csub;

    MatGetSubMatrix(&MatC, R, C, w, &Csub);
    // Each thread computes one element of Csub
    float Csub_rc = 0;
    // C_{R,C} = \sum_{K=0}^{W-1} A_{R,K} B_{K,C}
    for (int K = 0; K < W; K++) {
        MatGetSubMatrix(&MatA, R, K, w, &Asub);
        MatGetSubMatrix(&MatB, K, C, w, &Bsub);
        __syncthreads();
        Asubs[r * w + c] = MatGetElement(&Asub, r, c);
        Bsubs[r * w + c] = MatGetElement(&Bsub, r, c);
        __syncthreads();
        // Csub_{r,c} = \sum_{k=0}^{w-1} A_{r,k} B_{k,c}
        for (int k = 0; k < w; k++) {
            Csub_rc += Asubs[r * w + k] * Bsubs[k * w + c];
        }
    }
    MatSetElement(&Csub, r, c, Csub_rc);
}

void MatMultGPU(const Mat* A, const Mat* B, Mat* C, bool optimized)
{
    assert(A->width == B->height);
    assert(A->height == C->height);
    assert(B->width == C->width);

    // Load A to device memory
    Mat* d_A = MatCreateGPU(A->height, A->width);
    cudaMemcpy(d_A->elements, A->elements, A->size, cudaMemcpyHostToDevice);

    // Load B to device memory
    Mat* d_B = MatCreateGPU(B->height, B->width);
    cudaMemcpy(d_B->elements, B->elements, B->size, cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Mat* d_C = MatCreateGPU(C->height, C->width);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B->width / dimBlock.x, A->height / dimBlock.y);
    if (optimized) {
        size_t sharedMemSize = 2 * dimBlock.x * dimBlock.y * sizeof(float); // Total size for As and Bs
        MatMult_optimized<<<dimGrid, dimBlock, sharedMemSize>>>(*d_A, *d_B, *d_C);
    } else {
        MatMult_naive<<<dimGrid, dimBlock>>>(*d_A, *d_B, *d_C);
    }

    // Read C from device memory
    cudaMemcpy(C->elements, d_C->elements, C->size, cudaMemcpyDeviceToHost);

    // Free device memory
    MatFree(&d_A);
    MatFree(&d_B);
    MatFree(&d_C);
}

void MatMultHost(const Mat* A, const Mat* B, Mat* C)
{
    assert(A->width == B->height);
    assert(A->height == C->height);
    assert(B->width == C->width);

    for (int r = 0; r < A->height; r++) {
        for (int c = 0; c < B->width; c++) {
            MatSetElement(C, r, c, MatMultElement(A, B, r, c));
        }
    }
}
void MatPrint(Mat* A, const char name[])
{
    printf("%s = [\n", name);
    for (int i = 0; i < A->height; i++) {
        int j = 0;
        for (; j < A->width - 1; j++) {
            printf("% 5.1f ", MatGetElement(A, i, j));
        }
        printf("% 5.1f\n", MatGetElement(A, i, j));
    }
    printf("]\n");
}

bool MatEqual(Mat* A, Mat* B, float tol)
{
    if (A->height != B->height || A->width != B->width)
        return false;
    for (int r = 0; r < A->height; r++)
        for (int c = 0; c < A->width; c++) {
            if (fabs(MatGetElement(A, r, c) - MatGetElement(B, r, c)) > tol)
                return false;
        }
    return true;
}
