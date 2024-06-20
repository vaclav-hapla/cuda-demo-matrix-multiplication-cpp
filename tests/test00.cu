#include <assert.h>

#include "matmult.cuh"

// TODO this could be .c not .cu if the API was more isolated from CUDA

void MatFillX(Matrix* X)
{
    for (int r = 0; r < X->height; r++)
        for (int c = 0; c < X->width; c++) {
            SetElement(*X, r, c, (r + c == X->width - 1) ? 1.0 : 0.0);
        }
}

void MatFillY(Matrix* X)
{
    for (int r = 0; r < X->height; r++)
        for (int c = 0; c < X->width; c++) {
            SetElement(*X, r, c, (float)r * X->width + c);
        }
}

int main()
{
    int M = 8, N = 12;

    Matrix X;
    MatInitHost(&X, M, M);
    MatFillX(&X);
    MatPrint(X, "X");

    Matrix Y;
    MatInitHost(&Y, M, N);
    MatFillY(&Y);
    MatPrint(Y, "Y");

    Matrix Z;
    MatInitHost(&Z, M, N);
    MatMultHost(X, Y, Z);
    MatPrint(Z, "Z");

    assert(!MatEqual(Y, Z, 1e-10));

    Matrix Y1;
    MatInitHost(&Y1, M, N);
    MatMultHost(X, Z, Y1);
    assert(MatEqual(Y1, Y, 1e-10));

    Matrix Z_gpu_0;
    MatInitHost(&Z_gpu_0, M, N);
    MatMultGPU(X, Y, Z_gpu_0, false);
    assert(MatEqual(Z_gpu_0, Z, 1e-10));

    Matrix Z_gpu_1;
    MatInitHost(&Z_gpu_1, M, N);
    MatMultGPU(X, Y, Z_gpu_1, true);
    assert(MatEqual(Z_gpu_1, Z, 1e-10));
    return 0;
}
