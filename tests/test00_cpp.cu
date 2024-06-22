#include <cassert>
#include <iostream>

#include "matmult_cpp.cuh"

// TODO this could be .c not .cu if the API was more isolated from CUDA

void MatFillX(Matrix& X)
{
    auto height = X.getHeight();
    auto width  = X.getWidth();
    for (int r = 0; r < height; r++)
        for (int c = 0; c < width; c++) {
            X.setElement(r, c, (r + c == width - 1) ? 1.0 : 0.0);
        }
}

void MatFillY(Matrix& X)
{
    auto height = X.getHeight();
    auto width  = X.getWidth();
    for (int r = 0; r < height; r++)
        for (int c = 0; c < width; c++) {
            X.setElement(r, c, (float)r * width + c + 1);
        }
}

int main()
{
    int M = 8, N = 12;

    Matrix X(M, M);
    MatFillX(X);
    std::cout << "X: " << X << std::endl;

    {
        Matrix X_copy = X;
        assert(X.getElements() == X_copy.getElements());
    }
    assert(X.getElements());

    Matrix Y(M, N);
    MatFillY(Y);
    std::cout << "Y: " << Y << std::endl;

    Matrix Z(M, N);
    Z.multHost(X, Y);
    std::cout << "Z: " << Z << std::endl;

    assert(!Y.equal(Z, 1e-10));

    Matrix Y1(M, N);
    Y1.multHost(X, Z);
    assert(Y1.equal(Y, 1e-10));

    Matrix Z_gpu_0(M, N);
    Z_gpu_0.multGPU(X, Y, false);
    std::cout << "Z_gpu_0: " << Z_gpu_0 << std::endl;
    assert(Z_gpu_0.equal(Z, 1e-10));

    Matrix Z_gpu_1(M, N);
    Z_gpu_1.multGPU(X, Y, true);
    std::cout << "Z_gpu_1: " << Z_gpu_1 << std::endl;
    assert(Z_gpu_1.equal(Z, 1e-10));

    return 0;
}
