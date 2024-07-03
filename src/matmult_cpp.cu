#include <cassert>
#include <cmath>
#include <iomanip>

#include "matmult_cpp.cuh"

__global__ void MatIsZero_kernel(Matrix A, int* flg)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < A.getHeight() && c < A.getWidth()) {
        atomicAnd(flg, A(r, c) == 0);
    }
}

bool Matrix::isZero() const
{
    if (this->elements_cudaMalloc) {
        auto [gridDim, blockDim] = getGridAndBlockDim();

        int* flg = nullptr;
        cudaMallocManaged(&flg, sizeof(int));
        *flg = 1;
        MatIsZero_kernel<<<gridDim, blockDim>>>(*this, flg);
        cudaDeviceSynchronize();
        bool result = *flg;
        cudaFree(flg);
        return result;
    } else {
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                if ((*this)(r, c) != 0) {
                    return false;
                }
            }
        }
        return true;
    }
}

// Matrix multiplication kernel called by Mat::multGPU() - basic version
__global__ void MatMult_cpp_naive(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    C(r, c, A.multElement(B, r, c));
}

// Matrix multiplication kernel called by Mat::multGPU() - optimized version
// Should be run this way:
// size_t sharedMemSize = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float); // Total size for As and Bs
// MatMult_optimized<<<gridDim, blockDim, sharedMemSize>>>(A, B, C);
__global__ void MatMult_cpp_optimized(Matrix MatA, Matrix MatB, Matrix MatC)
{
    extern __shared__ char sharedMemory[];

    int R = blockIdx.y;
    int C = blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;
    int w = blockDim.x;
    int W = MatA.getWidth() / w;

    Matrix Asub(w, w, false);
    Matrix Bsub(w, w, false);
    Matrix Csub(w, w, false);

    Matrix Asub_s(w, w, (float*)sharedMemory);
    Matrix Bsub_s(w, w, (float*)&(sharedMemory[w * w * sizeof(float)]));

    MatC.getSubMatrix(R, C, w, Csub);
    // Each thread computes one element of Csub
    float Csub_rc = 0;
    // C_{R,C} = \sum_{K=0}^{W-1} A_{R,K} B_{K,C}
    for (int K = 0; K < W; K++) {
        MatA.getSubMatrix(R, K, w, Asub);
        MatB.getSubMatrix(K, C, w, Bsub);
        __syncthreads();
        Asub.copyElement(Asub_s, r, c);
        Bsub.copyElement(Bsub_s, r, c);
        __syncthreads();
        // Csub_{r,c} = \sum_{k=0}^{w-1} A_{r,k} B_{k,c}
        Csub_rc += Asub_s.multElement(Bsub_s, r, c);
    }
    Csub(r, c, Csub_rc);
}

void Matrix::multGPU(const Matrix& A, const Matrix& B, bool optimized)
{
    assert(A.width == B.height);
    assert(A.height == this->height);
    assert(B.width == this->width);

    // Load A to device memory
    Matrix d_A(A.height, A.width, false, true);
    cudaMemcpy(d_A.elements, A.elements, A.sizeInBytes(), cudaMemcpyHostToDevice);

    // Load B to device memory
    Matrix d_B(B.height, B.width, false, true);
    cudaMemcpy(d_B.elements, B.elements, B.sizeInBytes(), cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C(this->height, this->width, false, true);

    auto [dimGrid, dimBlock] = this->getGridAndBlockDim();

#ifndef NDEBUG
    assert(d_C.isZero());
#endif

    if (optimized) {
        size_t sharedMemSize = 2 * dimBlock.x * dimBlock.y * sizeof(float); // Total size for As and Bs
        MatMult_cpp_optimized<<<dimGrid, dimBlock, sharedMemSize>>>(d_A, d_B, d_C);
    } else {
        MatMult_cpp_naive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    }

#ifndef NDEBUG
    bool isAZero = d_A.isZero();
    bool isBZero = d_B.isZero();
    bool isCZero = d_C.isZero();
    if (isAZero || isBZero) {
        assert(isCZero);
    } else {
        assert(!isCZero);
    }
#endif

    // Read C from device memory
    cudaMemcpy(this->elements, d_C.elements, this->sizeInBytes(), cudaMemcpyDeviceToHost);

#ifndef NDEBUG
    assert(isZero() == isCZero);
#endif
}

void Matrix::multHost(const Matrix& A, const Matrix& B)
{
    assert(A.width == B.height);
    assert(A.height == this->height);
    assert(B.width == this->width);

    for (int r = 0; r < A.height; r++) {
        for (int c = 0; c < B.width; c++) {
            (*this)(r, c, A.multElement(B, r, c));
        }
    }
}

std::ostream& operator<<(std::ostream& os, const Matrix& A)
{
    // Save the current format settings
    std::ios oldState(nullptr);
    oldState.copyfmt(os);

    os << A.name << " = [\n";
    for (int i = 0; i < A.height; ++i) {
        for (int j = 0; j < A.width; ++j) {
            os << std::fixed << std::setw(5) << std::setprecision(1) << A(i, j);
            if (j < A.width - 1) {
                os << " ";
            }
        }
        os << "\n";
    }
    os << "]\n";

    // Restore the original format settings
    os.copyfmt(oldState);
    return os;
}

bool Matrix::equal(const Matrix& B, float tol) const
{
    if (height != B.height || width != B.width)
        return false;
    for (int r = 0; r < height; r++)
        for (int c = 0; c < width; c++) {
            if (fabs((*this)(r, c) - B(r, c)) > tol)
                return false;
        }
    return true;
}
