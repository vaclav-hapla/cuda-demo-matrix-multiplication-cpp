#pragma once

#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <utility>

// Dense matrix structure.
// Matrices are stored in row-major order:
//     M(row, col) = *(M.elements + row * M.stride + col)
class Matrix {
    int    width;
    int    height;
    int    blockSize;
    int    stride;
    float* elements;
    float* elements_malloc;
    float* elements_cudaMalloc;

    // Initialize inner matrix fields without allocating its elements
    __device__ __host__ inline void init(int height, int width, int blockSize)
    {
        this->height    = height;
        this->width     = width;
        this->blockSize = blockSize;
        this->stride    = width;

        this->elements            = nullptr;
        this->elements_malloc     = nullptr;
        this->elements_cudaMalloc = nullptr;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& mat);

public:
    // Constructor to create a matrix, with optional host and GPU allocation
    __device__ __host__ Matrix(int height, int width, bool allocateHost = true, bool allocateGPU = false, int blockSize = 4)
        : width(width)
        , height(height)
        , blockSize(blockSize)
        , stride(width)
        , elements(nullptr)
        , elements_malloc(nullptr)
        , elements_cudaMalloc(nullptr)
    {
#ifndef __CUDA_ARCH__
        if (allocateHost && allocateGPU) {
            throw std::runtime_error("Allocating both host and GPU memory simultaneously is currently not supported");
        }
#endif
        if (allocateHost) {
            elements_malloc = new float[height * width];
            elements        = elements_malloc;
        }
        if (allocateGPU) {
            cudaMalloc(&elements_cudaMalloc, height * width * sizeof(float));
            elements = elements_cudaMalloc;
        }
    }

    __device__ __host__ Matrix(int height, int width, float elements[], int blockSize = 4)
        : width(width)
        , height(height)
        , blockSize(blockSize)
        , stride(width)
        , elements(elements)
        , elements_malloc(nullptr)
        , elements_cudaMalloc(nullptr)
    {
    }

    // In copy constructor, we make sure elements_malloc and elements_cudaMalloc are not copied
    // and hence they are not deallocated in the destructor
    __device__ __host__ Matrix(const Matrix& other)
        : width(other.width)
        , height(other.height)
        , blockSize(other.blockSize)
        , stride(other.stride)
        , elements(other.elements)
        , elements_malloc(nullptr)
        , elements_cudaMalloc(nullptr)
    {
    }

    __device__ __host__ Matrix& operator=(const Matrix& other)
    {
        width     = other.width;
        height    = other.height;
        blockSize = other.blockSize;
        stride    = other.stride;
        elements  = other.elements;

        elements_malloc     = nullptr;
        elements_cudaMalloc = nullptr;
        return *this;
    }

    __device__ __host__ ~Matrix()
    {
        if (elements_malloc) {
            delete[] elements_malloc;
        }
        if (this->elements_cudaMalloc) {
            cudaFree(elements_cudaMalloc);
        }
    }

    __device__ __host__ int    getWidth() const { return width; }
    __device__ __host__ int    getHeight() const { return height; }
    __device__ __host__ size_t sizeInBytes() const { return height * width * sizeof(float); }
    __device__ __host__ int    getBlockSize() const { return blockSize; }

    std::pair<dim3, dim3> getGridAndBlockDim() const
    {
        dim3 blockDim(blockSize, blockSize);
        dim3 gridDim((width + blockSize - 1) / blockSize, (height + blockSize - 1) / blockSize);
        return std::make_pair(gridDim, blockDim);
    }

    __device__ __host__ const float* getElements() { return elements; }

    __device__ __host__ void setElements(float* elements) { this->elements = elements; }

    // Get a matrix element
    __device__ __host__ inline float getElement(int r, int c) const { return elements[r * stride + c]; }

    // Set a matrix element
    __device__ __host__ inline void setElement(int r, int c, float value) { elements[r * stride + c] = value; }

    // Copy a matrix element
    __device__ __host__ inline void copyElement(Matrix& B, int r, int c) const { B.setElement(r, c, getElement(r, c)); }

    // Get the blockSize x blockSize submatrix Asub of A,
    // located C submatrices to the right and R submatrices down
    // from the upper-left corner of A.
    __device__ __host__ inline void getSubMatrix(int R, int C, int blockSize, Matrix& Asub) const
    {
        assert(Asub.elements_cudaMalloc == nullptr && Asub.elements_malloc == nullptr);
        Asub.init(blockSize, blockSize, blockSize);
        Asub.stride   = stride;
        Asub.elements = &(elements[R * stride * blockSize + C * blockSize]);
    }

    // Compute a single element C(r,c) of the matrix-matrix product C = A * B
    // C_{r,c} = \sum_{k=0}^{w-1} A_{r,k} B_{k,c}
    __device__ __host__ inline float multElement(const Matrix& B, int r, int c) const
    {
        float C_rc = 0;
        for (int k = 0; k < width; k++) {
            C_rc += elements[r * stride + k] * B.elements[k * B.stride + c];
        }
        return C_rc;
    }

    bool isZero() const;

    // Matrix dimensions are assumed to be multiples of BLOCK_SIZE
    void multGPU(const Matrix& A, const Matrix& B, bool optimized);

    // Matrix dimensions are assumed to be multiples of BLOCK_SIZE
    void multHost(const Matrix& A, const Matrix& B);

    // Compare two matrices up to a given tolerance
    bool equal(const Matrix& B, float tol) const;
};
