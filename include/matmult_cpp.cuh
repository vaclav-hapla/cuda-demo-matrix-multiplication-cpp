#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <iostream>
#include <utility>

// Dense matrix structure.
// Matrices are stored in row-major order:
//     M(row, col) = *(M.elements + row * M.stride + col)
class Matrix {
    // std::string currently can't be used in device code
    const char* name = "";

    int width;
    int height;
    int blockSize;
    int stride;

    float* elements            = nullptr;
    float* elements_malloc     = nullptr;
    float* elements_cudaMalloc = nullptr;

    // Initialize inner matrix fields without allocating its elements
    __device__ __host__ inline void init(int height, int width, int blockSize)
    {
        this->height    = height;
        this->width     = width;
        this->blockSize = blockSize;
        this->stride    = width;
    }

public:
    // Constructor to create a matrix, with optional host and GPU allocation
    __device__ __host__ Matrix(int height, int width, bool allocateHost = true, bool allocateGPU = false, int blockSize = 4)
        : width(width)
        , height(height)
        , blockSize(blockSize)
        , stride(width)
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

    Matrix(const std::string& name, int height, int width, bool allocateHost = true, bool allocateGPU = false, int blockSize = 4)
        : Matrix(height, width, allocateHost, allocateGPU, blockSize)
    {
        setName(name);
    }

    // Constructor to create a matrix with pre-allocated elements
    __device__ __host__ Matrix(int height, int width, float elements[], int blockSize = 4)
        : width(width)
        , height(height)
        , blockSize(blockSize)
        , stride(width)
        , elements(elements)
    {
    }

    Matrix(const std::string& name, int height, int width, float elements[], int blockSize = 4)
        : Matrix(height, width, false, false, blockSize)
    {
        setName(name);
    }

    // Copy constructor (shallow copy)
    // We make sure elements_malloc and elements_cudaMalloc are not copied
    // and hence they are not deallocated in the destructor
    __device__ __host__ Matrix(const Matrix& other)
        : name(other.name)
        , width(other.width)
        , height(other.height)
        , blockSize(other.blockSize)
        , stride(other.stride)
        , elements(other.elements)
    {
    }

    // Copy-assignment operator (shallow copy)
    __device__ __host__ Matrix& operator=(const Matrix& other)
    {
        name      = other.name;
        width     = other.width;
        height    = other.height;
        blockSize = other.blockSize;
        stride    = other.stride;

        elements            = other.elements;
        elements_malloc     = nullptr;
        elements_cudaMalloc = nullptr;
        return *this;
    }

    // Move constructor
    __device__ __host__ Matrix(Matrix&& other)
        : name(other.name)
        , width(other.width)
        , height(other.height)
        , blockSize(other.blockSize)
        , stride(other.stride)
        , elements(other.elements)
        , elements_malloc(other.elements_malloc)
        , elements_cudaMalloc(other.elements_cudaMalloc)
    {
        other.elements            = nullptr;
        other.elements_malloc     = nullptr;
        other.elements_cudaMalloc = nullptr;
    }

    // Move-assignment operator
    __device__ __host__ Matrix& operator=(Matrix&& other)
    {
        name      = other.name;
        width     = other.width;
        height    = other.height;
        blockSize = other.blockSize;
        stride    = other.stride;

        elements            = other.elements;
        elements_malloc     = other.elements_malloc;
        elements_cudaMalloc = other.elements_cudaMalloc;

        other.elements            = nullptr;
        other.elements_malloc     = nullptr;
        other.elements_cudaMalloc = nullptr;
        return *this;
    }

    // Destructor
    __device__ __host__ ~Matrix()
    {
        if (elements_malloc) {
            delete[] elements_malloc;
        }
        if (this->elements_cudaMalloc) {
            cudaFree(elements_cudaMalloc);
        }
    }

    // Get the matrix name as std::string
    std::string getName() const { return std::string(this->name); }

    // Get the matrix name as char*
    const char* getNameCStr() const { return name; }

    // Set the matrix name via std::string
    void setName(const std::string& name) { this->name = name.c_str(); }

    // Set the matrix name via char*
    __device__ __host__ void setName(const char name[]) { this->name = name; }

    // Get the matrix width (number of columns)
    __device__ __host__ int getWidth() const { return width; }

    // Get the matrix height (number of rows)
    __device__ __host__ int getHeight() const { return height; }

    // Get the block size
    __device__ __host__ int getBlockSize() const { return blockSize; }

    // Get the element array size in bytes
    __device__ __host__ size_t sizeInBytes() const { return height * width * sizeof(float); }

    // Get the grid and block dimensions
    std::pair<dim3, dim3> getGridAndBlockDim() const
    {
        dim3 blockDim(blockSize, blockSize);
        dim3 gridDim((width + blockSize - 1) / blockSize, (height + blockSize - 1) / blockSize);
        return std::make_pair(gridDim, blockDim);
    }

    // Get the elements; the array should not be modified
    __device__ __host__ const float* getElements() { return elements; }

    // Get a matrix element
    __device__ __host__ float operator()(int r, int c) const { return elements[r * stride + c]; }

    // Get/set a matrix element via reference
    __device__ __host__ float& operator()(int r, int c) { return elements[r * stride + c]; }

    // Get the blockSize x blockSize submatrix Asub of A,
    // located C submatrices to the right and R submatrices down
    // from the upper-left corner of A.
    __device__ __host__ Matrix getSubMatrix(int R, int C, int blockSize) const
    {
        Matrix Asub(blockSize, blockSize, false, false, blockSize);
        Asub.stride   = stride;
        Asub.elements = &(elements[R * stride * blockSize + C * blockSize]);
        return Asub;
    }

    // Compute a single element C(r,c) of the matrix-matrix product C = A * B
    // C_{r,c} = \sum_{k=0}^{w-1} A_{r,k} B_{k,c}
    __device__ __host__ float multElement(const Matrix& B, int r, int c) const
    {
        float C_rc = 0;
        for (int k = 0; k < width; k++) {
            C_rc += elements[r * stride + k] * B.elements[k * B.stride + c];
        }
        return C_rc;
    }

    // Check if all elements are zero
    bool isZero() const;

    // Matrix dimensions are assumed to be multiples of BLOCK_SIZE
    void multGPU(const Matrix& A, const Matrix& B, bool optimized = true);

    // Matrix dimensions are assumed to be multiples of BLOCK_SIZE
    void multHost(const Matrix& A, const Matrix& B);

    // Compare two matrices up to a given tolerance
    bool equal(const Matrix& B, float tol) const;

    friend std::ostream& operator<<(std::ostream& os, const Matrix& mat);
};
