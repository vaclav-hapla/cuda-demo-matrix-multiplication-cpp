# CUDA Demo: Matrix Multiplication

* This repo contains a simple C library implementing matrix-matrix multiply
    - Serves mainly to show my capabilities
* There are three implementations:
    1. host: `MatMultHost()`,
    2. naive GPU: `MatMultGPU()` with `MatMult_naive()` kernel
    3. optimized GPU: `MatMultGPU()` with `MatMult_optimized()` kernel (using blocks in shared memory)
* The repo also features:
    - [`Makefile`](Makefile)
      - working with a standard Linux directory structure
      - building a separate shared library `lib/libcudamatmult.so`
      - building and running a test suite (`make test`)
    - [`.clangd`](.clangd) file for smooth clangd operation with CUDA
    - [`.clang-format`](.clang-format) file for unified formatting using clang-format
    - [Gitlab Workflow](.github/workflows/makefile.yml) running `make test` in a self-hosted CUDA-enabled runner
      - I've set up such runner running on an ETH workstation and connected to [the project on GitHub](https://github.com/haplav/cuda-demo-matrix-multiplication)
    - [MIT License](LICENSE)
