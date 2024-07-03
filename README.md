# CUDA Demo: Matrix Multiplication

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Makefile CI](https://github.com/vaclav-hapla/cuda-demo-matrix-multiplication-cpp/actions/workflows/makefile.yml/badge.svg)](https://github.com/vaclav-hapla/cuda-demo-matrix-multiplication-cpp/actions/workflows/makefile.yml)

* This repo contains a simple library with a C++ API, implementing matrix-matrix multiply
  - The C version can be found [here](https://github.com/vaclav-hapla/cuda-demo-matrix-multiplication-c).
* Serves mainly for demonstration
* There are three implementations of matrix-matrix multiply:
    1. host: `Matrix::multHost()`,
    2. naive GPU: `Matrix::multGPU()` with `MatMult_cpp_naive()` kernel
    3. optimized GPU: `Matrix::multGPU()` with `MatMult_cpp_optimized()` kernel (using blocks in shared memory)
* The repo also features:
    - [`Makefile`](Makefile)
      - working with a standard Linux directory structure
      - building a separate shared library `lib/libcudamatmult++.so`
      - building and running a test suite (`make test`)
    - [`.clangd`](.clangd) file for smooth clangd operation with CUDA
    - [`.clang-format`](.clang-format) file for unified formatting using clang-format
    - [`.vscode/`](.vscode/) for convenient building and debugging in VS Code
    - [Gitlab Workflow](.github/workflows/makefile.yml) running `make test` in a self-hosted CUDA-enabled runner
      - I've set up such a runner on an ETH workstation and connected to [the project on GitHub](https://github.com/vaclav-hapla/cuda-demo-matrix-multiplication-cpp)
    - [MIT License](LICENSE)
