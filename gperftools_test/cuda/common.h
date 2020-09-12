// Copyright (c) 2020 YuChen. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <sys/time.h>
#include <cstdlib>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(                                                                 \
          stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                                 \
    }                                                                          \
  }

#define CHECK_CUBLAS(call)                                                    \
  {                                                                           \
    cublasStatus_t err;                                                       \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS) {                            \
      fprintf(                                                                \
          stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__, __LINE__); \
      exit(1);                                                                \
    }                                                                         \
  }

#define CHECK_CURAND(call)                                                    \
  {                                                                           \
    curandStatus_t err;                                                       \
    if ((err = (call)) != CURAND_STATUS_SUCCESS) {                            \
      fprintf(                                                                \
          stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__, __LINE__); \
      exit(1);                                                                \
    }                                                                         \
  }

#define CHECK_CUFFT(call)                                                    \
  {                                                                          \
    cufftResult err;                                                         \
    if ((err = (call)) != CUFFT_SUCCESS) {                                   \
      fprintf(                                                               \
          stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__, __LINE__); \
      exit(1);                                                               \
    }                                                                        \
  }

#define CHECK_CUSPARSE(call)                                               \
  {                                                                        \
    cusparseStatus_t err;                                                  \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS) {                       \
      fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__); \
      cudaError_t cuda_err = cudaGetLastError();                           \
      if (cuda_err != cudaSuccess) {                                       \
        fprintf(stderr,                                                    \
                "  CUDA error \"%s\" also detected\n",                     \
                cudaGetErrorString(cuda_err));                             \
      }                                                                    \
      exit(1);                                                             \
    }                                                                      \
  }

inline double seconds() {
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return (static_cast<double>(tp.tv_sec) +
          static_cast<double>(tp.tv_usec) * 1.e-6);
}

inline double milliseconds() {
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return (static_cast<double>(tp.tv_sec) * 1.e3 +
          static_cast<double>(tp.tv_usec) * 1.e-3);
}

#endif  // _COMMON_H
