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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <type_traits>
#include "cuda/common.h"
#include "cuda/inner.h"

#define GET_VEC_TYPE(type__, size__) type__##size__
#define VECTORIZED_TYPE(type__, size__) GET_VEC_TYPE(type__, size__)
#define CAST_FUNC(dest, src) __##src##2##dest##_rn

namespace details {
template <typename T>
struct VecTypeTrait {};

template <>
struct VecTypeTrait<double> {
  using VecType2 = VECTORIZED_TYPE(double, 2);
  using VecType3 = VECTORIZED_TYPE(double, 3);
  using VecType4 = VECTORIZED_TYPE(double, 4);
};

template <>
struct VecTypeTrait<float> {
  using VecType2 = VECTORIZED_TYPE(float, 2);
  using VecType3 = VECTORIZED_TYPE(float, 3);
  using VecType4 = VECTORIZED_TYPE(float, 4);
};

template <>
struct VecTypeTrait<half> {
  using VecType2 = VECTORIZED_TYPE(half, 2);
};

template <>
struct VecTypeTrait<int> {
  using VecType2 = VECTORIZED_TYPE(int, 2);
  using VecType3 = VECTORIZED_TYPE(int, 3);
  using VecType4 = VECTORIZED_TYPE(int, 4);
};

}  // namespace details

template <typename SU, typename VT, typename VU>
__device__ void cast_gpu(const VT &in, VU *out) {
  // printf("----call the common----\n");
  out->x = static_cast<SU>(in.x);
  out->y = static_cast<SU>(in.y);
}

template <typename SU>
__device__ void cast_gpu(const half2 &in, float2 *out) {
  // printf("----__half22float2----\n");
  *out = __half22float2(in);
}

template <typename SU>
__device__ void cast_gpu(const float2 &in, half2 *out) {
  // printf("----__float22half2_rn----\n");
  *out = __float22half2_rn(in);
}

template <typename T, typename U>
__global__ void device_copy_vector2_kernel(T *d_in, U *d_out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < N / 2; i += blockDim.x * gridDim.x) {
    using VecT = typename details::VecTypeTrait<T>::VecType2;
    using VecU = typename details::VecTypeTrait<U>::VecType2;

    auto *in2 = &(reinterpret_cast<VecT *>(d_in)[i]);
    auto *out2 = &(reinterpret_cast<VecU *>(d_out)[i]);
    cast_gpu<U>(*in2, out2);
  }

  // in only one thread, process final elements (if there are any)
  if (idx == 0 && N % 2 == 1) d_out[N - 1] = d_in[N - 1];
}

void device_copy_vector2() {
  using src_type = half;
  using dest_type = float;
  int data_size = 10;
  src_type *d_in;
  dest_type *d_out;
  CHECK(cudaMalloc(&d_in, data_size * sizeof(src_type)));
  CHECK(cudaMalloc(&d_out, data_size * sizeof(dest_type)));

  std::unique_ptr<src_type[]> h_in = std::make_unique<src_type[]>(data_size);
  std::unique_ptr<dest_type[]> h_out = std::make_unique<dest_type[]>(data_size);

  for (int i = 0; i < data_size; i++) {
    h_in[i] = static_cast<src_type>(i * 3.14);
  }

  CHECK(cudaMemcpy(
      d_in, h_in.get(), data_size * sizeof(src_type), cudaMemcpyHostToDevice));
  int threads = 128;
  int blocks = (data_size / 2 + threads - 1) / threads;

  device_copy_vector2_kernel<src_type, dest_type><<<blocks, threads>>>(
      d_in, d_out, data_size);

  CHECK(cudaMemcpy(h_out.get(),
                   d_out,
                   data_size * sizeof(dest_type),
                   cudaMemcpyDeviceToHost));
  for (int i = 0; i < data_size; i++) {
    std::cout << h_out[i] << "\t";
  }
  std::cout << std::endl;

  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out));
}

int InnerMain(int argc, char **argv) {
  int deviceCount;  // Gets the number of compute-capable devices.
  CHECK(cudaGetDeviceCount(&deviceCount));
  std::cout << "----------The number of compute-capable devices: "
            << deviceCount << ".\n";

  int whichDev;  // Gets which device is currently being used.
  CHECK(cudaGetDevice(&whichDev));
  std::cout << "----------Currently use the device: " << whichDev << ".\n";

  int numSMs;  // Gets information about the device.
  CHECK(cudaDeviceGetAttribute(
      &numSMs, cudaDevAttrMultiProcessorCount, whichDev));
  std::cout
      << "----------The count of multiprocessor in the current used device: "
      << numSMs << ".\n";

  auto start = milliseconds();
  device_copy_vector2();
  auto end = milliseconds();
  std::cout << "----------The time costs: " << end - start << " ms.\n";

  // reset device before you leave
  CHECK(cudaDeviceReset());

  return (0);
}
