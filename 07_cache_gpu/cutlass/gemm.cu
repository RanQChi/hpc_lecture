#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#define DEBUG

#include <gemm/dispatch.h>
#include <gemm/epilogue_function.h>
#include "util/matrix.h"
#include "util/timer.h"

using namespace cutlass;

int main(int argc, const char **argv) {
  int m = 10240,k = 4096,n = 4096,g_timing_iterations = 10;
  float alpha = 1.0,beta = 0.0;

  typedef float value_t, accum_t;
  
  matrix<value_t> A(m, k);
  matrix<value_t> B(k, n);
  matrix<accum_t> C(m, n);
  matrix<accum_t> C2(m, n);
  A.random();
  B.random();
  C.fill_ramp(0,0);
  C2.fill_ramp(0,0);
  
  cublasHandle_t g_cublas_handle;
  cublasCreate(&g_cublas_handle);
  gpu_timer timer;

  
#pragma omp parallel for
  for (int i = 0; i < g_timing_iterations+2; i++) {
    if (i == 2) timer.start();
      cublasSgemm(
      		g_cublas_handle,
      		CUBLAS_OP_N,
      		CUBLAS_OP_N,
      		m,
      		n,
      		k,
      		&alpha,
     		A.d_data(),
      		m,
      		B.d_data(),
     		k,
     		&beta,
      		C.d_data(),
      		m);
  }
  timer.stop();
  
  double num_flops = (2 * m * n * int64_t(k)) + (2 * m * n);
  double tcublas = timer.elapsed_millis() / g_timing_iterations;
  double cublas_flops = num_flops / tcublas / 1.0e6;
  typedef gemm::blas_scaled_epilogue<float, float, float> epilogue_op_t;
  epilogue_op_t epilogue(alpha, beta);
  
//#pragma omp parallel for
  for (int i = 0; i < g_timing_iterations+2; i++) {
    if (i == 2) timer.start();
    gemm::dispatch<epilogue_op_t>(
    m,
    n,
    k,
    alpha,
    beta,
    A.d_data(),
    B.d_data(),
    C2.d_data()
    );
  }
  timer.stop();
  
  double tcutlass = timer.elapsed_millis() / g_timing_iterations;
  double cutlass_flops = num_flops / tcutlass / 1.0e6;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
//#pragma omp parallel for
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C.get(i,j) - C2.get(i,j));
    }
  }
  printf("error: %lf\n", err/n/m);
  cublasDestroy(g_cublas_handle);
  
}
