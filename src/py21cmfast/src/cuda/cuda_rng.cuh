#ifndef P21_CUDA_RNG_CUH
#define P21_CUDA_RNG_CUH

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

enum P21GslRngKind {
    P21_RNG_MT19937 = 0,
    P21_RNG_GFSR4 = 1,
    P21_RNG_CMRG = 2,
    P21_RNG_MRG = 3,
    P21_RNG_TAUS2 = 4,
};

bool p21_cuda_seed_rng_threads(unsigned long long random_seed, int n_threads, uint32_t *out_seeds,
                               cudaStream_t stream);

bool p21_cuda_fill_ugaussian_thread(int kind, uint32_t seed, double *d_out, size_t count,
                                    cudaStream_t stream);

#endif /* P21_CUDA_RNG_CUH */
