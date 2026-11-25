#ifndef P21_CUDA_COMMON_CUH
#define P21_CUDA_COMMON_CUH

#include <cufft.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

extern "C" {
#include "../Constants.h"
#include "../InputParameters.h"
#include "../OutputStructs.h"
#include "../cosmology.h"
#include "../elec_interp.h"
#include "../exceptions.h"
#include "../heating_helper_progs.h"
#include "../interp_tables.h"
#include "../interpolation.h"
#include "../recombinations.h"
#include "../scaling_relations.h"
#include "cuda_api.h"

void p21_get_ms_filter_params(double R_inner, double R_outer, double R_star,
                              double out_alpha_beta[4]);
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

static inline bool p21_cuda_check_impl(cudaError_t err, const char *expr, const char *file,
                                       int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[21cmFAST CUDA] %s failed at %s:%d: %s\n", expr, file, line,
                cudaGetErrorString(err));
        return false;
    }
    return true;
}

static inline bool p21_cufft_check_impl(cufftResult res, const char *expr, const char *file,
                                        int line) {
    if (res != CUFFT_SUCCESS) {
        fprintf(stderr, "[21cmFAST cuFFT] %s failed at %s:%d (code=%d)\n", expr, file, line,
                (int)res);
        return false;
    }
    return true;
}

#define P21_CUDA_OK(expr) p21_cuda_check_impl((expr), #expr, __FILE__, __LINE__)
#define P21_CUFFT_OK(expr) p21_cufft_check_impl((expr), #expr, __FILE__, __LINE__)

#ifndef D_PARA
#define D_PARA ((int)(simulation_options_global->NON_CUBIC_FACTOR * simulation_options_global->DIM))
#endif
#ifndef HII_D_PARA
#define HII_D_PARA \
    ((int)(simulation_options_global->NON_CUBIC_FACTOR * simulation_options_global->HII_DIM))
#endif

/* Indexing helpers matching indexing.h / indexing.c */
__host__ __device__ static inline size_t p21_grid_idx(int x, int y, int z, int ny, int nz) {
    return (size_t)z + (size_t)nz * ((size_t)y + (size_t)ny * (size_t)x);
}

__host__ __device__ static inline size_t p21_fft_r_idx(int x, int y, int z, int ny, int nz) {
    size_t zp = (size_t)(2 * (nz / 2 + 1));
    return (size_t)z + zp * ((size_t)y + (size_t)ny * (size_t)x);
}

__host__ __device__ static inline size_t p21_fft_c_idx(int x, int y, int z, int ny, int nz) {
    size_t hz = (size_t)(nz / 2 + 1);
    return (size_t)z + hz * ((size_t)y + (size_t)ny * (size_t)x);
}

__host__ __device__ static inline double p21_index_to_k(int idx, double len, int dim) {
    double buf = (idx <= dim / 2) ? (double)idx : (double)(idx - dim);
    return buf * 2.0 * M_PI / len;
}

__host__ __device__ static inline int p21_resample_coord(int idx_in, double dim_ratio) {
    return (int)(idx_in * dim_ratio + 0.5);
}

__host__ __device__ static inline int p21_wrap_coord(int v, int dim) {
    while (v >= dim) v -= dim;
    while (v < 0) v += dim;
    return v;
}

__device__ static inline void p21_atomic_add_double(double *addr, double val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
    unsigned long long int *addr_as_ull = (unsigned long long int *)addr;
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
#else
    atomicAdd(addr, val);
#endif
}

/* Runtime & stats (cuda_runtime.cu) */
bool p21_cuda_ensure_init(void);
cudaStream_t p21_cuda_stream(void);
int p21_cuda_current_device(void);
size_t p21_cuda_total_mem_bytes(void);
P21CudaStats *p21_cuda_stats_mut(void);
void p21_cuda_stats_add_d2h(size_t nbytes);

/* Memory pool & DeviceFieldRegistry (cuda_mempool.cu) */
void p21_cuda_pool_init(int device_id);
void *p21_cuda_pool_alloc(size_t nbytes, cudaStream_t stream);
void p21_cuda_pool_free(void *ptr, cudaStream_t stream);
void p21_cuda_pool_trim(void);

void *p21_cuda_registry_get_or_upload(const void *host_ptr, size_t nbytes, cudaStream_t stream);
bool p21_cuda_registry_put_copy(const void *host_ptr, const void *d_src, size_t nbytes,
                                cudaStream_t stream);
bool p21_cuda_registry_put_owned(const void *host_ptr, void *d_buf, size_t nbytes,
                                 cudaStream_t stream);

/* cuFFT cache & shared k-space filter (cuda_cufft.cu) */
bool p21_cuda_fft_r2c(const float *d_real, cufftComplex *d_complex, int nx, int ny, int nz,
                      bool padded, cudaStream_t stream);
bool p21_cuda_fft_c2r(cufftComplex *d_complex_scratch, float *d_real, int nx, int ny, int nz,
                      bool padded, cudaStream_t stream);
void p21_cuda_fft_clear_cache(void);

bool p21_cuda_copy_and_filter_box(const cufftComplex *d_src, cufftComplex *d_dst, int nx, int ny,
                                  int nz, double box_len, double non_cubic_factor, int filter_type,
                                  float R, float R_param, float R_star, float pre_scale,
                                  cudaStream_t stream);
bool p21_cuda_filter_box(cufftComplex *d_box, int nx, int ny, int nz, double box_len,
                         double non_cubic_factor, int filter_type, float R, float R_param,
                         float R_star, cudaStream_t stream);

/* Internal module cache cleanup hooks & shared GPU builders */
bool p21_cuda_build_mturn_grids(const ScalingConstants *consts, InitialConditions *ini_boxes,
                                TsBox *previous_spin_temp, IonizedBox *previous_ionize_box,
                                cudaStream_t stream, float **d_mturn_acg_out,
                                float **d_mturn_mcg_out, double log10_mturn_averages_out[2]);
void p21_cuda_radiation_invalidate_host_ptr(const void *ptr);
void p21_cuda_radiation_clear(void);
void p21_cuda_spin_temp_cleanup(void);
void p21_cuda_ionization_cleanup(void);

#endif /* P21_CUDA_COMMON_CUH */
