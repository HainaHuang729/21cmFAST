#include <math.h>

#include "cuda_common.cuh"

namespace {

constexpr int kBlockThreads = 256;

__global__ void brightness_temp_kernel(const float *density, const float *neutral_fraction,
                                       const float *spin_temp, float *brightness_temp,
                                       float *tau_21, size_t n, float const_factor, float T_rad,
                                       float redshift, int use_ts_fluct, double *block_sums,
                                       int *nonfinite_flag) {
    __shared__ double sdata[kBlockThreads];

    size_t tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    double local_sum = 0.0;
    int local_nf = 0;

    for (size_t i = idx; i < n; i += stride) {
        double pixel_deltax = (double)density[i];
        double pixel_x_HI = (double)neutral_fraction[i];
        double val = (double)const_factor * pixel_x_HI * (1.0 + pixel_deltax);
        float tb_f;

        if (use_ts_fluct) {
            double ts = (double)spin_temp[i];
            float val_tau_f = (float)(val * (1.0 + (double)redshift) / (1000.0 * ts));
            tau_21[i] = val_tau_f;
            tb_f = (float)((1.0 - exp(-(double)val_tau_f)) * 1000.0 * (ts - (double)T_rad) /
                           (1.0 + (double)redshift));
        } else {
            tb_f = (float)val;
        }

        brightness_temp[i] = tb_f;
        if (!isfinite(tb_f)) {
            local_nf = 1;
        }
        local_sum += (double)tb_f;
    }

    if (local_nf) {
        atomicOr(nonfinite_flag, 1);
    }

    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void finish_sum_kernel(const double *block_sums, int nblocks, double *out_sum) {
    __shared__ double sdata[kBlockThreads];
    size_t tid = threadIdx.x;
    double acc = 0.0;
    for (int i = (int)tid; i < nblocks; i += kBlockThreads) {
        acc += block_sums[i];
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out_sum = sdata[0];
    }
}

}  // namespace

extern "C" int p21_cuda_compute_brightness_temp(float redshift, TsBox *spin_temp,
                                                IonizedBox *ionized_box,
                                                PerturbedField *perturb_field,
                                                BrightnessTemp *box) {
    if (!ionized_box || !perturb_field || !box || !p21_cuda_get_enabled()) {
        return -1;
    }
    if (simulation_options_global->HII_DIM <= 1) {
        return -1;
    }
    int use_ts_fluct = astro_options_global->USE_TS_FLUCT ? 1 : 0;
    if (use_ts_fluct && (!spin_temp || !spin_temp->spin_temperature || !box->tau_21)) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    const int nx = simulation_options_global->HII_DIM;
    const int ny = simulation_options_global->HII_DIM;
    const int nz = HII_D_PARA;
    const size_t total = (size_t)nx * (size_t)ny * (size_t)nz;
    const size_t nbytes = total * sizeof(float);
    cudaStream_t stream = p21_cuda_stream();

    const float *d_density =
        (const float *)p21_cuda_registry_get_or_upload(perturb_field->density, nbytes, stream);
    const float *d_xhi =
        (const float *)p21_cuda_registry_get_or_upload(ionized_box->neutral_fraction, nbytes, stream);
    const float *d_ts = nullptr;
    if (use_ts_fluct) {
        d_ts = (const float *)p21_cuda_registry_get_or_upload(spin_temp->spin_temperature, nbytes,
                                                              stream);
    }
    if (!d_density || !d_xhi || (use_ts_fluct && !d_ts)) {
        return -1;
    }

    float *d_bt = (float *)p21_cuda_pool_alloc(nbytes, stream);
    float *d_tau = use_ts_fluct ? (float *)p21_cuda_pool_alloc(nbytes, stream) : nullptr;
    int blocks = (int)((total + kBlockThreads - 1) / kBlockThreads);
    if (blocks > 1024) {
        blocks = 1024;
    }
    double *d_block_sums = (double *)p21_cuda_pool_alloc((size_t)blocks * sizeof(double), stream);
    double *d_scalar_sum = (double *)p21_cuda_pool_alloc(sizeof(double), stream);
    int *d_nonfinite = (int *)p21_cuda_pool_alloc(sizeof(int), stream);
    if (!d_bt || (use_ts_fluct && !d_tau) || !d_block_sums || !d_scalar_sum || !d_nonfinite) {
        p21_cuda_pool_free(d_bt, stream);
        p21_cuda_pool_free(d_tau, stream);
        p21_cuda_pool_free(d_block_sums, stream);
        p21_cuda_pool_free(d_scalar_sum, stream);
        p21_cuda_pool_free(d_nonfinite, stream);
        return -1;
    }

    float T_rad = (float)(physconst.T_cmb * (1.0 + redshift));
    float const_factor =
        (float)(27.0 *
                (cosmo_params_global->OMb * cosmo_params_global->hlittle *
                 cosmo_params_global->hlittle / 0.023) *
                sqrt((0.15 / (cosmo_params_global->OMm) / (cosmo_params_global->hlittle) /
                      (cosmo_params_global->hlittle)) *
                     (1.0 + redshift) / 10.0));

    bool ok = P21_CUDA_OK(cudaMemsetAsync(d_nonfinite, 0, sizeof(int), stream));
    if (ok) {
        brightness_temp_kernel<<<blocks, kBlockThreads, 0, stream>>>(
            d_density, d_xhi, d_ts, d_bt, d_tau, total, const_factor, T_rad, redshift,
            use_ts_fluct, d_block_sums, d_nonfinite);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    if (ok) {
        finish_sum_kernel<<<1, kBlockThreads, 0, stream>>>(d_block_sums, blocks, d_scalar_sum);
        ok = P21_CUDA_OK(cudaGetLastError());
    }

    double h_ave_sum = 0.0;
    int h_nonfinite = 0;
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(box->brightness_temp, d_bt, nbytes, cudaMemcpyDeviceToHost,
                                         stream));
        if (ok) {
            p21_cuda_stats_add_d2h(nbytes);
        }
    }
    if (ok && use_ts_fluct) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(box->tau_21, d_tau, nbytes, cudaMemcpyDeviceToHost, stream));
        if (ok) {
            p21_cuda_stats_add_d2h(nbytes);
        }
    }
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(&h_ave_sum, d_scalar_sum, sizeof(double),
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(&h_nonfinite, d_nonfinite, sizeof(int),
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }

    p21_cuda_pool_free(d_bt, stream);
    p21_cuda_pool_free(d_tau, stream);
    p21_cuda_pool_free(d_block_sums, stream);
    p21_cuda_pool_free(d_scalar_sum, stream);
    p21_cuda_pool_free(d_nonfinite, stream);

    if (!ok) {
        return -1;
    }
    if (h_nonfinite != 0 || !isfinite(h_ave_sum)) {
        return InfinityorNaNError;
    }

    p21_cuda_stats_mut()->brightness_runs++;
    return 0;
}
