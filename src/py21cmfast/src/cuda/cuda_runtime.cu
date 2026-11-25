#include <mutex>
#include <string.h>

#include "cuda_common.cuh"

namespace {

std::mutex g_runtime_mu;
P21CudaStats g_stats = {};
int g_enabled = 0;
int g_fast_halo_sampler = 0;
bool g_initialized = false;
cudaStream_t g_stream = nullptr;
int g_device_id = -1;
int g_sm_count = 0;
size_t g_total_mem_bytes = 0;

}  // namespace

P21CudaStats *p21_cuda_stats_mut(void) { return &g_stats; }

void p21_cuda_stats_add_d2h(size_t nbytes) {
    g_stats.registry_d2h_bytes += (unsigned long long int)nbytes;
}

int p21_cuda_current_device(void) { return g_device_id; }

size_t p21_cuda_total_mem_bytes(void) { return g_total_mem_bytes; }

cudaStream_t p21_cuda_stream(void) { return g_stream; }

bool p21_cuda_ensure_init(void) {
    std::lock_guard<std::mutex> lock(g_runtime_mu);
    if (g_initialized) {
        return true;
    }

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        cudaGetLastError();
        return false;
    }

    int dev = 0;
    if (!P21_CUDA_OK(cudaGetDevice(&dev))) {
        return false;
    }
    if (!P21_CUDA_OK(cudaSetDevice(dev))) {
        return false;
    }

    int sm_count = 0;
    if (!P21_CUDA_OK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev))) {
        sm_count = 0;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (!P21_CUDA_OK(cudaMemGetInfo(&free_bytes, &total_bytes))) {
        total_bytes = 0;
    }

    cudaStream_t stream = nullptr;
    if (!P21_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))) {
        return false;
    }

    p21_cuda_pool_init(dev);

    g_device_id = dev;
    g_sm_count = sm_count;
    g_total_mem_bytes = total_bytes;
    g_stream = stream;
    g_initialized = true;
    return true;
}

extern "C" {

int p21_cuda_is_compiled(void) { return 1; }

int p21_cuda_is_available(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return count > 0 ? 1 : 0;
}

void p21_cuda_set_enabled(int enabled) { g_enabled = enabled ? 1 : 0; }

int p21_cuda_get_enabled(void) { return g_enabled; }

void p21_cuda_set_fast_halo_sampler(int enabled) { g_fast_halo_sampler = enabled ? 1 : 0; }

int p21_cuda_get_fast_halo_sampler(void) { return g_fast_halo_sampler; }

void p21_cuda_reset_stats(void) {
    std::lock_guard<std::mutex> lock(g_runtime_mu);
    memset(&g_stats, 0, sizeof(g_stats));
    g_stats.compiled = 1;
    g_stats.enabled = g_enabled;
    g_stats.device_id = g_device_id;
    g_stats.sm_count = g_sm_count;
    g_stats.total_mem_bytes = (unsigned long long int)g_total_mem_bytes;
}

void p21_cuda_get_stats(P21CudaStats *out) {
    if (!out) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_runtime_mu);
    *out = g_stats;
    out->compiled = 1;
    out->enabled = g_enabled;
    out->device_id = g_device_id;
    out->sm_count = g_sm_count;
    out->total_mem_bytes = (unsigned long long int)g_total_mem_bytes;
    if (g_initialized) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            out->free_mem_bytes = (unsigned long long int)free_bytes;
            out->total_mem_bytes = (unsigned long long int)total_bytes;
        } else {
            cudaGetLastError();
        }
    }
}

void p21_cuda_cleanup(void) {
    p21_cuda_registry_clear();
    p21_cuda_spin_temp_cleanup();
    p21_cuda_ionization_cleanup();
    p21_cuda_fft_clear_cache();
    p21_cuda_pool_trim();

    std::lock_guard<std::mutex> lock(g_runtime_mu);
    if (g_initialized) {
        if (g_stream) {
            cudaStreamSynchronize(g_stream);
            cudaStreamDestroy(g_stream);
            g_stream = nullptr;
        }
        g_initialized = false;
    }
    g_device_id = -1;
    g_sm_count = 0;
}

}  // extern "C"
