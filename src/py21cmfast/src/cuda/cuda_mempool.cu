#include <algorithm>
#include <mutex>
#include <string.h>
#include <unordered_map>

#include "cuda_common.cuh"

namespace {

bool g_pool_supported = false;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
cudaMemPool_t g_mem_pool = nullptr;
#endif

struct RegistryEntry {
    void *d_ptr;
    size_t nbytes;
    uint64_t hash;
    uint64_t lru_tick;
};

std::mutex g_reg_mu;
std::unordered_map<const void *, RegistryEntry> g_registry;
size_t g_registry_bytes = 0;
uint64_t g_lru_counter = 0;

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t read_u64_unaligned(const uint8_t *p) {
    uint64_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

/* Full-buffer 4-lane fingerprint to prevent stale-cache and ABA collisions. */
uint64_t p21_hash_host_bytes(const void *ptr, size_t nbytes) {
    const uint8_t *p = (const uint8_t *)ptr;
    uint64_t h0 = 0x243f6a8885a308d3ULL ^ (uint64_t)nbytes;
    uint64_t h1 = 0x13198a2e03707344ULL ^ (uint64_t)nbytes;
    uint64_t h2 = 0xa4093822299f31d0ULL;
    uint64_t h3 = 0x082efa98ec4e6c89ULL;

    size_t i = 0;
    for (; i + 32 <= nbytes; i += 32) {
        h0 = (h0 ^ read_u64_unaligned(p + i)) * 0x9e3779b97f4a7c15ULL;
        h1 = (h1 ^ read_u64_unaligned(p + i + 8)) * 0xbf58476d1ce4e5b9ULL;
        h2 = (h2 ^ read_u64_unaligned(p + i + 16)) * 0x94d049bb133111ebULL;
        h3 = (h3 ^ read_u64_unaligned(p + i + 24)) * 0xd6e8feb86659fd93ULL;
    }
    for (; i + 8 <= nbytes; i += 8) {
        h0 = splitmix64(h0 ^ read_u64_unaligned(p + i));
    }
    uint64_t tail = 0;
    for (size_t b = 0; i + b < nbytes; b++) {
        tail |= ((uint64_t)p[i + b]) << (8 * b);
    }
    if (i < nbytes) {
        h1 = splitmix64(h1 ^ tail);
    }
    return splitmix64(h0 ^ ((h1 << 17) | (h1 >> 47)) ^ h2 ^ ((h3 << 31) | (h3 >> 33)));
}

size_t registry_budget_bytes(void) {
    size_t total = p21_cuda_total_mem_bytes();
    if (total == 0) {
        return 1024ULL << 20;
    }
    size_t half = total / 2;
    size_t four_gib = 4ULL << 30;
    return std::min(half, four_gib);
}

void evict_lru_locked(size_t incoming_bytes, const void *keep_host_ptr, cudaStream_t stream) {
    size_t budget = registry_budget_bytes();
    while (!g_registry.empty() && g_registry_bytes + incoming_bytes > budget) {
        auto victim = g_registry.end();
        uint64_t best_tick = UINT64_MAX;
        for (auto it = g_registry.begin(); it != g_registry.end(); ++it) {
            if (it->first == keep_host_ptr) {
                continue;
            }
            if (it->second.lru_tick < best_tick) {
                best_tick = it->second.lru_tick;
                victim = it;
            }
        }
        if (victim == g_registry.end()) {
            break;
        }
        void *d_old = victim->second.d_ptr;
        g_registry_bytes -= victim->second.nbytes;
        g_registry.erase(victim);
        p21_cuda_pool_free(d_old, stream);
        p21_cuda_stats_mut()->registry_invalidations++;
    }
}

}  // namespace

void p21_cuda_pool_init(int device_id) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    int pool_attr = 0;
    if (cudaDeviceGetAttribute(&pool_attr, cudaDevAttrMemoryPoolsSupported, device_id) ==
            cudaSuccess &&
        pool_attr > 0) {
        cudaMemPool_t pool = nullptr;
        if (cudaDeviceGetDefaultMemPool(&pool, device_id) == cudaSuccess && pool != nullptr) {
            uint64_t threshold = UINT64_MAX;
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
            cudaGetLastError();
            g_mem_pool = pool;
            g_pool_supported = true;
            return;
        }
    }
    cudaGetLastError();
    g_mem_pool = nullptr;
    g_pool_supported = false;
#else
    (void)device_id;
    g_pool_supported = false;
#endif
}

void *p21_cuda_pool_alloc(size_t nbytes, cudaStream_t stream) {
    size_t bytes = nbytes > 0 ? nbytes : 256;
    void *ptr = nullptr;
    cudaError_t err;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    if (g_pool_supported && g_mem_pool != nullptr && stream != nullptr) {
        err = cudaMallocFromPoolAsync(&ptr, bytes, g_mem_pool, stream);
    } else if (stream != nullptr) {
        err = cudaMallocAsync(&ptr, bytes, stream);
        if (err != cudaSuccess) {
            cudaGetLastError();
            err = cudaMalloc(&ptr, bytes);
        }
    } else {
        err = cudaMalloc(&ptr, bytes);
    }
#else
    (void)stream;
    err = cudaMalloc(&ptr, bytes);
#endif
    if (!P21_CUDA_OK(err)) {
        return nullptr;
    }
    P21CudaStats *st = p21_cuda_stats_mut();
    st->mempool_alloc_calls++;
    st->mempool_bytes_allocated += (unsigned long long int)bytes;
    return ptr;
}

void p21_cuda_pool_free(void *ptr, cudaStream_t stream) {
    if (!ptr) {
        return;
    }
    p21_cuda_stats_mut()->mempool_free_calls++;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    if (g_pool_supported && stream != nullptr) {
        if (cudaFreeAsync(ptr, stream) == cudaSuccess) {
            return;
        }
        cudaGetLastError();
    }
#else
    (void)stream;
#endif
    cudaFree(ptr);
}

void p21_cuda_pool_trim(void) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    if (g_pool_supported && g_mem_pool != nullptr) {
        cudaDeviceSynchronize();
        cudaMemPoolTrimTo(g_mem_pool, 0);
        cudaGetLastError();
    }
#endif
}

void *p21_cuda_registry_get_or_upload(const void *host_ptr, size_t nbytes, cudaStream_t stream) {
    if (!host_ptr || nbytes == 0) {
        return nullptr;
    }
    uint64_t hash = p21_hash_host_bytes(host_ptr, nbytes);

    {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        auto it = g_registry.find(host_ptr);
        if (it != g_registry.end()) {
            if (it->second.nbytes == nbytes && it->second.hash == hash) {
                it->second.lru_tick = ++g_lru_counter;
                p21_cuda_stats_mut()->registry_hits++;
                return it->second.d_ptr;
            }
            void *d_stale = it->second.d_ptr;
            g_registry_bytes -= it->second.nbytes;
            g_registry.erase(it);
            p21_cuda_pool_free(d_stale, stream);
            p21_cuda_stats_mut()->registry_invalidations++;
        }
    }

    void *d_buf = p21_cuda_pool_alloc(nbytes, stream);
    if (!d_buf) {
        return nullptr;
    }
    if (!P21_CUDA_OK(cudaMemcpyAsync(d_buf, host_ptr, nbytes, cudaMemcpyHostToDevice, stream))) {
        p21_cuda_pool_free(d_buf, stream);
        return nullptr;
    }

    P21CudaStats *st = p21_cuda_stats_mut();
    st->registry_misses++;
    st->registry_h2d_bytes += (unsigned long long int)nbytes;

    std::lock_guard<std::mutex> lock(g_reg_mu);
    auto existing = g_registry.find(host_ptr);
    if (existing != g_registry.end()) {
        p21_cuda_pool_free(existing->second.d_ptr, stream);
        g_registry_bytes -= existing->second.nbytes;
        g_registry.erase(existing);
    }
    evict_lru_locked(nbytes, host_ptr, stream);
    g_registry.emplace(host_ptr, RegistryEntry{d_buf, nbytes, hash, ++g_lru_counter});
    g_registry_bytes += nbytes;
    return d_buf;
}

bool p21_cuda_registry_put_copy(const void *host_ptr, const void *d_src, size_t nbytes,
                                cudaStream_t stream) {
    if (!host_ptr || !d_src || nbytes == 0) {
        return false;
    }
    /* Compute hash after the stream finishes D2H copy so host_ptr matches device data. */
    uint64_t hash = p21_hash_host_bytes(host_ptr, nbytes);

    void *d_cached = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        auto it = g_registry.find(host_ptr);
        if (it != g_registry.end()) {
            if (it->second.nbytes == nbytes) {
                d_cached = it->second.d_ptr;
            } else {
                p21_cuda_pool_free(it->second.d_ptr, stream);
                g_registry_bytes -= it->second.nbytes;
                g_registry.erase(it);
                p21_cuda_stats_mut()->registry_invalidations++;
            }
        }
    }

    if (!d_cached) {
        d_cached = p21_cuda_pool_alloc(nbytes, stream);
        if (!d_cached) {
            return false;
        }
    }

    if (!P21_CUDA_OK(cudaMemcpyAsync(d_cached, d_src, nbytes, cudaMemcpyDeviceToDevice, stream))) {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        auto it = g_registry.find(host_ptr);
        if (it == g_registry.end() || it->second.d_ptr != d_cached) {
            p21_cuda_pool_free(d_cached, stream);
        }
        return false;
    }

    std::lock_guard<std::mutex> lock(g_reg_mu);
    auto it = g_registry.find(host_ptr);
    if (it != g_registry.end()) {
        if (it->second.d_ptr != d_cached) {
            p21_cuda_pool_free(it->second.d_ptr, stream);
            g_registry_bytes -= it->second.nbytes;
            it->second.d_ptr = d_cached;
            it->second.nbytes = nbytes;
            g_registry_bytes += nbytes;
        }
        it->second.hash = hash;
        it->second.lru_tick = ++g_lru_counter;
    } else {
        evict_lru_locked(nbytes, host_ptr, stream);
        g_registry.emplace(host_ptr, RegistryEntry{d_cached, nbytes, hash, ++g_lru_counter});
        g_registry_bytes += nbytes;
    }
    return true;
}

bool p21_cuda_registry_put_owned(const void *host_ptr, void *d_buf, size_t nbytes,
                                 cudaStream_t stream) {
    if (!host_ptr || !d_buf || nbytes == 0) {
        return false;
    }
    /* Caller guarantees host_ptr already reflects completed D2H contents. */
    uint64_t hash = p21_hash_host_bytes(host_ptr, nbytes);

    std::lock_guard<std::mutex> lock(g_reg_mu);
    auto existing = g_registry.find(host_ptr);
    if (existing != g_registry.end()) {
        if (existing->second.d_ptr != d_buf) {
            p21_cuda_pool_free(existing->second.d_ptr, stream);
        }
        g_registry_bytes -= existing->second.nbytes;
        g_registry.erase(existing);
        p21_cuda_stats_mut()->registry_invalidations++;
    }
    evict_lru_locked(nbytes, host_ptr, stream);
    g_registry.emplace(host_ptr, RegistryEntry{d_buf, nbytes, hash, ++g_lru_counter});
    g_registry_bytes += nbytes;
    return true;
}

extern "C" {

void p21_cuda_invalidate_host_ptr(const void *ptr) {
    if (!ptr) {
        return;
    }
    void *d_old = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        if (!g_registry.empty()) {
            auto it = g_registry.find(ptr);
            if (it != g_registry.end()) {
                d_old = it->second.d_ptr;
                g_registry_bytes -= it->second.nbytes;
                g_registry.erase(it);
                p21_cuda_stats_mut()->registry_invalidations++;
            }
        }
    }
    if (d_old) {
        p21_cuda_pool_free(d_old, p21_cuda_stream());
    }
    p21_cuda_radiation_invalidate_host_ptr(ptr);
}

void p21_cuda_registry_clear(void) {
    {
        std::lock_guard<std::mutex> lock(g_reg_mu);
        cudaStream_t stream = p21_cuda_stream();
        for (auto &kv : g_registry) {
            p21_cuda_pool_free(kv.second.d_ptr, stream);
            p21_cuda_stats_mut()->registry_invalidations++;
        }
        g_registry.clear();
        g_registry_bytes = 0;
    }
    p21_cuda_radiation_clear();
}

}  // extern "C"
