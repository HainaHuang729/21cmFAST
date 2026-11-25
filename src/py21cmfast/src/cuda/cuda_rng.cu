#include "cuda_rng.cuh"

#include <algorithm>
#include <limits.h>
#include <math.h>
#include <vector>

#include "cuda_common.cuh"

namespace {

constexpr int kBlockSize = 256;
constexpr size_t kMtBlockLen = 624;
constexpr int kChooseCandCap = 4096;

struct ChooseCandidate {
    uint32_t idx;
    uint32_t raw;
};

__global__ void mt19937_stream_kernel(uint32_t *state, uint32_t *out, int blocks, uint32_t seed,
                                      int init) {
    __shared__ uint32_t mt[624];
    int lane = threadIdx.x;
    if (init && lane == 0) {
        mt[0] = seed;
        for (int j = 1; j < 624; ++j) {
            mt[j] = 1812433253u * (mt[j - 1] ^ (mt[j - 1] >> 30)) + (uint32_t)j;
        }
    } else if (!init) {
        for (int j = lane; j < 624; j += blockDim.x) {
            mt[j] = state[j];
        }
    }
    __syncthreads();

    for (int block = 0; block < blocks; ++block) {
        for (int first = 0; first < 623; first += 227) {
            int j = first + lane;
            uint32_t next = 0;
            if (lane < 227 && j < 623) {
                uint32_t mix = (mt[j] & 0x80000000u) | (mt[j + 1] & 0x7fffffffu);
                next = mt[(j + 397) % 624] ^ (mix >> 1) ^ ((0u - (mix & 1u)) & 0x9908b0dfu);
            }
            __syncthreads();
            if (lane < 227 && j < 623) {
                mt[j] = next;
            }
            __syncthreads();
        }
        if (lane == 0) {
            uint32_t mix = (mt[623] & 0x80000000u) | (mt[0] & 0x7fffffffu);
            mt[623] = mt[396] ^ (mix >> 1) ^ ((0u - (mix & 1u)) & 0x9908b0dfu);
        }
        __syncthreads();
        for (int j = lane; j < 624; j += blockDim.x) {
            uint32_t v = mt[j];
            v ^= v >> 11;
            v ^= (v << 7) & 0x9d2c5680u;
            v ^= (v << 15) & 0xefc60000u;
            v ^= v >> 18;
            out[(size_t)block * 624 + (size_t)j] = v;
        }
        __syncthreads();
    }

    for (int j = lane; j < 624; j += blockDim.x) {
        state[j] = mt[j];
    }
}

__global__ void gfsr4_stream_kernel(uint32_t *ring, uint32_t *out, int count, uint32_t seed,
                                    int init, int offset) {
    int lane = threadIdx.x;
    if (init) {
        for (int j = lane; j < 16384; j += blockDim.x) {
            uint32_t v = seed;
            uint32_t base = 69069u;
            for (int skip = 32 * j; skip; skip >>= 1, base *= base) {
                if (skip & 1) {
                    v *= base;
                }
            }
            uint32_t word = 0;
            for (int b = 0; b < 32; ++b) {
                v *= 69069u;
                word = (word << 1) | (v >> 31);
            }
            if (j >= 7 && (j - 7) % 3 == 0 && (j - 7) / 3 < 32) {
                int b = (j - 7) / 3;
                word = (word & (0xffffffffu >> b)) | (0x80000000u >> b);
            }
            ring[j] = word;
        }
    }
    __syncthreads();

    for (int off = 0; off < count; off += blockDim.x) {
        int j = off + lane;
        int at = (33 + offset + j) & 16383;
        if (j < count) {
            uint32_t v = ring[(at - 471) & 16383] ^ ring[(at - 1586) & 16383] ^
                         ring[(at - 6988) & 16383] ^ ring[(at - 9689) & 16383];
            ring[at] = v;
            out[j] = v;
        }
        __syncthreads();
    }
}

__global__ void cmrg_stream_kernel(uint32_t *state, uint32_t *out, int count, uint32_t seed,
                                   int init) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const long long m1 = 2147483647LL;
    const long long m2 = 2145483479LL;
    long long x1, x2, x3, y1, y2, y3;
    int warm = 0;
    if (init) {
        uint32_t s = seed ? seed : 1u;
        s *= 69069u;
        x1 = (long long)(s % 2147483647u);
        s *= 69069u;
        x2 = (long long)(s % 2147483647u);
        s *= 69069u;
        x3 = (long long)(s % 2147483647u);
        s *= 69069u;
        y1 = (long long)(s % 2145483479u);
        s *= 69069u;
        y2 = (long long)(s % 2145483479u);
        s *= 69069u;
        y3 = (long long)(s % 2145483479u);
        warm = 7;
    } else {
        x1 = (long long)state[0];
        x2 = (long long)state[1];
        x3 = (long long)state[2];
        y1 = (long long)state[3];
        y2 = (long long)state[4];
        y3 = (long long)state[5];
    }

    for (int step = 0; step < warm + count; ++step) {
        uint32_t h3 = (uint32_t)x3 / 11714u;
        long long p3 = 183326LL * (x3 - (long long)h3 * 11714LL) - (long long)h3 * 2883LL;
        uint32_t h2 = (uint32_t)x2 / 33921u;
        long long p2 = 63308LL * (x2 - (long long)h2 * 33921LL) - (long long)h2 * 12979LL;
        if (p3 < 0) p3 += m1;
        if (p2 < 0) p2 += m1;
        x3 = x2;
        x2 = x1;
        x1 = p2 - p3;
        if (x1 < 0) x1 += m1;

        uint32_t k3 = (uint32_t)y3 / 3976u;
        long long q3 = 539608LL * (y3 - (long long)k3 * 3976LL) - (long long)k3 * 2071LL;
        uint32_t k1 = (uint32_t)y1 / 24919u;
        long long q1 = 86098LL * (y1 - (long long)k1 * 24919LL) - (long long)k1 * 7417LL;
        if (q3 < 0) q3 += m2;
        if (q1 < 0) q1 += m2;
        y3 = y2;
        y2 = y1;
        y1 = q1 - q3;
        if (y1 < 0) y1 += m2;

        long long ans = x1 - y1;
        if (ans < 0) ans += m1;
        if (step >= warm) {
            out[step - warm] = (uint32_t)ans;
        }
    }

    state[0] = (uint32_t)x1;
    state[1] = (uint32_t)x2;
    state[2] = (uint32_t)x3;
    state[3] = (uint32_t)y1;
    state[4] = (uint32_t)y2;
    state[5] = (uint32_t)y3;
}

__global__ void mrg_stream_kernel(uint32_t *state, uint32_t *out, int count, uint32_t seed,
                                  int init) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const long long m = 2147483647LL;
    long long x1, x2, x3, x4, x5;
    int warm = 0;
    if (init) {
        uint32_t s = seed ? seed : 1u;
        s *= 69069u;
        x1 = (long long)(s % 2147483647u);
        s *= 69069u;
        x2 = (long long)(s % 2147483647u);
        s *= 69069u;
        x3 = (long long)(s % 2147483647u);
        s *= 69069u;
        x4 = (long long)(s % 2147483647u);
        s *= 69069u;
        x5 = (long long)(s % 2147483647u);
        warm = 6;
    } else {
        x1 = (long long)state[0];
        x2 = (long long)state[1];
        x3 = (long long)state[2];
        x4 = (long long)state[3];
        x5 = (long long)state[4];
    }

    for (int step = 0; step < warm + count; ++step) {
        uint32_t h5 = (uint32_t)x5 / 20554u;
        long long p5 = 104480LL * (x5 - (long long)h5 * 20554LL) - (long long)h5 * 1727LL;
        if (p5 > 0) p5 -= m;
        uint32_t h1 = (uint32_t)x1 / 20u;
        long long p1 = 107374182LL * (x1 - (long long)h1 * 20LL) - (long long)h1 * 7LL;
        if (p1 < 0) p1 += m;
        p1 += p5;
        if (p1 < 0) p1 += m;
        x5 = x4;
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = p1;
        if (step >= warm) {
            out[step - warm] = (uint32_t)x1;
        }
    }

    state[0] = (uint32_t)x1;
    state[1] = (uint32_t)x2;
    state[2] = (uint32_t)x3;
    state[3] = (uint32_t)x4;
    state[4] = (uint32_t)x5;
}

__global__ void taus2_stream_kernel(uint32_t *state, uint32_t *out, int count, uint32_t seed,
                                    int init) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    uint32_t s1, s2, s3;
    int warm = 0;
    if (init) {
        uint32_t s = seed ? seed : 1u;
        s1 = 69069u * s;
        if (s1 < 2u) s1 += 2u;
        s2 = 69069u * s1;
        if (s2 < 8u) s2 += 8u;
        s3 = 69069u * s2;
        if (s3 < 16u) s3 += 16u;
        warm = 6;
    } else {
        s1 = state[0];
        s2 = state[1];
        s3 = state[2];
    }

    for (int step = 0; step < warm + count; ++step) {
        uint32_t b1 = ((s1 << 13) ^ s1) >> 19;
        s1 = ((s1 & 4294967294u) << 12) ^ b1;
        uint32_t b2 = ((s2 << 2) ^ s2) >> 25;
        s2 = ((s2 & 4294967288u) << 4) ^ b2;
        uint32_t b3 = ((s3 << 3) ^ s3) >> 11;
        s3 = ((s3 & 4294967280u) << 17) ^ b3;
        if (step >= warm) {
            out[step - warm] = s1 ^ s2 ^ s3;
        }
    }

    state[0] = s1;
    state[1] = s2;
    state[2] = s3;
}

__global__ void choose_candidates_kernel(const uint32_t *raw, uint32_t valid_len, uint32_t start,
                                         uint32_t total, uint32_t rem_k,
                                         ChooseCandidate *cands, uint32_t *cand_count) {
    uint32_t idx = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= valid_len) {
        return;
    }
    uint32_t t = start + idx;
    uint32_t r = raw[idx];
    double u = (double)r / 4294967296.0;
    if ((double)(total - t) * u < (double)rem_k) {
        uint32_t slot = atomicAdd(cand_count, 1u);
        if (slot < (uint32_t)kChooseCandCap) {
            cands[slot] = ChooseCandidate{idx, r};
        }
    }
}

__global__ void detect_zero_kernel(const uint32_t *raw, size_t n, uint32_t *has_zero) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && raw[idx] == 0u) {
        atomicExch(has_zero, 1u);
    }
}

__device__ static inline uint32_t block_reduce_sum_256(uint32_t val, uint32_t *s_data) {
    int tid = threadIdx.x;
    s_data[tid] = val;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    return s_data[0];
}

__device__ static inline uint32_t block_exclusive_scan_256(uint32_t val, uint32_t *s_data,
                                                           uint32_t *block_sum_out) {
    int tid = threadIdx.x;
    s_data[tid] = val;
    __syncthreads();
    for (int offset = 1; offset < 256; offset <<= 1) {
        uint32_t addend = (tid >= offset) ? s_data[tid - offset] : 0u;
        __syncthreads();
        s_data[tid] += addend;
        __syncthreads();
    }
    uint32_t total = s_data[255];
    if (block_sum_out) {
        *block_sum_out = total;
    }
    return s_data[tid] - val;
}

__global__ void scan_block_counts_kernel(const uint32_t *block_counts, uint32_t *block_offsets,
                                         size_t num_blocks, uint32_t *total_out) {
    __shared__ uint32_t s_data[256];
    int tid = threadIdx.x;
    uint32_t running = 0u;
    for (size_t base = 0; base < num_blocks; base += 256) {
        size_t pos = base + (size_t)tid;
        uint32_t val = (pos < num_blocks) ? block_counts[pos] : 0u;
        uint32_t chunk_sum = 0u;
        uint32_t excl = block_exclusive_scan_256(val, s_data, &chunk_sum);
        if (pos < num_blocks) {
            block_offsets[pos] = running + excl;
        }
        running += chunk_sum;
        __syncthreads();
    }
    if (tid == 0 && total_out) {
        *total_out = running;
    }
}

__global__ void nonzero_count_blocks_kernel(const uint32_t *raw, size_t n, uint32_t *block_counts) {
    __shared__ uint32_t s_data[256];
    size_t idx = (size_t)blockIdx.x * kBlockSize + threadIdx.x;
    uint32_t ok = (idx < n && raw[idx] != 0u) ? 1u : 0u;
    uint32_t sum = block_reduce_sum_256(ok, s_data);
    if (threadIdx.x == 0) {
        block_counts[blockIdx.x] = sum;
    }
}

__global__ void nonzero_scatter_kernel(const uint32_t *raw, size_t n,
                                       const uint32_t *block_offsets, uint32_t *dst) {
    __shared__ uint32_t s_data[256];
    size_t idx = (size_t)blockIdx.x * kBlockSize + threadIdx.x;
    uint32_t ok = (idx < n && raw[idx] != 0u) ? 1u : 0u;
    uint32_t excl = block_exclusive_scan_256(ok, s_data, nullptr);
    if (ok) {
        dst[block_offsets[blockIdx.x] + excl] = raw[idx];
    }
}

__global__ void box_muller_eval_kernel(const uint32_t *nz, size_t n_pairs, double denom,
                                       double *vals, uint8_t *valid_flags,
                                       uint32_t *block_counts) {
    __shared__ uint32_t s_data[256];
    size_t p = (size_t)blockIdx.x * kBlockSize + threadIdx.x;
    uint32_t ok = 0u;
    double out_val = 0.0;
    if (p < n_pairs) {
        uint32_t u0 = nz[2 * p];
        uint32_t u1 = nz[2 * p + 1];
        double x = -1.0 + 2.0 * ((double)u0 / denom);
        double y = -1.0 + 2.0 * ((double)u1 / denom);
        double r2 = x * x + y * y;
        if (r2 <= 1.0 && r2 != 0.0) {
            ok = 1u;
            out_val = y * sqrt(-2.0 * log(r2) / r2);
        }
        vals[p] = out_val;
        valid_flags[p] = (uint8_t)ok;
    }
    uint32_t sum = block_reduce_sum_256(ok, s_data);
    if (threadIdx.x == 0) {
        block_counts[blockIdx.x] = sum;
    }
}

__global__ void box_muller_scatter_kernel(const double *vals, const uint8_t *valid_flags,
                                          size_t n_pairs, const uint32_t *block_offsets,
                                          double *d_out, size_t written, size_t max_count) {
    __shared__ uint32_t s_data[256];
    size_t p = (size_t)blockIdx.x * kBlockSize + threadIdx.x;
    uint32_t ok = (p < n_pairs && valid_flags[p]) ? 1u : 0u;
    uint32_t excl = block_exclusive_scan_256(ok, s_data, nullptr);
    if (ok) {
        size_t dst_idx = written + (size_t)block_offsets[blockIdx.x] + (size_t)excl;
        if (dst_idx < max_count) {
            d_out[dst_idx] = vals[p];
        }
    }
}

double rng_denom_for_kind(int kind) {
    switch (kind) {
        case P21_RNG_CMRG:
        case P21_RNG_MRG:
            return 2147483647.0;
        default:
            return 4294967296.0;
    }
}

size_t rng_state_words_for_kind(int kind) {
    switch (kind) {
        case P21_RNG_MT19937:
            return 624;
        case P21_RNG_GFSR4:
            return 16384;
        case P21_RNG_CMRG:
            return 6;
        case P21_RNG_MRG:
            return 5;
        case P21_RNG_TAUS2:
            return 3;
        default:
            return 624;
    }
}

bool launch_rng_stream(int kind, uint32_t *d_state, uint32_t *d_raw, size_t raw_count,
                       uint32_t seed, int init, size_t offset, cudaStream_t stream) {
    switch (kind) {
        case P21_RNG_MT19937: {
            int blocks = (int)(raw_count / kMtBlockLen);
            mt19937_stream_kernel<<<1, 256, 0, stream>>>(d_state, d_raw, blocks, seed, init);
            break;
        }
        case P21_RNG_GFSR4:
            gfsr4_stream_kernel<<<1, 256, 0, stream>>>(d_state, d_raw, (int)raw_count, seed, init,
                                                       (int)(offset % 16384));
            break;
        case P21_RNG_CMRG:
            cmrg_stream_kernel<<<1, 1, 0, stream>>>(d_state, d_raw, (int)raw_count, seed, init);
            break;
        case P21_RNG_MRG:
            mrg_stream_kernel<<<1, 1, 0, stream>>>(d_state, d_raw, (int)raw_count, seed, init);
            break;
        case P21_RNG_TAUS2:
            taus2_stream_kernel<<<1, 1, 0, stream>>>(d_state, d_raw, (int)raw_count, seed, init);
            break;
        default:
            return false;
    }
    return P21_CUDA_OK(cudaGetLastError());
}

}  // namespace

bool p21_cuda_seed_rng_threads(unsigned long long random_seed, int n_threads, uint32_t *out_seeds,
                               cudaStream_t stream) {
    if (n_threads <= 0 || !out_seeds) {
        return false;
    }

    const uint32_t total = (uint32_t)(INT_MAX / 16);
    const size_t batch_blocks = 4096;
    const size_t batch_len = batch_blocks * kMtBlockLen;
    const uint32_t mt_seed = (random_seed == 0ULL) ? 4357u : (uint32_t)(random_seed & 0xFFFFFFFFULL);

    uint32_t *d_state = (uint32_t *)p21_cuda_pool_alloc(kMtBlockLen * sizeof(uint32_t), stream);
    uint32_t *d_raw = (uint32_t *)p21_cuda_pool_alloc(batch_len * sizeof(uint32_t), stream);
    ChooseCandidate *d_cands =
        (ChooseCandidate *)p21_cuda_pool_alloc(kChooseCandCap * sizeof(ChooseCandidate), stream);
    uint32_t *d_cand_count = (uint32_t *)p21_cuda_pool_alloc(sizeof(uint32_t), stream);

    if (!d_state || !d_raw || !d_cands || !d_cand_count) {
        p21_cuda_pool_free(d_cand_count, stream);
        p21_cuda_pool_free(d_cands, stream);
        p21_cuda_pool_free(d_raw, stream);
        p21_cuda_pool_free(d_state, stream);
        return false;
    }

    bool ok = true;
    int chosen_count = 0;
    uint32_t start = 0u;
    int init = 1;
    uint32_t shuffle_pos = 0u;
    uint32_t cur_batch_len = 0u;
    std::vector<ChooseCandidate> h_cands;

    while (chosen_count < n_threads && start < total) {
        mt19937_stream_kernel<<<1, 256, 0, stream>>>(d_state, d_raw, (int)batch_blocks, mt_seed,
                                                     init);
        init = 0;
        if (!P21_CUDA_OK(cudaGetLastError())) {
            ok = false;
            break;
        }

        uint32_t valid_len =
            (uint32_t)std::min((size_t)(total - start), batch_len);
        cur_batch_len = (uint32_t)batch_len;
        if (!P21_CUDA_OK(cudaMemsetAsync(d_cand_count, 0, sizeof(uint32_t), stream))) {
            ok = false;
            break;
        }

        int grid = (int)((valid_len + kBlockSize - 1) / kBlockSize);
        choose_candidates_kernel<<<grid, kBlockSize, 0, stream>>>(
            d_raw, valid_len, start, total, (uint32_t)(n_threads - chosen_count), d_cands,
            d_cand_count);
        if (!P21_CUDA_OK(cudaGetLastError())) {
            ok = false;
            break;
        }

        uint32_t h_cand_count = 0u;
        if (!P21_CUDA_OK(cudaMemcpyAsync(&h_cand_count, d_cand_count, sizeof(uint32_t),
                                         cudaMemcpyDeviceToHost, stream)) ||
            !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
            ok = false;
            break;
        }

        if (h_cand_count > (uint32_t)kChooseCandCap) {
            std::vector<uint32_t> h_raw(valid_len);
            if (!P21_CUDA_OK(cudaMemcpyAsync(h_raw.data(), d_raw, valid_len * sizeof(uint32_t),
                                             cudaMemcpyDeviceToHost, stream)) ||
                !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
                ok = false;
                break;
            }
            for (uint32_t idx = 0; idx < valid_len && chosen_count < n_threads; ++idx) {
                uint32_t t = start + idx;
                double u = (double)h_raw[idx] / 4294967296.0;
                if ((double)(total - t) * u < (double)(n_threads - chosen_count)) {
                    out_seeds[chosen_count++] = t;
                    if (chosen_count == n_threads) {
                        shuffle_pos = idx + 1;
                        break;
                    }
                }
            }
        } else if (h_cand_count > 0u) {
            h_cands.resize(h_cand_count);
            if (!P21_CUDA_OK(cudaMemcpyAsync(h_cands.data(), d_cands,
                                             h_cand_count * sizeof(ChooseCandidate),
                                             cudaMemcpyDeviceToHost, stream)) ||
                !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
                ok = false;
                break;
            }
            std::sort(h_cands.begin(), h_cands.end(),
                      [](const ChooseCandidate &a, const ChooseCandidate &b) {
                          return a.idx < b.idx;
                      });
            for (const ChooseCandidate &cand : h_cands) {
                uint32_t t = start + cand.idx;
                double u = (double)cand.raw / 4294967296.0;
                if ((double)(total - t) * u < (double)(n_threads - chosen_count)) {
                    out_seeds[chosen_count++] = t;
                    if (chosen_count == n_threads) {
                        shuffle_pos = cand.idx + 1;
                        break;
                    }
                }
            }
        }

        start += valid_len;
    }

    if (ok && chosen_count != n_threads) {
        ok = false;
    }

    /* Exact gsl_ran_shuffle(rseed, out_seeds, n_threads, sizeof(unsigned int)). */
    if (ok && n_threads > 1) {
        std::vector<uint32_t> h_tail;
        uint32_t tail_avail = (shuffle_pos < cur_batch_len) ? (cur_batch_len - shuffle_pos) : 0u;
        uint32_t copy_len = std::min(tail_avail, (uint32_t)kMtBlockLen);
        if (copy_len > 0u) {
            h_tail.resize(copy_len);
            if (!P21_CUDA_OK(cudaMemcpyAsync(h_tail.data(), d_raw + shuffle_pos,
                                             copy_len * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                             stream)) ||
                !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
                ok = false;
            }
        }

        uint32_t pos = 0u;
        for (int i = n_threads - 1; ok && i > 0; --i) {
            uint32_t scale = 0xFFFFFFFFu / (uint32_t)(i + 1);
            uint32_t j = 0u;
            do {
                if (pos >= h_tail.size()) {
                    mt19937_stream_kernel<<<1, 256, 0, stream>>>(d_state, d_raw, 1, mt_seed, 0);
                    h_tail.resize(kMtBlockLen);
                    if (!P21_CUDA_OK(cudaGetLastError()) ||
                        !P21_CUDA_OK(cudaMemcpyAsync(h_tail.data(), d_raw,
                                                     kMtBlockLen * sizeof(uint32_t),
                                                     cudaMemcpyDeviceToHost, stream)) ||
                        !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
                        ok = false;
                        break;
                    }
                    pos = 0u;
                }
                j = h_tail[pos++] / scale;
            } while (j > (uint32_t)i);
            if (ok) {
                std::swap(out_seeds[i], out_seeds[j]);
            }
        }
    }

    p21_cuda_pool_free(d_cand_count, stream);
    p21_cuda_pool_free(d_cands, stream);
    p21_cuda_pool_free(d_raw, stream);
    p21_cuda_pool_free(d_state, stream);
    return ok;
}

bool p21_cuda_fill_ugaussian_thread(int kind, uint32_t seed, double *d_out, size_t count,
                                    cudaStream_t stream) {
    if (count == 0) {
        return true;
    }
    if (!d_out) {
        return false;
    }

    const int rng_kind = ((kind % 5) + 5) % 5;
    const uint32_t eff_seed = (rng_kind == P21_RNG_MT19937 || rng_kind == P21_RNG_GFSR4)
                                  ? (seed == 0u ? 4357u : seed)
                                  : (seed == 0u ? 1u : seed);
    const double denom = rng_denom_for_kind(rng_kind);
    const size_t state_words = rng_state_words_for_kind(rng_kind);

    const size_t max_chunk = 4096 * kMtBlockLen;
    size_t first_chunk = (size_t)((double)count * 2.75) + 1248;
    first_chunk = ((first_chunk + kMtBlockLen - 1) / kMtBlockLen) * kMtBlockLen;
    const size_t cap_raw = std::min(max_chunk, std::max(kMtBlockLen, first_chunk));
    const size_t max_pairs = (cap_raw + 1) / 2;
    const size_t max_blocks = (std::max(cap_raw, max_pairs) + kBlockSize - 1) / kBlockSize;

    uint32_t *d_state = (uint32_t *)p21_cuda_pool_alloc(state_words * sizeof(uint32_t), stream);
    uint32_t *d_raw = (uint32_t *)p21_cuda_pool_alloc(cap_raw * sizeof(uint32_t), stream);
    uint32_t *d_nz = (uint32_t *)p21_cuda_pool_alloc((cap_raw + 1) * sizeof(uint32_t), stream);
    double *d_vals = (double *)p21_cuda_pool_alloc(max_pairs * sizeof(double), stream);
    uint8_t *d_valid = (uint8_t *)p21_cuda_pool_alloc(max_pairs * sizeof(uint8_t), stream);
    uint32_t *d_bcounts =
        (uint32_t *)p21_cuda_pool_alloc(max_blocks * sizeof(uint32_t), stream);
    uint32_t *d_boffsets =
        (uint32_t *)p21_cuda_pool_alloc(max_blocks * sizeof(uint32_t), stream);
    uint32_t *d_scalar = (uint32_t *)p21_cuda_pool_alloc(sizeof(uint32_t), stream);

    if (!d_state || !d_raw || !d_nz || !d_vals || !d_valid || !d_bcounts || !d_boffsets ||
        !d_scalar) {
        p21_cuda_pool_free(d_scalar, stream);
        p21_cuda_pool_free(d_boffsets, stream);
        p21_cuda_pool_free(d_bcounts, stream);
        p21_cuda_pool_free(d_valid, stream);
        p21_cuda_pool_free(d_vals, stream);
        p21_cuda_pool_free(d_nz, stream);
        p21_cuda_pool_free(d_raw, stream);
        p21_cuda_pool_free(d_state, stream);
        return false;
    }

    bool ok = true;
    size_t written = 0;
    size_t carry = 0;
    size_t raw_offset = 0;
    int init = 1;

    while (written < count && ok) {
        size_t rem = count - written;
        size_t raw_len = (size_t)((double)rem * 2.75) + 1248;
        raw_len = ((raw_len + kMtBlockLen - 1) / kMtBlockLen) * kMtBlockLen;
        raw_len = std::min(cap_raw, std::max(kMtBlockLen, raw_len));

        if (!launch_rng_stream(rng_kind, d_state, d_raw, raw_len, eff_seed, init, raw_offset,
                               stream)) {
            ok = false;
            break;
        }
        init = 0;
        raw_offset += raw_len;

        if (!P21_CUDA_OK(cudaMemsetAsync(d_scalar, 0, sizeof(uint32_t), stream))) {
            ok = false;
            break;
        }
        int zgrid = (int)((raw_len + kBlockSize - 1) / kBlockSize);
        detect_zero_kernel<<<zgrid, kBlockSize, 0, stream>>>(d_raw, raw_len, d_scalar);
        uint32_t h_has_zero = 0u;
        if (!P21_CUDA_OK(cudaGetLastError()) ||
            !P21_CUDA_OK(cudaMemcpyAsync(&h_has_zero, d_scalar, sizeof(uint32_t),
                                         cudaMemcpyDeviceToHost, stream)) ||
            !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
            ok = false;
            break;
        }

        size_t total_nz = carry;
        if (!h_has_zero) {
            if (!P21_CUDA_OK(cudaMemcpyAsync(d_nz + carry, d_raw, raw_len * sizeof(uint32_t),
                                             cudaMemcpyDeviceToDevice, stream))) {
                ok = false;
                break;
            }
            total_nz += raw_len;
        } else {
            nonzero_count_blocks_kernel<<<zgrid, kBlockSize, 0, stream>>>(d_raw, raw_len,
                                                                          d_bcounts);
            scan_block_counts_kernel<<<1, 256, 0, stream>>>(d_bcounts, d_boffsets, (size_t)zgrid,
                                                            d_scalar);
            nonzero_scatter_kernel<<<zgrid, kBlockSize, 0, stream>>>(d_raw, raw_len, d_boffsets,
                                                                     d_nz + carry);
            uint32_t h_nz_added = 0u;
            if (!P21_CUDA_OK(cudaGetLastError()) ||
                !P21_CUDA_OK(cudaMemcpyAsync(&h_nz_added, d_scalar, sizeof(uint32_t),
                                             cudaMemcpyDeviceToHost, stream)) ||
                !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
                ok = false;
                break;
            }
            total_nz += (size_t)h_nz_added;
        }

        size_t n_pairs = total_nz / 2;
        size_t next_carry = total_nz % 2;
        if (n_pairs > 0) {
            int pgrid = (int)((n_pairs + kBlockSize - 1) / kBlockSize);
            box_muller_eval_kernel<<<pgrid, kBlockSize, 0, stream>>>(d_nz, n_pairs, denom, d_vals,
                                                                     d_valid, d_bcounts);
            scan_block_counts_kernel<<<1, 256, 0, stream>>>(d_bcounts, d_boffsets, (size_t)pgrid,
                                                            d_scalar);
            box_muller_scatter_kernel<<<pgrid, kBlockSize, 0, stream>>>(
                d_vals, d_valid, n_pairs, d_boffsets, d_out, written, count);
            uint32_t h_valid_pairs = 0u;
            if (!P21_CUDA_OK(cudaGetLastError()) ||
                !P21_CUDA_OK(cudaMemcpyAsync(&h_valid_pairs, d_scalar, sizeof(uint32_t),
                                             cudaMemcpyDeviceToHost, stream)) ||
                !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
                ok = false;
                break;
            }
            written += std::min((size_t)h_valid_pairs, rem);
        }

        if (next_carry == 1 && written < count) {
            if (!P21_CUDA_OK(cudaMemcpyAsync(d_nz, d_nz + total_nz - 1, sizeof(uint32_t),
                                             cudaMemcpyDeviceToDevice, stream))) {
                ok = false;
                break;
            }
        }
        carry = next_carry;
    }

    p21_cuda_pool_free(d_scalar, stream);
    p21_cuda_pool_free(d_boffsets, stream);
    p21_cuda_pool_free(d_bcounts, stream);
    p21_cuda_pool_free(d_valid, stream);
    p21_cuda_pool_free(d_vals, stream);
    p21_cuda_pool_free(d_nz, stream);
    p21_cuda_pool_free(d_raw, stream);
    p21_cuda_pool_free(d_state, stream);
    return ok;
}
