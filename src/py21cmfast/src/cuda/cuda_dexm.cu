#include <math.h>
#include <stdint.h>
#include <string.h>

#include "cuda_common.cuh"

extern "C" {
#include "../hmf.h"
}

struct P21CudaDexmCtx {
    int grid_dim = 0;
    int z_dim = 0;
    size_t total_pixels = 0;
    unsigned long long int total_halo_num = 0;
    uint8_t *d_in_halo = nullptr;
    float *d_found_masses = nullptr;
    float *d_found_coords = nullptr;
    cudaStream_t stream = nullptr;
};

namespace {

constexpr int kBlockThreads = 256;
constexpr int kMaxCompactBlocks = 1024;

__device__ static inline int p21_pixel_in_halo(int grid_dim, int z_dim, int x, int x_index, int y,
                                               int y_index, int z, int z_index,
                                               float Rsq_curr_index) {
    float xsq = (float)(x - x_index) * (float)(x - x_index);
    float ysq = (float)(y - y_index) * (float)(y - y_index);
    float zsq = (float)(z - z_index) * (float)(z - z_index);

    float xp = (float)(x - x_index + grid_dim);
    float yp = (float)(y - y_index + grid_dim);
    float zp = (float)(z - z_index + z_dim);
    float xm = (float)(x - x_index - grid_dim);
    float ym = (float)(y - y_index - grid_dim);
    float zm = (float)(z - z_index - z_dim);

    float xs[3] = {xsq, xp * xp, xm * xm};
    float ys[3] = {ysq, yp * yp, ym * ym};
    float zs[3] = {zsq, zp * zp, zm * zm};

    for (int ix = 0; ix < 3; ++ix) {
        for (int iy = 0; iy < 3; ++iy) {
            for (int iz = 0; iz < 3; ++iz) {
                if (Rsq_curr_index > (xs[ix] + ys[iy] + zs[iz])) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

__global__ void dexm_paint_sphere_all_halos_kernel(
    uint64_t n_halos, int grid_dim, int z_dim, float box_len, float extra_R,
    const int *found_x, const int *found_y, const int *found_z, const float *found_R,
    uint8_t *forbidden) {
    uint64_t h = blockIdx.x;
    if (h >= n_halos) {
        return;
    }

    int hx = found_x[h];
    int hy = found_y[h];
    int hz = found_z[h];
    float R_eff = found_R[h] + extra_R;
    float R_grid = R_eff / box_len * (float)grid_dim;
    int R_idx = (int)ceilf(R_grid);
    float Rsq = R_grid * R_grid;

    int span = 2 * R_idx + 1;
    uint64_t total_vox = (uint64_t)span * (uint64_t)span * (uint64_t)span;
    for (uint64_t v = threadIdx.x; v < total_vox; v += blockDim.x) {
        int dz = (int)(v % (uint64_t)span) - R_idx;
        uint64_t q = v / (uint64_t)span;
        int dy = (int)(q % (uint64_t)span) - R_idx;
        int dx = (int)(q / (uint64_t)span) - R_idx;

        int xi = p21_wrap_coord(hx + dx, grid_dim);
        int yi = p21_wrap_coord(hy + dy, grid_dim);
        int zi = p21_wrap_coord(hz + dz, z_dim);
        if (p21_pixel_in_halo(grid_dim, z_dim, hx, xi, hy, yi, hz, zi, Rsq)) {
            forbidden[p21_grid_idx(xi, yi, zi, grid_dim, z_dim)] = 1;
        }
    }
}

__global__ void dexm_count_cands_kernel(size_t total, size_t chunk, const float *delta_m_raw,
                                        float gf_over_totpix, float delta_crit, int opt_mode,
                                        const uint8_t *in_halo, const uint8_t *forbidden,
                                        uint64_t *blk_counts) {
    __shared__ uint64_t sh[kBlockThreads];
    size_t tid = threadIdx.x;
    size_t base = (size_t)blockIdx.x * chunk;
    size_t limit = base + chunk;
    if (limit > total) limit = total;

    uint64_t cnt = 0;
    const uint8_t *mask = opt_mode ? forbidden : in_halo;
    for (size_t idx = base + tid; idx < limit; idx += kBlockThreads) {
        float dm = delta_m_raw[idx] * gf_over_totpix;
        if (dm > delta_crit && mask[idx] == 0) {
            cnt++;
        }
    }
    sh[tid] = cnt;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            sh[tid] += sh[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        blk_counts[blockIdx.x] = sh[0];
    }
}

__global__ void dexm_scan_blocks_kernel(int nblocks, const uint64_t *blk_counts,
                                        uint64_t *blk_offsets, uint64_t *total_out) {
    __shared__ uint64_t sh[kMaxCompactBlocks];
    int tid = (int)threadIdx.x;
    sh[tid] = (tid < nblocks) ? blk_counts[tid] : 0ULL;
    __syncthreads();

    for (int offset = 1; offset < kMaxCompactBlocks; offset <<= 1) {
        uint64_t val = (tid >= offset) ? sh[tid - offset] : 0ULL;
        __syncthreads();
        sh[tid] += val;
        __syncthreads();
    }
    if (tid < nblocks) {
        blk_offsets[tid] = (tid == 0) ? 0ULL : sh[tid - 1];
    }
    if (tid == 0) {
        *total_out = (nblocks > 0) ? sh[nblocks - 1] : 0ULL;
    }
}

__global__ void dexm_write_cands_kernel(size_t total, size_t chunk, const float *delta_m_raw,
                                        float gf_over_totpix, float delta_crit, int opt_mode,
                                        const uint8_t *in_halo, const uint8_t *forbidden,
                                        const uint64_t *blk_offsets, uint64_t *cand_idx_out) {
    __shared__ int s_scan[kBlockThreads];
    int tid = (int)threadIdx.x;
    size_t base = (size_t)blockIdx.x * chunk;
    size_t limit = base + chunk;
    if (limit > total) limit = total;

    uint64_t pos = blk_offsets[blockIdx.x];
    const uint8_t *mask = opt_mode ? forbidden : in_halo;

    for (size_t off = 0; base + off < limit; off += kBlockThreads) {
        size_t idx = base + off + (size_t)tid;
        int pred = 0;
        if (idx < limit) {
            float dm = delta_m_raw[idx] * gf_over_totpix;
            if (dm > delta_crit && mask[idx] == 0) {
                pred = 1;
            }
        }
        s_scan[tid] = pred;
        __syncthreads();

        for (int s = 1; s < kBlockThreads; s <<= 1) {
            int add = (tid >= s) ? s_scan[tid - s] : 0;
            __syncthreads();
            s_scan[tid] += add;
            __syncthreads();
        }
        int incl = s_scan[tid];
        int excl = incl - pred;
        int tile_total = s_scan[kBlockThreads - 1];
        if (pred) {
            cand_idx_out[pos + (uint64_t)excl] = (uint64_t)idx;
        }
        pos += (uint64_t)tile_total;
        __syncthreads();
    }
}

__global__ void dexm_accept_candidates_kernel(
    uint64_t n_cands, const uint64_t *cand_idx, int grid_dim, int z_dim, float M, float R,
    float R_ov, float R_forb, float box_len, int opt_mode, uint8_t *in_halo, uint8_t *forbidden,
    float *halo_catalog, int *found_x, int *found_y, int *found_z, float *found_R,
    uint64_t *d_total_halo_num) {
    __shared__ int s_skip;
    __shared__ int s_overlap;

    int tid = (int)threadIdx.x;
    uint64_t n_found = *d_total_halo_num;

    const float R_paint_grid = R / box_len * (float)grid_dim;
    const int R_paint_idx = (int)ceilf(R_paint_grid);
    const float Rsq_paint = R_paint_grid * R_paint_grid;
    const int paint_span = 2 * R_paint_idx + 1;
    const uint64_t paint_vox =
        (uint64_t)paint_span * (uint64_t)paint_span * (uint64_t)paint_span;

    const float R_ov_grid = R_ov / box_len * (float)grid_dim;
    const int R_ov_idx = (int)ceilf(R_ov_grid);
    const float Rsq_ov = R_ov_grid * R_ov_grid;
    const int ov_span = 2 * R_ov_idx + 1;
    const uint64_t ov_vox = (uint64_t)ov_span * (uint64_t)ov_span * (uint64_t)ov_span;

    const float R_forb_grid = R_forb / box_len * (float)grid_dim;
    const int R_forb_idx = (int)ceilf(R_forb_grid);
    const float Rsq_forb = R_forb_grid * R_forb_grid;
    const int forb_span = 2 * R_forb_idx + 1;
    const uint64_t forb_vox =
        (uint64_t)forb_span * (uint64_t)forb_span * (uint64_t)forb_span;

    const uint64_t yz = (uint64_t)grid_dim * (uint64_t)z_dim;

    for (uint64_t c = 0; c < n_cands; ++c) {
        uint64_t idx = cand_idx[c];
        if (tid == 0) {
            s_skip = (opt_mode ? (forbidden[idx] != 0) : (in_halo[idx] != 0)) ? 1 : 0;
            s_overlap = 0;
        }
        __syncthreads();
        if (s_skip) {
            continue;
        }

        int x = (int)(idx / yz);
        uint64_t rem = idx - (uint64_t)x * yz;
        int y = (int)(rem / (uint64_t)z_dim);
        int z = (int)(rem - (uint64_t)y * (uint64_t)z_dim);

        if (!opt_mode) {
            for (uint64_t v = (uint64_t)tid; v < ov_vox; v += blockDim.x) {
                if (*((volatile int *)&s_overlap)) break;
                int dz = (int)(v % (uint64_t)ov_span) - R_ov_idx;
                uint64_t q = v / (uint64_t)ov_span;
                int dy = (int)(q % (uint64_t)ov_span) - R_ov_idx;
                int dx = (int)(q / (uint64_t)ov_span) - R_ov_idx;

                int xi = p21_wrap_coord(x + dx, grid_dim);
                int yi = p21_wrap_coord(y + dy, grid_dim);
                int zi = p21_wrap_coord(z + dz, z_dim);
                uint64_t cidx = (uint64_t)p21_grid_idx(xi, yi, zi, grid_dim, z_dim);
                if (in_halo[cidx] && p21_pixel_in_halo(grid_dim, z_dim, x, xi, y, yi, z, zi, Rsq_ov)) {
                    atomicExch(&s_overlap, 1);
                    break;
                }
            }
            __syncthreads();
            if (s_overlap) {
                continue;
            }
        }

        for (uint64_t v = (uint64_t)tid; v < paint_vox; v += blockDim.x) {
            int dz = (int)(v % (uint64_t)paint_span) - R_paint_idx;
            uint64_t q = v / (uint64_t)paint_span;
            int dy = (int)(q % (uint64_t)paint_span) - R_paint_idx;
            int dx = (int)(q / (uint64_t)paint_span) - R_paint_idx;

            int xi = p21_wrap_coord(x + dx, grid_dim);
            int yi = p21_wrap_coord(y + dy, grid_dim);
            int zi = p21_wrap_coord(z + dz, z_dim);
            uint64_t cidx = (uint64_t)p21_grid_idx(xi, yi, zi, grid_dim, z_dim);
            if (!in_halo[cidx] &&
                p21_pixel_in_halo(grid_dim, z_dim, x, xi, y, yi, z, zi, Rsq_paint)) {
                in_halo[cidx] = 1;
            }
        }

        if (opt_mode) {
            for (uint64_t v = (uint64_t)tid; v < forb_vox; v += blockDim.x) {
                int dz = (int)(v % (uint64_t)forb_span) - R_forb_idx;
                uint64_t q = v / (uint64_t)forb_span;
                int dy = (int)(q % (uint64_t)forb_span) - R_forb_idx;
                int dx = (int)(q / (uint64_t)forb_span) - R_forb_idx;

                int xi = p21_wrap_coord(x + dx, grid_dim);
                int yi = p21_wrap_coord(y + dy, grid_dim);
                int zi = p21_wrap_coord(z + dz, z_dim);
                uint64_t cidx = (uint64_t)p21_grid_idx(xi, yi, zi, grid_dim, z_dim);
                if (!forbidden[cidx] &&
                    p21_pixel_in_halo(grid_dim, z_dim, x, xi, y, yi, z, zi, Rsq_forb)) {
                    forbidden[cidx] = 1;
                }
            }
        }

        if (tid == 0) {
            halo_catalog[idx] = M;
            found_x[n_found] = x;
            found_y[n_found] = y;
            found_z[n_found] = z;
            found_R[n_found] = R;
        }
        n_found++;
        __syncthreads();
    }

    if (tid == 0) {
        *d_total_halo_num = n_found;
    }
}

__global__ void dexm_count_halos_kernel(size_t total, size_t chunk, const float *halo_catalog,
                                         uint64_t *blk_counts) {
    __shared__ uint64_t sh[kBlockThreads];
    size_t tid = threadIdx.x;
    size_t base = (size_t)blockIdx.x * chunk;
    size_t limit = base + chunk;
    if (limit > total) limit = total;

    uint64_t cnt = 0;
    for (size_t idx = base + tid; idx < limit; idx += kBlockThreads) {
        if (halo_catalog[idx] > 0.0f) {
            cnt++;
        }
    }
    sh[tid] = cnt;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            sh[tid] += sh[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        blk_counts[blockIdx.x] = sh[0];
    }
}

__global__ void dexm_pack_halos_kernel(size_t total, size_t chunk, int grid_dim, int z_dim,
                                       double cell_length, const float *halo_catalog,
                                       const uint64_t *blk_offsets, float *out_masses,
                                       float *out_coords) {
    __shared__ int s_scan[kBlockThreads];
    int tid = (int)threadIdx.x;
    size_t base = (size_t)blockIdx.x * chunk;
    size_t limit = base + chunk;
    if (limit > total) limit = total;

    uint64_t pos = blk_offsets[blockIdx.x];
    const size_t yz = (size_t)grid_dim * (size_t)z_dim;

    for (size_t off = 0; base + off < limit; off += kBlockThreads) {
        size_t idx = base + off + (size_t)tid;
        float m = 0.0f;
        int pred = 0;
        if (idx < limit) {
            m = halo_catalog[idx];
            if (m > 0.0f) {
                pred = 1;
            }
        }
        s_scan[tid] = pred;
        __syncthreads();

        for (int s = 1; s < kBlockThreads; s <<= 1) {
            int add = (tid >= s) ? s_scan[tid - s] : 0;
            __syncthreads();
            s_scan[tid] += add;
            __syncthreads();
        }
        int excl = s_scan[tid] - pred;
        int tile_total = s_scan[kBlockThreads - 1];
        if (pred) {
            uint64_t dst = pos + (uint64_t)excl;
            int x = (int)(idx / yz);
            size_t rem = idx - (size_t)x * yz;
            int y = (int)(rem / (size_t)z_dim);
            int z = (int)(rem - (size_t)y * (size_t)z_dim);
            out_masses[dst] = m;
            out_coords[3 * dst + 0] = (float)((double)x * cell_length);
            out_coords[3 * dst + 1] = (float)((double)y * cell_length);
            out_coords[3 * dst + 2] = (float)((double)z * cell_length);
        }
        pos += (uint64_t)tile_total;
        __syncthreads();
    }
}

__global__ void dexm_u8_to_float_kernel(size_t n, const uint8_t *in_halo, float *out_f) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out_f[idx] = in_halo[idx] ? 1.0f : 0.0f;
    }
}

__global__ void dexm_downsample_overlap_kernel(int lo_nx, int lo_nz, int hi_nx, int hi_nz,
                                               float f_pixel_factor, double inv_tot_pix,
                                               const float *hi_filtered, float *overlap_out) {
    size_t lo_total = (size_t)lo_nx * (size_t)lo_nx * (size_t)lo_nz;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lo_total) {
        return;
    }
    size_t lo_yz = (size_t)lo_nx * (size_t)lo_nz;
    int i = (int)(idx / lo_yz);
    size_t rem = idx - (size_t)i * lo_yz;
    int j = (int)(rem / (size_t)lo_nz);
    int k = (int)(rem - (size_t)j * (size_t)lo_nz);

    int hi_i = (int)(i * f_pixel_factor + 0.5f);
    int hi_j = (int)(j * f_pixel_factor + 0.5f);
    int hi_k = (int)(k * f_pixel_factor + 0.5f);
    if (hi_i >= hi_nx) hi_i = hi_nx - 1;
    if (hi_j >= hi_nx) hi_j = hi_nx - 1;
    if (hi_k >= hi_nz) hi_k = hi_nz - 1;

    double val = (double)hi_filtered[p21_grid_idx(hi_i, hi_j, hi_k, hi_nx, hi_nz)] * inv_tot_pix;
    if (val < 0.0) val = 0.0;
    if (val > 1.0) val = 1.0;
    overlap_out[idx] = (float)val;
}

void compute_chunk_grid(size_t total, int *nblocks_out, size_t *chunk_out) {
    int max_needed = (int)((total + kBlockThreads - 1) / kBlockThreads);
    int nblocks = max_needed < kMaxCompactBlocks ? max_needed : kMaxCompactBlocks;
    if (nblocks < 1) nblocks = 1;
    size_t raw_chunk = (total + (size_t)nblocks - 1) / (size_t)nblocks;
    size_t chunk = ((raw_chunk + kBlockThreads - 1) / kBlockThreads) * kBlockThreads;
    *nblocks_out = nblocks;
    *chunk_out = chunk;
}

bool ensure_found_capacity(uint64_t needed, uint64_t *cap, int **d_fx, int **d_fy, int **d_fz,
                           float **d_fr, cudaStream_t stream) {
    if (needed <= *cap) {
        return true;
    }
    uint64_t new_cap = (*cap == 0) ? 4096ULL : (*cap * 2ULL);
    while (new_cap < needed) {
        new_cap *= 2ULL;
    }
    int *nx = (int *)p21_cuda_pool_alloc(new_cap * sizeof(int), stream);
    int *ny = (int *)p21_cuda_pool_alloc(new_cap * sizeof(int), stream);
    int *nz = (int *)p21_cuda_pool_alloc(new_cap * sizeof(int), stream);
    float *nr = (float *)p21_cuda_pool_alloc(new_cap * sizeof(float), stream);
    if (!nx || !ny || !nz || !nr) {
        p21_cuda_pool_free(nx, stream);
        p21_cuda_pool_free(ny, stream);
        p21_cuda_pool_free(nz, stream);
        p21_cuda_pool_free(nr, stream);
        return false;
    }
    if (*cap > 0) {
        bool ok = P21_CUDA_OK(cudaMemcpyAsync(nx, *d_fx, (*cap) * sizeof(int),
                                              cudaMemcpyDeviceToDevice, stream)) &&
                  P21_CUDA_OK(cudaMemcpyAsync(ny, *d_fy, (*cap) * sizeof(int),
                                              cudaMemcpyDeviceToDevice, stream)) &&
                  P21_CUDA_OK(cudaMemcpyAsync(nz, *d_fz, (*cap) * sizeof(int),
                                              cudaMemcpyDeviceToDevice, stream)) &&
                  P21_CUDA_OK(cudaMemcpyAsync(nr, *d_fr, (*cap) * sizeof(float),
                                              cudaMemcpyDeviceToDevice, stream));
        if (!ok) {
            p21_cuda_pool_free(nx, stream);
            p21_cuda_pool_free(ny, stream);
            p21_cuda_pool_free(nz, stream);
            p21_cuda_pool_free(nr, stream);
            return false;
        }
    }
    p21_cuda_pool_free(*d_fx, stream);
    p21_cuda_pool_free(*d_fy, stream);
    p21_cuda_pool_free(*d_fz, stream);
    p21_cuda_pool_free(*d_fr, stream);
    *d_fx = nx;
    *d_fy = ny;
    *d_fz = nz;
    *d_fr = nr;
    *cap = new_cap;
    return true;
}

}  // namespace

extern "C" P21CudaDexmCtx *p21_cuda_dexm_run(float redshift, InitialConditions *boxes, float M_MIN,
                                             float Delta_R,
                                             unsigned long long int *total_halo_num_out) {
    if (!p21_cuda_get_enabled() || !boxes || !boxes->hires_density || !total_halo_num_out) {
        return nullptr;
    }
    if (!p21_cuda_ensure_init()) {
        return nullptr;
    }

    const int grid_dim = simulation_options_global->DIM;
    const int z_dim = D_PARA;
    const size_t total = (size_t)grid_dim * (size_t)grid_dim * (size_t)z_dim;
    const size_t kspace_pixels = (size_t)grid_dim * (size_t)grid_dim * (size_t)(z_dim / 2 + 1);
    const size_t rbytes = total * sizeof(float);
    const size_t cbytes = kspace_pixels * sizeof(cufftComplex);
    cudaStream_t stream = p21_cuda_stream();

    const float *d_hires =
        (const float *)p21_cuda_registry_get_or_upload(boxes->hires_density, rbytes, stream);
    if (!d_hires) {
        return nullptr;
    }

    cufftComplex *d_dens_k_saved = (cufftComplex *)p21_cuda_pool_alloc(cbytes, stream);
    cufftComplex *d_dens_k = (cufftComplex *)p21_cuda_pool_alloc(cbytes, stream);
    float *d_delta_m = (float *)p21_cuda_pool_alloc(rbytes, stream);
    float *d_halo_catalog = (float *)p21_cuda_pool_alloc(rbytes, stream);
    uint8_t *d_in_halo = (uint8_t *)p21_cuda_pool_alloc(total, stream);
    const int dexm_optimize = matter_options_global->DEXM_OPTIMIZE ? 1 : 0;
    uint8_t *d_forbidden = dexm_optimize ? (uint8_t *)p21_cuda_pool_alloc(total, stream) : nullptr;

    int nblocks = 0;
    size_t chunk = 0;
    compute_chunk_grid(total, &nblocks, &chunk);
    uint64_t *d_blk_cnt = (uint64_t *)p21_cuda_pool_alloc(kMaxCompactBlocks * sizeof(uint64_t), stream);
    uint64_t *d_blk_off = (uint64_t *)p21_cuda_pool_alloc(kMaxCompactBlocks * sizeof(uint64_t), stream);
    uint64_t *d_u64_tmp = (uint64_t *)p21_cuda_pool_alloc(2 * sizeof(uint64_t), stream);

    bool ok = d_dens_k_saved && d_dens_k && d_delta_m && d_halo_catalog && d_in_halo &&
              (!dexm_optimize || d_forbidden) && d_blk_cnt && d_blk_off && d_u64_tmp;

    if (ok) {
        ok = P21_CUDA_OK(cudaMemsetAsync(d_in_halo, 0, total, stream)) &&
             P21_CUDA_OK(cudaMemsetAsync(d_halo_catalog, 0, rbytes, stream)) &&
             P21_CUDA_OK(cudaMemsetAsync(d_u64_tmp + 1, 0, sizeof(uint64_t), stream));
        if (ok && dexm_optimize) {
            ok = P21_CUDA_OK(cudaMemsetAsync(d_forbidden, 0, total, stream));
        }
    }

    if (ok) {
        ok = p21_cuda_fft_r2c(d_hires, d_dens_k_saved, grid_dim, grid_dim, z_dim, false, stream);
    }

    int *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;
    float *d_fr = nullptr;
    uint64_t found_cap = 0;
    uint64_t total_halo_num = 0;

    uint64_t *d_cand_idx = nullptr;
    uint64_t cand_cap = 0;

    const float growth_factor = (float)dicke(redshift);
    const float gf_over_totpix = growth_factor / (float)total;
    const float box_len = simulation_options_global->BOX_LEN;
    const double ncf = simulation_options_global->NON_CUBIC_FACTOR;
    const int halo_filter = matter_options_global->HALO_FILTER;
    const double delta_r_factor = simulation_options_global->DELTA_R_FACTOR;
    const double opt_minmass = simulation_options_global->DEXM_OPTIMIZE_MINMASS;
    const double r_overlap = simulation_options_global->DEXM_R_OVERLAP;

    float R = MtoR(M_MIN * 1.01f);
    while (R < physconst.l_factor * box_len) {
        R *= delta_r_factor;
    }

    while (ok && (R > 0.5f * Delta_R) && (RtoM(R) >= M_MIN)) {
        float M = RtoM(R);
        float delta_crit = growth_factor *
                           (float)sheth_delc_dexm(physconst.delta_c_sph / growth_factor, sigma_z0(M));
        if ((sigma_z0(M) * growth_factor * 7.0f) < delta_crit) {
            R /= delta_r_factor;
            continue;
        }

        ok = p21_cuda_copy_and_filter_box(d_dens_k_saved, d_dens_k, grid_dim, grid_dim, z_dim,
                                          box_len, ncf, halo_filter, R, 0.0f, 0.0f, 1.0f, stream) &&
             p21_cuda_fft_c2r(d_dens_k, d_delta_m, grid_dim, grid_dim, z_dim, false, stream);
        if (!ok) break;

        int opt_mode = (dexm_optimize && ((double)M > opt_minmass)) ? 1 : 0;
        if (opt_mode) {
            ok = P21_CUDA_OK(cudaMemsetAsync(d_forbidden, 0, total, stream));
            if (ok && total_halo_num > 0) {
                dexm_paint_sphere_all_halos_kernel<<<(unsigned int)total_halo_num, kBlockThreads, 0,
                                                    stream>>>(
                    total_halo_num, grid_dim, z_dim, box_len, (float)(r_overlap * (double)R), d_fx,
                    d_fy, d_fz, d_fr, d_forbidden);
                ok = P21_CUDA_OK(cudaGetLastError());
            }
            if (!ok) break;
        }

        dexm_count_cands_kernel<<<nblocks, kBlockThreads, 0, stream>>>(
            total, chunk, d_delta_m, gf_over_totpix, delta_crit, opt_mode, d_in_halo, d_forbidden,
            d_blk_cnt);
        ok = P21_CUDA_OK(cudaGetLastError());
        if (ok) {
            dexm_scan_blocks_kernel<<<1, kMaxCompactBlocks, 0, stream>>>(
                nblocks, d_blk_cnt, d_blk_off, d_u64_tmp);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
        uint64_t n_cands = 0;
        if (ok) {
            ok = P21_CUDA_OK(cudaMemcpyAsync(&n_cands, d_u64_tmp, sizeof(uint64_t),
                                             cudaMemcpyDeviceToHost, stream)) &&
                 P21_CUDA_OK(cudaStreamSynchronize(stream));
        }
        if (!ok) break;

        if (n_cands > 0) {
            if (n_cands > cand_cap) {
                p21_cuda_pool_free(d_cand_idx, stream);
                cand_cap = n_cands;
                d_cand_idx = (uint64_t *)p21_cuda_pool_alloc(cand_cap * sizeof(uint64_t), stream);
                if (!d_cand_idx) {
                    ok = false;
                    break;
                }
            }
            if (!ensure_found_capacity(total_halo_num + n_cands, &found_cap, &d_fx, &d_fy, &d_fz,
                                       &d_fr, stream)) {
                ok = false;
                break;
            }

            dexm_write_cands_kernel<<<nblocks, kBlockThreads, 0, stream>>>(
                total, chunk, d_delta_m, gf_over_totpix, delta_crit, opt_mode, d_in_halo,
                d_forbidden, d_blk_off, d_cand_idx);
            ok = P21_CUDA_OK(cudaGetLastError());
            if (ok) {
                float R_ov = (float)(r_overlap * (double)R);
                float R_forb = (float)((1.0 + r_overlap) * (double)R);
                dexm_accept_candidates_kernel<<<1, kBlockThreads, 0, stream>>>(
                    n_cands, d_cand_idx, grid_dim, z_dim, M, R, R_ov, R_forb, box_len, opt_mode,
                    d_in_halo, d_forbidden, d_halo_catalog, d_fx, d_fy, d_fz, d_fr, d_u64_tmp + 1);
                ok = P21_CUDA_OK(cudaGetLastError());
            }
            if (ok) {
                ok = P21_CUDA_OK(cudaMemcpyAsync(&total_halo_num, d_u64_tmp + 1, sizeof(uint64_t),
                                                 cudaMemcpyDeviceToHost, stream)) &&
                     P21_CUDA_OK(cudaStreamSynchronize(stream));
            }
        }

        R /= delta_r_factor;
    }

    p21_cuda_pool_free(d_dens_k_saved, stream);
    p21_cuda_pool_free(d_dens_k, stream);
    p21_cuda_pool_free(d_delta_m, stream);
    p21_cuda_pool_free(d_forbidden, stream);
    p21_cuda_pool_free(d_cand_idx, stream);
    p21_cuda_pool_free(d_fx, stream);
    p21_cuda_pool_free(d_fy, stream);
    p21_cuda_pool_free(d_fz, stream);
    p21_cuda_pool_free(d_fr, stream);

    float *d_found_masses = nullptr;
    float *d_found_coords = nullptr;
    if (ok && total_halo_num > 0) {
        d_found_masses = (float *)p21_cuda_pool_alloc(total_halo_num * sizeof(float), stream);
        d_found_coords = (float *)p21_cuda_pool_alloc(3ULL * total_halo_num * sizeof(float), stream);
        if (!d_found_masses || !d_found_coords) {
            ok = false;
        }
        if (ok) {
            dexm_count_halos_kernel<<<nblocks, kBlockThreads, 0, stream>>>(
                total, chunk, d_halo_catalog, d_blk_cnt);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
        if (ok) {
            dexm_scan_blocks_kernel<<<1, kMaxCompactBlocks, 0, stream>>>(
                nblocks, d_blk_cnt, d_blk_off, d_u64_tmp);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
        if (ok) {
            double cell_length = (double)box_len / (double)grid_dim;
            dexm_pack_halos_kernel<<<nblocks, kBlockThreads, 0, stream>>>(
                total, chunk, grid_dim, z_dim, cell_length, d_halo_catalog, d_blk_off,
                d_found_masses, d_found_coords);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
        if (ok) {
            ok = P21_CUDA_OK(cudaStreamSynchronize(stream));
        }
    }

    p21_cuda_pool_free(d_halo_catalog, stream);
    p21_cuda_pool_free(d_blk_cnt, stream);
    p21_cuda_pool_free(d_blk_off, stream);
    p21_cuda_pool_free(d_u64_tmp, stream);

    if (!ok) {
        p21_cuda_pool_free(d_in_halo, stream);
        p21_cuda_pool_free(d_found_masses, stream);
        p21_cuda_pool_free(d_found_coords, stream);
        return nullptr;
    }

    *total_halo_num_out = total_halo_num;
    auto *ctx = new P21CudaDexmCtx;
    ctx->grid_dim = grid_dim;
    ctx->z_dim = z_dim;
    ctx->total_pixels = total;
    ctx->total_halo_num = total_halo_num;
    ctx->d_in_halo = d_in_halo;
    ctx->d_found_masses = d_found_masses;
    ctx->d_found_coords = d_found_coords;
    ctx->stream = stream;
    return ctx;
}

extern "C" int p21_cuda_dexm_finish(P21CudaDexmCtx *ctx, HaloCatalog *halos_dexm,
                                    float *halo_overlap_box_out) {
    if (!ctx || !halos_dexm) {
        return -1;
    }
    cudaStream_t stream = ctx->stream;
    bool ok = true;

    if (ctx->total_halo_num > 0) {
        size_t mbytes = (size_t)ctx->total_halo_num * sizeof(float);
        size_t cbytes = 3ULL * mbytes;
        ok = P21_CUDA_OK(cudaMemcpyAsync(halos_dexm->halo_masses, ctx->d_found_masses, mbytes,
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(halos_dexm->halo_coords, ctx->d_found_coords, cbytes,
                                         cudaMemcpyDeviceToHost, stream));
        if (ok) {
            p21_cuda_stats_add_d2h(mbytes + cbytes);
            ok = P21_CUDA_OK(cudaStreamSynchronize(stream));
        }
    }

    if (ok && halo_overlap_box_out != nullptr) {
        const int grid_dim = ctx->grid_dim;
        const int z_dim = ctx->z_dim;
        const size_t total = ctx->total_pixels;
        const int lo_nx = simulation_options_global->HII_DIM;
        const int lo_nz = HII_D_PARA;
        const size_t lo_total = (size_t)lo_nx * (size_t)lo_nx * (size_t)lo_nz;
        const size_t lo_bytes = lo_total * sizeof(float);
        const size_t kspace_pixels =
            (size_t)grid_dim * (size_t)grid_dim * (size_t)(z_dim / 2 + 1);

        float *d_in_halo_f = (float *)p21_cuda_pool_alloc(total * sizeof(float), stream);
        cufftComplex *d_k =
            (cufftComplex *)p21_cuda_pool_alloc(kspace_pixels * sizeof(cufftComplex), stream);
        float *d_overlap = (float *)p21_cuda_pool_alloc(lo_bytes, stream);

        if (!d_in_halo_f || !d_k || !d_overlap) {
            ok = false;
        }
        if (ok) {
            int g_hi = (int)((total + kBlockThreads - 1) / kBlockThreads);
            dexm_u8_to_float_kernel<<<g_hi, kBlockThreads, 0, stream>>>(
                total, ctx->d_in_halo, d_in_halo_f);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 p21_cuda_fft_r2c(d_in_halo_f, d_k, grid_dim, grid_dim, z_dim, false, stream);
        }
        if (ok && grid_dim != lo_nx) {
            float R_smooth =
                (float)(physconst.l_factor * simulation_options_global->BOX_LEN / (double)lo_nx);
            ok = p21_cuda_filter_box(d_k, grid_dim, grid_dim, z_dim,
                                     simulation_options_global->BOX_LEN,
                                     simulation_options_global->NON_CUBIC_FACTOR, FILTER_TOPHAT,
                                     R_smooth, 0.0f, 0.0f, stream);
        }
        if (ok) {
            ok = p21_cuda_fft_c2r(d_k, d_in_halo_f, grid_dim, grid_dim, z_dim, false, stream);
        }
        if (ok) {
            int g_lo = (int)((lo_total + kBlockThreads - 1) / kBlockThreads);
            float f_pix = (float)grid_dim / (float)lo_nx;
            dexm_downsample_overlap_kernel<<<g_lo, kBlockThreads, 0, stream>>>(
                lo_nx, lo_nz, grid_dim, z_dim, f_pix, 1.0 / (double)total, d_in_halo_f, d_overlap);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
        if (ok) {
            ok = P21_CUDA_OK(cudaMemcpyAsync(halo_overlap_box_out, d_overlap, lo_bytes,
                                             cudaMemcpyDeviceToHost, stream)) &&
                 P21_CUDA_OK(cudaStreamSynchronize(stream));
            if (ok) {
                p21_cuda_stats_add_d2h(lo_bytes);
                if (!p21_cuda_registry_put_owned(halo_overlap_box_out, d_overlap, lo_bytes,
                                                 stream)) {
                    p21_cuda_pool_free(d_overlap, stream);
                }
                d_overlap = nullptr;
            }
        }
        p21_cuda_pool_free(d_in_halo_f, stream);
        p21_cuda_pool_free(d_k, stream);
        p21_cuda_pool_free(d_overlap, stream);
    }

    p21_cuda_pool_free(ctx->d_in_halo, stream);
    p21_cuda_pool_free(ctx->d_found_masses, stream);
    p21_cuda_pool_free(ctx->d_found_coords, stream);
    delete ctx;

    if (!ok) {
        return -1;
    }
    p21_cuda_stats_mut()->halo_dexm_runs++;
    return 0;
}

extern "C" void p21_cuda_dexm_abort(P21CudaDexmCtx *ctx) {
    if (!ctx) return;
    p21_cuda_pool_free(ctx->d_in_halo, ctx->stream);
    p21_cuda_pool_free(ctx->d_found_masses, ctx->stream);
    p21_cuda_pool_free(ctx->d_found_coords, ctx->stream);
    delete ctx;
}
