#ifndef P21_CUDA_TABLES_CUH
#define P21_CUDA_TABLES_CUH

#include <vector>

#include "cuda_common.cuh"

namespace {

__constant__ double kGL32PosX[16] = {
    4.83076656877383173e-02, 1.44471961582796488e-01, 2.39287362252137065e-01,
    3.31868602282127667e-01, 4.21351276130635333e-01, 5.06899908932229359e-01,
    5.87715757240762304e-01, 6.63044266930215231e-01, 7.32182118740289711e-01,
    7.94483795967942386e-01, 8.49367613732569970e-01, 8.96321155766052091e-01,
    9.34906075937739667e-01, 9.64762255587506390e-01, 9.85611511545268271e-01,
    9.97263861849481570e-01,
};

__constant__ double kGL32PosW[16] = {
    9.65400885147275761e-02, 9.56387200792745973e-02, 9.38443990808043860e-02,
    9.11738786957637104e-02, 8.76520930044036861e-02, 8.33119242269466237e-02,
    7.81938957870701168e-02, 7.23457941088481993e-02, 6.58222227763615997e-02,
    5.86840934785354609e-02, 5.09980592623759596e-02, 4.28358980222269259e-02,
    3.42738629130216327e-02, 2.53920653092627284e-02, 1.62743947309064857e-02,
    7.01861000947036177e-03,
};

}  // namespace

/* Device regular-grid lookup tables matching interpolation.c verbatim. */
struct P21DevTable1Df {
    const float *y_arr;
    double x_min;
    double x_width;
    int nx;
};

struct P21DevTable2Df {
    const float *z_arr;
    double x_min;
    double x_width;
    double y_min;
    double y_width;
    int nx;
    int ny;
};

struct P21DevTable1Dd {
    const double *y_arr;
    double x_min;
    double x_width;
    int nx;
};

struct P21DevTable2Dd {
    const double *z_arr;
    double x_min;
    double x_width;
    double y_min;
    double y_width;
    int nx;
    int ny;
};

__device__ static inline double p21_eval_rg1d_f(double x, const P21DevTable1Df &t) {
    int idx = (int)floor((x - t.x_min) / t.x_width);
    if (idx < 0) {
        idx = 0;
    } else if (idx > t.nx - 2) {
        idx = t.nx - 2;
    }
    /* Match EvaluateRGTable1D_f (interpolation.c:127) cast of idx to float. */
    double table_val = t.x_min + t.x_width * (float)idx;
    double ip = (x - table_val) / t.x_width;
    return (double)t.y_arr[idx] * (1.0 - ip) + (double)t.y_arr[idx + 1] * ip;
}

__device__ static inline double p21_eval_rg2d_f(double x, double y, const P21DevTable2Df &t) {
    int x_idx = (int)floor((x - t.x_min) / t.x_width);
    int y_idx = (int)floor((y - t.y_min) / t.y_width);
    if (x_idx < 0) {
        x_idx = 0;
    } else if (x_idx > t.nx - 2) {
        x_idx = t.nx - 2;
    }
    if (y_idx < 0) {
        y_idx = 0;
    } else if (y_idx > t.ny - 2) {
        y_idx = t.ny - 2;
    }
    double x_table = t.x_min + t.x_width * (double)x_idx;
    double y_table = t.y_min + t.y_width * (double)y_idx;
    double ipx = (x - x_table) / t.x_width;
    double ipy = (y - y_table) / t.y_width;

    const float *row0 = t.z_arr + (size_t)x_idx * t.ny;
    const float *row1 = row0 + t.ny;
    double left_edge = (double)row0[y_idx] * (1.0 - ipy) + (double)row0[y_idx + 1] * ipy;
    double right_edge = (double)row1[y_idx] * (1.0 - ipy) + (double)row1[y_idx + 1] * ipy;
    return left_edge * (1.0 - ipx) + right_edge * ipx;
}

__device__ static inline double p21_eval_rg1d_d(double x, const P21DevTable1Dd &t) {
    int idx = (int)floor((x - t.x_min) / t.x_width);
    if (idx < 0) {
        idx = 0;
    } else if (idx > t.nx - 2) {
        idx = t.nx - 2;
    }
    double table_val = t.x_min + t.x_width * (double)idx;
    double ip = (x - table_val) / t.x_width;
    return t.y_arr[idx] * (1.0 - ip) + t.y_arr[idx + 1] * ip;
}

__device__ static inline double p21_eval_rg2d_d(double x, double y, const P21DevTable2Dd &t) {
    int x_idx = (int)floor((x - t.x_min) / t.x_width);
    int y_idx = (int)floor((y - t.y_min) / t.y_width);
    if (x_idx < 0) {
        x_idx = 0;
    } else if (x_idx > t.nx - 2) {
        x_idx = t.nx - 2;
    }
    if (y_idx < 0) {
        y_idx = 0;
    } else if (y_idx > t.ny - 2) {
        y_idx = t.ny - 2;
    }
    double x_table = t.x_min + t.x_width * (double)x_idx;
    double y_table = t.y_min + t.y_width * (double)y_idx;
    double ipx = (x - x_table) / t.x_width;
    double ipy = (y - y_table) / t.y_width;

    const double *row0 = t.z_arr + (size_t)x_idx * t.ny;
    const double *row1 = row0 + t.ny;
    double left_edge = row0[y_idx] * (1.0 - ipy) + row0[y_idx + 1] * ipy;
    double right_edge = row1[y_idx] * (1.0 - ipy) + row1[y_idx + 1] * ipy;
    return left_edge * (1.0 - ipx) + right_edge * ipx;
}

__device__ static inline double p21_eval_sigma_inverse_dev(double sigma,
                                                          const P21DevTable1Df &sig_tbl) {
    int n = sig_tbl.nx;
    if (sigma >= (double)sig_tbl.y_arr[0]) {
        return sig_tbl.x_min;
    }
    if (sigma <= (double)sig_tbl.y_arr[n - 1]) {
        return sig_tbl.x_min + (double)(n - 1) * sig_tbl.x_width;
    }
    /* sig_tbl.y_arr is strictly decreasing from index 0 to n - 1 */
    int lo = 0;
    int hi = n - 2;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if ((double)sig_tbl.y_arr[mid] >= sigma) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    double s0 = (double)sig_tbl.y_arr[lo];
    double s1 = (double)sig_tbl.y_arr[lo + 1];
    double lnm0 = sig_tbl.x_min + (double)lo * sig_tbl.x_width;
    double lnm1 = sig_tbl.x_min + (double)(lo + 1) * sig_tbl.x_width;
    double denom = s1 - s0;
    if (fabs(denom) < 1e-30) {
        return lnm0;
    }
    double frac = (sigma - s0) / denom;
    return lnm0 + frac * (lnm1 - lnm0);
}

static inline bool p21_upload_rg1d_f(const RGTable1D_f *src, P21DevTable1Df *dst,
                                     cudaStream_t stream) {
    dst->y_arr = nullptr;
    dst->nx = 0;
    if (!src || !src->allocated || src->n_bin < 2 || !src->y_arr) {
        return false;
    }
    size_t bytes = (size_t)src->n_bin * sizeof(float);
    float *d_ptr = (float *)p21_cuda_pool_alloc(bytes, stream);
    if (!d_ptr) {
        return false;
    }
    if (!P21_CUDA_OK(cudaMemcpyAsync(d_ptr, src->y_arr, bytes, cudaMemcpyHostToDevice, stream))) {
        p21_cuda_pool_free(d_ptr, stream);
        return false;
    }
    dst->y_arr = d_ptr;
    dst->x_min = src->x_min;
    dst->x_width = src->x_width;
    dst->nx = src->n_bin;
    return true;
}

static inline bool p21_upload_rg2d_f(const RGTable2D_f *src, P21DevTable2Df *dst,
                                     cudaStream_t stream) {
    dst->z_arr = nullptr;
    dst->nx = 0;
    dst->ny = 0;
    if (!src || !src->allocated || src->nx_bin < 2 || src->ny_bin < 2 || !src->z_arr) {
        return false;
    }
    size_t total = (size_t)src->nx_bin * (size_t)src->ny_bin;
    std::vector<float> flat(total);
    for (int i = 0; i < src->nx_bin; ++i) {
        memcpy(&flat[(size_t)i * src->ny_bin], src->z_arr[i], (size_t)src->ny_bin * sizeof(float));
    }
    size_t bytes = total * sizeof(float);
    float *d_ptr = (float *)p21_cuda_pool_alloc(bytes, stream);
    if (!d_ptr) {
        return false;
    }
    if (!P21_CUDA_OK(cudaMemcpyAsync(d_ptr, flat.data(), bytes, cudaMemcpyHostToDevice, stream))) {
        p21_cuda_pool_free(d_ptr, stream);
        return false;
    }
    dst->z_arr = d_ptr;
    dst->x_min = src->x_min;
    dst->x_width = src->x_width;
    dst->y_min = src->y_min;
    dst->y_width = src->y_width;
    dst->nx = src->nx_bin;
    dst->ny = src->ny_bin;
    return true;
}

static inline bool p21_upload_rg1d_d(const RGTable1D *src, P21DevTable1Dd *dst,
                                     cudaStream_t stream) {
    dst->y_arr = nullptr;
    dst->nx = 0;
    if (!src || !src->allocated || src->n_bin < 2 || !src->y_arr) {
        return false;
    }
    size_t bytes = (size_t)src->n_bin * sizeof(double);
    double *d_ptr = (double *)p21_cuda_pool_alloc(bytes, stream);
    if (!d_ptr) {
        return false;
    }
    if (!P21_CUDA_OK(cudaMemcpyAsync(d_ptr, src->y_arr, bytes, cudaMemcpyHostToDevice, stream))) {
        p21_cuda_pool_free(d_ptr, stream);
        return false;
    }
    dst->y_arr = d_ptr;
    dst->x_min = src->x_min;
    dst->x_width = src->x_width;
    dst->nx = src->n_bin;
    return true;
}

static inline bool p21_upload_rg2d_d(const RGTable2D *src, P21DevTable2Dd *dst,
                                     cudaStream_t stream) {
    dst->z_arr = nullptr;
    dst->nx = 0;
    dst->ny = 0;
    if (!src || !src->allocated || src->nx_bin < 2 || src->ny_bin < 2 || !src->z_arr) {
        return false;
    }
    size_t total = (size_t)src->nx_bin * (size_t)src->ny_bin;
    std::vector<double> flat(total);
    for (int i = 0; i < src->nx_bin; ++i) {
        memcpy(&flat[(size_t)i * src->ny_bin], src->z_arr[i], (size_t)src->ny_bin * sizeof(double));
    }
    size_t bytes = total * sizeof(double);
    double *d_ptr = (double *)p21_cuda_pool_alloc(bytes, stream);
    if (!d_ptr) {
        return false;
    }
    if (!P21_CUDA_OK(cudaMemcpyAsync(d_ptr, flat.data(), bytes, cudaMemcpyHostToDevice, stream))) {
        p21_cuda_pool_free(d_ptr, stream);
        return false;
    }
    dst->z_arr = d_ptr;
    dst->x_min = src->x_min;
    dst->x_width = src->x_width;
    dst->y_min = src->y_min;
    dst->y_width = src->y_width;
    dst->nx = src->nx_bin;
    dst->ny = src->ny_bin;
    return true;
}

static inline void p21_free_dev_table(P21DevTable1Df *t, cudaStream_t stream) {
    if (t->y_arr) {
        p21_cuda_pool_free(const_cast<float *>(t->y_arr), stream);
        t->y_arr = nullptr;
    }
    t->nx = 0;
}

static inline void p21_free_dev_table(P21DevTable2Df *t, cudaStream_t stream) {
    if (t->z_arr) {
        p21_cuda_pool_free(const_cast<float *>(t->z_arr), stream);
        t->z_arr = nullptr;
    }
    t->nx = 0;
    t->ny = 0;
}

static inline void p21_free_dev_table(P21DevTable1Dd *t, cudaStream_t stream) {
    if (t->y_arr) {
        p21_cuda_pool_free(const_cast<double *>(t->y_arr), stream);
        t->y_arr = nullptr;
    }
    t->nx = 0;
}

static inline void p21_free_dev_table(P21DevTable2Dd *t, cudaStream_t stream) {
    if (t->z_arr) {
        p21_cuda_pool_free(const_cast<double *>(t->z_arr), stream);
        t->z_arr = nullptr;
    }
    t->nx = 0;
    t->ny = 0;
}

/* Host-side parameter bundle for device compute_mturns_inhomogeneous. */
struct P21MturnParams {
    double z;
    double mturn_acg_homogeneous;
    double a_lw;
    double beta_lw;
    double a_vcb;
    double beta_vcb;
    double sigma_vcb;
    double mturn_stellar_feedback;
    int use_reion_feedback;
    int use_mcgs;
};

__device__ static inline double p21_reionization_feedback(float z, float Gamma_halo_HII,
                                                          float z_IN) {
    if (z > z_IN) {
        return 1e-40;
    }
    constexpr double kHaloBias = 2.0;
    constexpr double kM0 = 3e9;
    constexpr double kA = 0.17;
    constexpr double kB = -2.1;
    constexpr double kC = 2.0;
    constexpr double kD = 2.5;
    return kM0 * pow(kHaloBias * (double)Gamma_halo_HII, kA) * pow((1.0 + (double)z) / 10.0, kB) *
           pow(1.0 - pow((1.0 + (double)z) / (1.0 + (double)z_IN), kC), kD);
}

__device__ static inline double p21_molecular_cooling_threshold_fb(float z, float J_21_LW,
                                                                   float vcb,
                                                                   const P21MturnParams &p) {
    double mcrit_noLW = 3.314e7 * pow(1.0 + (double)z, -1.5);
    double f_LW = 1.0 + p.a_lw * pow((double)J_21_LW, p.beta_lw);
    double f_vcb = pow(1.0 + p.a_vcb * (double)vcb / p.sigma_vcb, p.beta_vcb);
    return mcrit_noLW * f_LW * f_vcb;
}

__device__ static inline void p21_compute_mturns_inhomogeneous(const P21MturnParams &p,
                                                               float J_21_LW, float vcb,
                                                               float Gamma12, float z_reion,
                                                               float *M_turn_acg,
                                                               float *M_turn_mcg) {
    float M_turn_r = 0.0f;
    float acg = (float)p.mturn_acg_homogeneous;
    if (p.use_reion_feedback) {
        M_turn_r = (float)p21_reionization_feedback((float)p.z, Gamma12, z_reion);
        acg = fmaxf(acg, M_turn_r);
    }
    *M_turn_acg = acg;
    if (p.use_mcgs) {
        float mcg = (float)fmax(p21_molecular_cooling_threshold_fb((float)p.z, J_21_LW, vcb, p),
                                p.mturn_stellar_feedback);
        if (p.use_reion_feedback) {
            mcg = fmaxf(mcg, M_turn_r);
        }
        *M_turn_mcg = mcg;
    } else {
        *M_turn_mcg = 0.0f;
    }
}

#endif /* P21_CUDA_TABLES_CUH */
