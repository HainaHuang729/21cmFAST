#include <list>
#include <mutex>

#include "cuda_common.cuh"

namespace {

struct PlanKey {
    int nx;
    int ny;
    int nz;
    int type;
    bool padded;

    bool operator==(const PlanKey &other) const {
        return nx == other.nx && ny == other.ny && nz == other.nz && type == other.type &&
               padded == other.padded;
    }
};

struct PlanEntry {
    PlanKey key;
    cufftHandle handle;
};

constexpr size_t kMaxCachedPlans = 32;
std::mutex g_fft_mu;
std::list<PlanEntry> g_plan_lru;

bool create_plan_3d(const PlanKey &k, cufftHandle *out_plan) {
    cufftHandle plan = 0;
    cufftResult res;
    if (!k.padded) {
        res = cufftPlan3d(&plan, k.nx, k.ny, k.nz, (cufftType)k.type);
    } else {
        int n[3] = {k.nx, k.ny, k.nz};
        int r_last = 2 * (k.nz / 2 + 1);
        int c_last = k.nz / 2 + 1;
        int r_embed[3] = {k.nx, k.ny, r_last};
        int c_embed[3] = {k.nx, k.ny, c_last};
        int r_dist = k.nx * k.ny * r_last;
        int c_dist = k.nx * k.ny * c_last;

        int *inembed = (k.type == CUFFT_R2C) ? r_embed : c_embed;
        int *onembed = (k.type == CUFFT_R2C) ? c_embed : r_embed;
        int idist = (k.type == CUFFT_R2C) ? r_dist : c_dist;
        int odist = (k.type == CUFFT_R2C) ? c_dist : r_dist;

        res = cufftPlanMany(&plan, 3, n, inembed, 1, idist, onembed, 1, odist, (cufftType)k.type,
                            1);
    }
    if (!P21_CUFFT_OK(res)) {
        return false;
    }
    *out_plan = plan;
    return true;
}

bool acquire_plan(int nx, int ny, int nz, cufftType type, bool padded, cudaStream_t stream,
                  cufftHandle *out_plan) {
    PlanKey key{nx, ny, nz, (int)type, padded};

    std::lock_guard<std::mutex> lock(g_fft_mu);
    for (auto it = g_plan_lru.begin(); it != g_plan_lru.end(); ++it) {
        if (it->key == key) {
            g_plan_lru.splice(g_plan_lru.begin(), g_plan_lru, it);
            cufftSetStream(it->handle, stream);
            p21_cuda_stats_mut()->cufft_plan_hits++;
            *out_plan = it->handle;
            return true;
        }
    }

    cufftHandle plan = 0;
    if (!create_plan_3d(key, &plan)) {
        return false;
    }
    if (!P21_CUFFT_OK(cufftSetStream(plan, stream))) {
        cufftDestroy(plan);
        return false;
    }

    p21_cuda_stats_mut()->cufft_plan_misses++;
    while (g_plan_lru.size() >= kMaxCachedPlans) {
        cufftDestroy(g_plan_lru.back().handle);
        g_plan_lru.pop_back();
        p21_cuda_stats_mut()->cufft_plan_evictions++;
    }
    g_plan_lru.push_front(PlanEntry{key, plan});
    *out_plan = plan;
    return true;
}

__device__ static inline double p21_real_tophat_w(double kR) {
    if (kR < 1e-4) {
        return 1.0 - kR * kR / 10.0;
    }
    return 3.0 * pow(kR, -3) * (sin(kR) - cos(kR) * kR);
}

__device__ static inline double p21_sharp_k_w(double kR) {
    return (kR * 0.413566994 > 1.0) ? 0.0 : 1.0;
}

__device__ static inline double p21_gaussian_w(double kR_sq) {
    return exp(-0.643 * 0.643 * kR_sq / 2.0);
}

__device__ static inline double p21_exp_mfp_w(double k, double R, double mfp, double exp_term) {
    double kR = k * R;
    double ratio = mfp / R;
    if (kR < 1e-4) {
        double r2 = ratio * ratio;
        double r3 = r2 * ratio;
        double ts_0 = 6.0 * r3 - exp_term * (6.0 * r3 + 6.0 * r2 + 3.0 * ratio);
        return ts_0 + (exp_term * (2.0 * r2 + 0.5 * ratio) - 2.0 * ts_0 * r2) * kR * kR;
    }
    double r2 = ratio * ratio;
    double r3 = r2 * ratio;
    double f = (kR * kR * r2 + 2.0 * ratio + 1.0) * ratio * cos(kR);
    f += (kR * kR * (r2 - r3) + ratio + 1.0) * sin(kR) / kR;
    f *= exp_term;
    f -= 2.0 * r2;
    double d = (kR * ratio) * (kR * ratio) + 1.0;
    f *= -3.0 * ratio / (d * d);
    return f;
}

__device__ static inline double p21_spherical_shell_w(double k, double R_inner, double R_outer) {
    double kR_inner = k * R_inner;
    double kR_outer = k * R_outer;
    if (kR_outer < 1e-4) {
        double q = R_inner / R_outer;
        return 1.0 - kR_outer * kR_outer / 10.0 * (pow(q, 5) - 1.0) / (pow(q, 3) - 1.0);
    }
    return 3.0 / (pow(kR_outer, 3) - pow(kR_inner, 3)) *
           (sin(kR_outer) - cos(kR_outer) * kR_outer - sin(kR_inner) + cos(kR_inner) * kR_inner);
}

/* Reciprocal gamma 1/Gamma(x) matching gsl_sf_gammainv. */
__device__ static inline double p21_gammainv(double x) {
    if (x > 0.0) {
        return (x > 171.6) ? 0.0 : (1.0 / tgamma(x));
    }
    /* Euler reflection: 1/Gamma(x) = Gamma(1-x) * sin(pi*x) / pi.
     * In asymptotic_2F3 (alpha < 10), 1-x <= 5.5 so tgamma(1-x) is small and finite. */
    return tgamma(1.0 - x) * sinpi(x) / M_PI;
}

/* Matches filtering.c:asymptotic_2F3 verbatim (including upstream Stirling line 229). */
__device__ static inline double p21_asymptotic_2F3(double kR, double alpha, double beta) {
    double a1 = (2.0 + alpha) / 2.0;
    double a2 = (3.0 + alpha) / 2.0;
    double b1 = 5.0 / 2.0;
    double b2 = (2.0 + alpha + beta) / 2.0;
    double b3 = (3.0 + alpha + beta) / 2.0;

    double gamma_b1 = 3.0 / 4.0;
    double gamma_b2_over_a1, gamma_b3_over_a2;
    double gamma_a1 = 0.0, gamma_a2 = 0.0;

    if (a1 < 20.0) {
        gamma_a1 = tgamma(a1);
        gamma_a2 = tgamma(a2);
        gamma_b2_over_a1 = tgamma(b2) / gamma_a1;
        gamma_b3_over_a2 = tgamma(b3) / gamma_a2;
    } else {
        double y = beta / 2.0;
        gamma_b2_over_a1 =
            pow(a1, y) *
            exp((a1 + y - 0.5) * (y / a1 - y * y / (2.0 * a1 * a1) + y * y * y / (3.0 * a1 * a1 * a1)) -
                y);
        gamma_b3_over_a2 =
            pow(a2, y) *
            exp((a2 + y - 0.5) * (y / a2 - y * y / (2.0 * a2 * a1) + y * y * y / (3.0 * a2 * a2 * a2)) -
                y);
    }

    double decay_term1 = 0.0;
    double decay_term2 = 0.0;
    if (alpha < 10.0) {
        double gamma_b1_minus_a1_inv = p21_gammainv(b1 - a1);
        double gamma_b1_minus_a2_inv = p21_gammainv(b1 - a2);
        double gamma_b2_minus_a2_inv = p21_gammainv(b2 - a2);
        decay_term1 = M_PI * gamma_a1 * gamma_b1_minus_a1_inv / tgamma(b2 - a1) / tgamma(b3 - a1) /
                      pow(kR / 2.0, alpha + 2.0);
        decay_term2 = -2.0 * M_PI * gamma_a2 * gamma_b1_minus_a2_inv * gamma_b2_minus_a2_inv /
                      tgamma(b3 - a2) / pow(kR / 2.0, alpha + 3.0);
    }

    double phase = kR - M_PI * (2.0 + beta) / 2.0;
    double F_asymp = (cos(phase) - (1.0 + (alpha - 1.0) * beta) / kR * sin(phase)) /
                     pow(kR / 2.0, beta + 2.0);
    F_asymp += decay_term1 + decay_term2;
    F_asymp *= gamma_b1 * gamma_b2_over_a1 * gamma_b3_over_a2;
    return F_asymp;
}

__device__ static inline double p21_hyper_2F3(double kR, double alpha, double beta) {
    if (kR == 0.0) {
        return 1.0;
    }
    if (beta == 0.0) {
        return 3.0 / pow(kR, 3) * (sin(kR) - cos(kR) * kR);
    }
    if (kR < 30.0) {
        constexpr int kMaxTerms = 1000;
        double sum = 0.0;
        double term = 1.0;
        for (int n = 1; n < kMaxTerms; ++n) {
            sum += term;
            term *= -1.0 / (1.0 + beta / (alpha + 2.0 * n)) /
                    (1.0 + beta / (alpha + 1.0 + 2.0 * n)) * kR * kR / (2.0 * n) /
                    (2.0 * n + 3.0);
            if (fabs(term) < fabs(sum) * 1e-4) {
                break;
            }
        }
        return sum;
    } else {
        double F_ms = p21_asymptotic_2F3(kR, alpha, beta);
        double F_sl = 3.0 / pow(kR, 3) * (sin(kR) - cos(kR) * kR);
        return (fabs(F_ms) < fabs(F_sl)) ? F_ms : F_sl;
    }
}

__device__ static inline double p21_multiple_scattering_w(double k, double R_inner, double R_outer,
                                                          double alpha_in, double beta_in,
                                                          double alpha_out, double beta_out) {
    double kR_in = k * R_inner;
    double kR_out = k * R_outer;
    double r_out3 = R_outer * R_outer * R_outer;
    double r_in3 = R_inner * R_inner * R_inner;
    double W = r_out3 * p21_hyper_2F3(kR_out, alpha_out, beta_out) -
               r_in3 * p21_hyper_2F3(kR_in, alpha_in, beta_in);
    return W / (r_out3 - r_in3);
}

__global__ void copy_filter_box_kernel(const cufftComplex *src, cufftComplex *dst, int nx, int ny,
                                       int nz, double dk_x, double dk_y, double dk_z,
                                       int filter_type, float R, float R_param, double exp_term,
                                       double alpha_in, double beta_in, double alpha_out,
                                       double beta_out, float pre_scale) {
    int hz = nz / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nx * ny * hz;
    if (idx >= total) {
        return;
    }

    int z = (int)(idx % hz);
    size_t q = idx / hz;
    int y = (int)(q % ny);
    int x = (int)(q / ny);

    float kx = (x > nx / 2) ? (float)((x - nx) * dk_x) : (float)(x * dk_x);
    float ky = (y > ny / 2) ? (float)((y - ny) * dk_y) : (float)(y * dk_y);
    float kz = (float)(z * dk_z);
    float k_mag_sq = kx * kx + ky * ky + kz * kz;

    double w = 1.0;
    if (filter_type == FILTER_TOPHAT) {
        float kR = sqrtf(k_mag_sq) * R;
        w = p21_real_tophat_w((double)kR);
    } else if (filter_type == FILTER_SHARP_K) {
        float kR = sqrtf(k_mag_sq) * R;
        w = p21_sharp_k_w((double)kR);
    } else if (filter_type == FILTER_GAUSSIAN) {
        float kR = k_mag_sq * R * R;
        w = p21_gaussian_w((double)kR);
    } else if (filter_type == FILTER_EXP_MFP) {
        w = p21_exp_mfp_w(sqrt((double)k_mag_sq), (double)R, (double)R_param, exp_term);
    } else if (filter_type == FILTER_SPHERICAL_SHELL_STRAIGHT_LINE) {
        w = p21_spherical_shell_w(sqrt((double)k_mag_sq), (double)R, (double)R_param);
    } else if (filter_type == FILTER_SPHERICAL_SHELL_MULTIPLE_SCATTERING) {
        w = p21_multiple_scattering_w(sqrt((double)k_mag_sq), (double)R, (double)R_param, alpha_in,
                                      beta_in, alpha_out, beta_out);
    }

    cufftComplex v = src[idx];
    if (pre_scale != 1.0f) {
        v.x *= pre_scale;
        v.y *= pre_scale;
    }
    v.x = (float)((double)v.x * w);
    v.y = (float)((double)v.y * w);
    dst[idx] = v;
}

}  // namespace

bool p21_cuda_fft_r2c(const float *d_real, cufftComplex *d_complex, int nx, int ny, int nz,
                      bool padded, cudaStream_t stream) {
    cufftHandle plan = 0;
    if (!acquire_plan(nx, ny, nz, CUFFT_R2C, padded, stream, &plan)) {
        return false;
    }
    if (!P21_CUFFT_OK(cufftExecR2C(plan, (cufftReal *)const_cast<float *>(d_real), d_complex))) {
        return false;
    }
    p21_cuda_stats_mut()->cufft_exec_r2c++;
    return true;
}

bool p21_cuda_fft_c2r(cufftComplex *d_complex_scratch, float *d_real, int nx, int ny, int nz,
                      bool padded, cudaStream_t stream) {
    cufftHandle plan = 0;
    if (!acquire_plan(nx, ny, nz, CUFFT_C2R, padded, stream, &plan)) {
        return false;
    }
    if (!P21_CUFFT_OK(cufftExecC2R(plan, d_complex_scratch, (cufftReal *)d_real))) {
        return false;
    }
    p21_cuda_stats_mut()->cufft_exec_c2r++;
    return true;
}

void p21_cuda_fft_clear_cache(void) {
    std::lock_guard<std::mutex> lock(g_fft_mu);
    for (auto &entry : g_plan_lru) {
        cufftDestroy(entry.handle);
    }
    g_plan_lru.clear();
}

bool p21_cuda_copy_and_filter_box(const cufftComplex *d_src, cufftComplex *d_dst, int nx, int ny,
                                  int nz, double box_len, double non_cubic_factor, int filter_type,
                                  float R, float R_param, float R_star, float pre_scale,
                                  cudaStream_t stream) {
    if (!d_src || !d_dst) {
        return false;
    }
    if (filter_type != FILTER_TOPHAT && filter_type != FILTER_SHARP_K &&
        filter_type != FILTER_GAUSSIAN && filter_type != FILTER_EXP_MFP &&
        filter_type != FILTER_SPHERICAL_SHELL_STRAIGHT_LINE &&
        filter_type != FILTER_SPHERICAL_SHELL_MULTIPLE_SCATTERING) {
        return false;
    }

    double dk_x = 2.0 * M_PI / box_len;
    double dk_y = 2.0 * M_PI / box_len;
    double dk_z = 2.0 * M_PI / (box_len * non_cubic_factor);

    double exp_term = 0.0;
    double ab[4] = {0.0, 0.0, 0.0, 0.0};
    if (filter_type == FILTER_EXP_MFP) {
        exp_term = exp(-(double)R / (double)R_param);
    } else if (filter_type == FILTER_SPHERICAL_SHELL_MULTIPLE_SCATTERING) {
        p21_get_ms_filter_params((double)R, (double)R_param, (double)R_star, ab);
    }

    size_t total = (size_t)nx * ny * (nz / 2 + 1);
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    copy_filter_box_kernel<<<blocks, threads, 0, stream>>>(
        d_src, d_dst, nx, ny, nz, dk_x, dk_y, dk_z, filter_type, R, R_param, exp_term, ab[0], ab[1],
        ab[2], ab[3], pre_scale);
    return P21_CUDA_OK(cudaGetLastError());
}

bool p21_cuda_filter_box(cufftComplex *d_box, int nx, int ny, int nz, double box_len,
                         double non_cubic_factor, int filter_type, float R, float R_param,
                         float R_star, cudaStream_t stream) {
    return p21_cuda_copy_and_filter_box(d_box, d_box, nx, ny, nz, box_len, non_cubic_factor,
                                        filter_type, R, R_param, R_star, 1.0f, stream);
}
