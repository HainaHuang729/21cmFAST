#include <math.h>
#include <stdint.h>
#include <string.h>

#include "cuda_common.cuh"
#include "cuda_halo_props.cuh"
#include "cuda_tables.cuh"

extern "C" {
#include "../hmf.h"
}

namespace {

constexpr int kBlockThreads = 256;
constexpr int kMaxScanBlocks = 1024;
constexpr int kLocalCap = 512;

/* ==================== Part A: Conditional HMF & dNdM Table Builders ==================== */

__device__ static inline double p21_eval_sigma_dev(double lnM, const P21DevTable1Df &sig) {
    return p21_eval_rg1d_f(lnM, sig);
}

__device__ static inline double p21_eval_dsigmasqdm_dev(double lnM, const P21DevTable1Df &dsig) {
    return -pow(10.0, p21_eval_rg1d_f(lnM, dsig));
}

__device__ static inline double p21_sheth_delc_fixed_dev(double del, double sig) {
    constexpr double kJa = 0.73;
    constexpr double kJb = 0.34;
    constexpr double kJc = 0.81;
    return sqrt(kJa) * del * (1.0 + kJb * pow((sig * sig) / (kJa * del * del), kJc));
}

__device__ static inline double p21_get_delta_crit_dev(int hmf, double sigma, double growthf,
                                                       double delta_c_sph, double delta_c_delos) {
    if (hmf == HMF_DELOS) {
        return delta_c_delos;
    }
    if (hmf == HMF_ST) {
        return p21_sheth_delc_fixed_dev(delta_c_sph / growthf, sigma) * growthf;
    }
    return delta_c_sph;
}

__device__ static inline double p21_st_taylor_factor_dev(double sig, double sig_cond,
                                                         double growthf, double delta_c_sph,
                                                         double *zeroth_order) {
    constexpr double kJa = 0.73;
    constexpr double kAlpha = 0.81;
    constexpr double kBeta = 0.34;

    double del = delta_c_sph / growthf;
    double sigsq = sig * sig;
    double sigsq_inv = 1.0 / sigsq;
    double sigcsq = sig_cond * sig_cond;
    double sigdiff = (sig == sig_cond) ? 1e-6 : (sigsq - sigcsq);

    double t_array[6];
    t_array[0] = 1.0;
    for (int i = 1; i < 6; ++i) {
        t_array[i] = t_array[i - 1] * (-sigdiff) / (double)i * (kAlpha - (double)i + 1.0) *
                     sigsq_inv;
    }
    double result = 0.0;
    for (int i = 5; i >= 0; --i) {
        result += t_array[i];
    }

    double prefactor_1 = sqrt(kJa) * del;
    double prefactor_2 = kBeta * pow(sigsq_inv * (kJa * del * del), -kAlpha);
    result = prefactor_1 * (1.0 + prefactor_2 * result);
    if (zeroth_order != nullptr) {
        *zeroth_order = prefactor_1 * (1.0 + prefactor_2);
    }
    return result;
}

__device__ static inline double p21_conditional_hmf_dev(
    int hmf, double growthf, double lnM, double delta_cond, double sigma_cond, double delta_c_sph,
    double delta_c_delos, const P21DevTable1Df &sig_tbl, const P21DevTable1Df &dsig_tbl) {
    double sigma1 = p21_eval_sigma_dev(lnM, sig_tbl);
    if (sigma1 < sigma_cond) {
        return 0.0;
    }
    double dsigmasqdm = p21_eval_dsigmasqdm_dev(lnM, dsig_tbl);
    double sigdiff_inv =
        (sigma1 == sigma_cond) ? 1e6 : 1.0 / (sigma1 * sigma1 - sigma_cond * sigma_cond);

    if (hmf == HMF_ST) {
        double delta_0 = delta_cond / growthf;
        double barrier = 0.0;
        double factor =
            p21_st_taylor_factor_dev(sigma1, sigma_cond, growthf, delta_c_sph, &barrier) - delta_0;
        return -dsigmasqdm * factor * pow(sigdiff_inv, 1.5) *
               exp(-(barrier - delta_0) * (barrier - delta_0) * 0.5 * sigdiff_inv) /
               sqrt(2.0 * M_PI);
    }
    if (hmf == HMF_DELOS) {
        constexpr double kCoeffNu = 0.519;
        constexpr double kIndexNu = 0.582;
        constexpr double kExpFactor = -0.469;
        double dsigmadm = dsigmasqdm * 0.5;
        double nu = (delta_c_delos - delta_cond) * sqrt(sigdiff_inv) / growthf;
        if (nu <= 0.0) {
            return 0.0;
        }
        double dfdnu = kCoeffNu * pow(nu, kIndexNu) * exp(kExpFactor * nu * nu);
        return dfdnu * fabs(dsigmadm) * sigdiff_inv;
    }
    /* HMF_PS and EPS fallback for WATSON / WATSON_Z / REED07 / YUNG24 (hmf.c:529-530). */
    double del = (delta_c_sph - delta_cond) / growthf;
    return -del * dsigmasqdm * pow(sigdiff_inv, 1.5) * exp(-del * del * 0.5 * sigdiff_inv) /
           sqrt(2.0 * M_PI);
}

__device__ static inline double panel_edge(double upper, double span, int p) {
    double t = 1.0 - (double)p / (double)kBlockThreads;
    double t2 = t * t;
    return upper - span * (t2 * t2);
}

__device__ static inline void integrate_panel_gl32(
    double a, double b, int hmf, double growth_out, double delta, double sigma_cond,
    double delta_c_sph, double delta_c_delos, const P21DevTable1Df &sig_tbl,
    const P21DevTable1Df &dsig_tbl, double *n_out, double *f_out) {
    if (b <= a) {
        *n_out = 0.0;
        *f_out = 0.0;
        return;
    }
    double mid = 0.5 * (a + b);
    double half = 0.5 * (b - a);
    double sn = 0.0;
    double sf = 0.0;
    for (int q = 0; q < 16; ++q) {
        double dx = half * kGL32PosX[q];
        double w = kGL32PosW[q];
        double xm = mid - dx;
        double xp = mid + dx;
        double cm = p21_conditional_hmf_dev(hmf, growth_out, xm, delta, sigma_cond, delta_c_sph,
                                            delta_c_delos, sig_tbl, dsig_tbl);
        double cp = p21_conditional_hmf_dev(hmf, growth_out, xp, delta, sigma_cond, delta_c_sph,
                                            delta_c_delos, sig_tbl, dsig_tbl);
        sn += w * (cm + cp);
        if (f_out != nullptr) {
            sf += w * (exp(xm) * cm + exp(xp) * cp);
        }
    }
    *n_out = half * sn;
    if (f_out != nullptr) {
        *f_out = half * sf;
    }
}

__global__ void dndm_tables_kernel(int nx, double xmin, double xmax, double ymin, double ymax,
                                   double growth_out, double param, int from_catalog, int hmf,
                                   double delta_c_sph, double delta_c_delos, double rhocrit_omm,
                                   double max_deltac_frac, double fract_float_err,
                                   P21DevTable1Df sig_tbl, P21DevTable1Df dsig_tbl,
                                   double *nhalo_out, double *fcoll_out) {
    __shared__ double sh_n[kBlockThreads];
    __shared__ double sh_f[kBlockThreads];

    int i = blockIdx.x;
    int tid = threadIdx.x;
    if (i >= nx) {
        return;
    }

    double x = xmin + (xmax - xmin) * (double)i / (double)(nx - 1);
    double lnM_cond, sigma_cond, delta;
    if (from_catalog) {
        lnM_cond = x;
        sigma_cond = p21_eval_sigma_dev(lnM_cond, sig_tbl);
        delta = p21_get_delta_crit_dev(hmf, sigma_cond, param, delta_c_sph, delta_c_delos) /
                param * growth_out;
    } else {
        lnM_cond = param;
        sigma_cond = p21_eval_sigma_dev(lnM_cond, sig_tbl);
        delta = x;
    }

    if (ymin >= lnM_cond || ymin >= ymax) {
        if (tid == 0) {
            nhalo_out[i] = 0.0;
            fcoll_out[i] = 0.0;
        }
        return;
    }

    double dcrit = p21_get_delta_crit_dev(hmf, sigma_cond, growth_out, delta_c_sph, delta_c_delos);
    if (delta > max_deltac_frac * dcrit) {
        if (tid == 0) {
            if (lnM_cond * (1.0 - fract_float_err) <= ymax) {
                nhalo_out[i] = rhocrit_omm / exp(lnM_cond);
                fcoll_out[i] = 1.0;
            } else {
                nhalo_out[i] = 0.0;
                fcoll_out[i] = 0.0;
            }
        }
        return;
    }

    double upper = fmin(ymax, lnM_cond);
    double span = upper - ymin;
    double a = panel_edge(upper, span, tid);
    double b = panel_edge(upper, span, tid + 1);
    double pn = 0.0, pf = 0.0;
    integrate_panel_gl32(a, b, hmf, growth_out, delta, sigma_cond, delta_c_sph, delta_c_delos,
                         sig_tbl, dsig_tbl, &pn, &pf);

    sh_n[tid] = pn;
    sh_f[tid] = pf;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_n[tid] += sh_n[tid + s];
            sh_f[tid] += sh_f[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        nhalo_out[i] = sh_n[0] * rhocrit_omm;
        fcoll_out[i] = sh_f[0];
    }
}

__global__ void dndm_inverse_kernel(int nx, int np, double xmin, double xmax, double lnM_min,
                                    double growth_out, double param, int from_catalog, int hmf,
                                    double delta_c_sph, double delta_c_delos,
                                    double max_deltac_frac, double min_lp,
                                    P21DevTable1Df sig_tbl, P21DevTable1Df dsig_tbl,
                                    double *z_flat_out) {
    __shared__ double sh_panel[kBlockThreads];
    __shared__ double sh_sfx[kBlockThreads + 1];

    int i = blockIdx.x;
    int tid = threadIdx.x;
    if (i >= nx) {
        return;
    }

    double *row = z_flat_out + (size_t)i * (size_t)np;
    double x = (i == nx - 1) ? xmax : (xmin + (xmax - xmin) * (double)i / (double)(nx - 1));

    double lnM_cond, sigma_cond, delta;
    if (from_catalog) {
        lnM_cond = x;
        sigma_cond = p21_eval_sigma_dev(lnM_cond, sig_tbl);
        delta = p21_get_delta_crit_dev(hmf, sigma_cond, param, delta_c_sph, delta_c_delos) /
                param * growth_out;
    } else {
        lnM_cond = param;
        sigma_cond = p21_eval_sigma_dev(lnM_cond, sig_tbl);
        delta = x;
        double dcrit =
            p21_get_delta_crit_dev(hmf, sigma_cond, growth_out, delta_c_sph, delta_c_delos);
        if (delta > max_deltac_frac * dcrit) {
            if (tid == 0) {
                row[0] = 0.0;
            }
            for (int k = 1 + tid; k < np; k += blockDim.x) {
                row[k] = 1.0;
            }
            return;
        }
    }

    double M_cond = exp(lnM_cond);
    double ratio_min = exp(lnM_min) / M_cond;

    if (lnM_min >= lnM_cond) {
        for (int k = tid; k < np; k += blockDim.x) {
            row[k] = (k >= 1 && k < np - 1) ? ratio_min : 0.0;
        }
        return;
    }

    double span = lnM_cond - lnM_min;
    double a = panel_edge(lnM_cond, span, tid);
    double b = panel_edge(lnM_cond, span, tid + 1);
    double pn = 0.0;
    integrate_panel_gl32(a, b, hmf, growth_out, delta, sigma_cond, delta_c_sph, delta_c_delos,
                         sig_tbl, dsig_tbl, &pn, nullptr);
    sh_panel[tid] = pn;
    __syncthreads();

    if (tid == 0) {
        sh_sfx[kBlockThreads] = 0.0;
        double acc = 0.0;
        for (int p = kBlockThreads - 1; p >= 0; --p) {
            acc += sh_panel[p];
            sh_sfx[p] = acc;
        }
    }
    __syncthreads();

    double norm = sh_sfx[0];
    if (norm <= 0.0) {
        for (int k = tid; k < np; k += blockDim.x) {
            row[k] = (k >= 1 && k < np - 1) ? ratio_min : 0.0;
        }
        return;
    }

    if (tid == 0) {
        row[np - 1] = ratio_min;
    }

    for (int k = tid; k < np - 1; k += blockDim.x) {
        double pa_k = min_lp * (1.0 - (double)k / (double)(np - 1));
        double target = norm * exp(pa_k);
        if (target <= 0.0) {
            row[k] = 1.0;
            continue;
        }
        if (target >= norm) {
            row[k] = ratio_min;
            continue;
        }

        int lo = 0;
        int hi = kBlockThreads - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (sh_sfx[mid + 1] >= target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        double p_lo = panel_edge(lnM_cond, span, lo);
        double p_hi = panel_edge(lnM_cond, span, lo + 1);
        double base_sfx = sh_sfx[lo + 1];
        double r_lo = p_lo;
        double r_hi = p_hi;
        for (int it = 0; it < 24; ++it) {
            double mid = 0.5 * (r_lo + r_hi);
            double tail = 0.0;
            integrate_panel_gl32(mid, p_hi, hmf, growth_out, delta, sigma_cond, delta_c_sph,
                                 delta_c_delos, sig_tbl, dsig_tbl, &tail, nullptr);
            if (base_sfx + tail >= target) {
                r_lo = mid;
            } else {
                r_hi = mid;
            }
        }
        row[k] = exp(0.5 * (r_lo + r_hi)) / M_cond;
    }
}

/* ==================== Part B: Deterministic Philox4x32-10 PRNG ==================== */

struct P21PhiloxRng {
    uint32_t k0, k1;
    uint32_t c0, c1, c2, c3;
    uint32_t buf[4];
    int idx;

    __device__ void refill() {
        uint32_t v0 = c0, v1 = c1, v2 = c2, v3 = c3;
        uint32_t rk0 = k0, rk1 = k1;
        constexpr uint32_t kM0 = 0xD2511F53U;
        constexpr uint32_t kM1 = 0xCD9E8D57U;
        constexpr uint32_t kW0 = 0x9E3779B9U;
        constexpr uint32_t kW1 = 0xBB67AE85U;

        for (int r = 0; r < 10; ++r) {
            uint32_t hi0 = __umulhi(kM0, v0);
            uint32_t lo0 = kM0 * v0;
            uint32_t hi1 = __umulhi(kM1, v2);
            uint32_t lo1 = kM1 * v2;
            uint32_t nv0 = hi1 ^ v1 ^ rk0;
            uint32_t nv1 = lo1;
            uint32_t nv2 = hi0 ^ v3 ^ rk1;
            uint32_t nv3 = lo0;
            v0 = nv0;
            v1 = nv1;
            v2 = nv2;
            v3 = nv3;
            rk0 += kW0;
            rk1 += kW1;
        }
        buf[0] = v0;
        buf[1] = v1;
        buf[2] = v2;
        buf[3] = v3;
        idx = 0;
        if (++c0 == 0U) {
            c1 = (c1 & 0xFF000000U) | ((c1 + 1U) & 0x00FFFFFFU);
        }
    }

    __device__ void init(uint64_t seed, uint64_t stream_id, uint32_t substream = 0) {
        k0 = (uint32_t)(seed & 0xFFFFFFFFULL);
        k1 = (uint32_t)(seed >> 32);
        c0 = 0U;
        c1 = (substream & 0xFFU) << 24;
        c2 = (uint32_t)(stream_id & 0xFFFFFFFFULL);
        c3 = (uint32_t)(stream_id >> 32);
        idx = 4;
    }

    __device__ uint32_t next_u32() {
        if (idx >= 4) {
            refill();
        }
        return buf[idx++];
    }

    __device__ double uniform() {
        uint64_t hi = (uint64_t)next_u32();
        uint64_t lo = (uint64_t)next_u32();
        uint64_t u53 = ((hi << 21) ^ (lo >> 11)) & ((1ULL << 53) - 1ULL);
        return (double)u53 * (1.0 / 9007199254740992.0);
    }

    __device__ double ugaussian() {
        double u1 = uniform();
        if (u1 < 1e-300) {
            u1 = 1e-300;
        }
        double u2 = uniform();
        return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    }

    __device__ double ugaussian_tail(double a) {
        if (a <= 0.0) {
            double z = ugaussian();
            return fabs(z);
        }
        if (a < 1.0) {
            while (true) {
                double z = ugaussian();
                if (z > a) {
                    return z;
                }
            }
        }
        double s = a * a;
        while (true) {
            double u = uniform();
            double v = uniform();
            double d = 1.0 - u;
            if (d < 1e-300) {
                d = 1e-300;
            }
            double x = sqrt(s - 2.0 * log(d));
            if (x * v <= a) {
                return x;
            }
        }
    }

    __device__ int poisson(double lam) {
        if (lam <= 0.0) {
            return 0;
        }
        if (lam < 30.0) {
            double L = exp(-lam);
            int k = 0;
            double p = 1.0;
            do {
                ++k;
                double u = 1.0 - uniform();
                p *= u;
            } while (p > L);
            return k - 1;
        }
        /* Hormann PTRS rejection algorithm for large lambda >= 30. */
        double slam = sqrt(lam);
        double loglam = log(lam);
        double b = 0.931 + 2.53 * slam;
        double a = -0.059 + 0.02483 * b;
        double inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
        double v_r = 0.9277 - 3.6224 / (b - 2.0);
        while (true) {
            double u = uniform() - 0.5;
            double v = uniform();
            double us = 0.5 - fabs(u);
            double kf = floor((2.0 * a / us + b) * u + lam + 0.43);
            if (kf < 0.0) {
                continue;
            }
            int k = (int)kf;
            if (us >= 0.07 && v <= v_r) {
                return k;
            }
            if (us < 0.013 && v > us) {
                continue;
            }
            if ((log(v) + log(inv_alpha) - log(a / (us * us) + b)) <=
                (-lam + (double)k * loglam - lgamma((double)k + 1.0))) {
                return k;
            }
        }
    }
};

/* ==================== Part C: GPU Halo Sampler Kernels ==================== */

struct SamplerKernelParams {
    int from_catalog;
    int hmf;
    int sample_method;
    double growth_in;
    double growth_out;
    double M_min;
    double lnM_min;
    double sigma_min;
    double M_max_tables;
    double M_cond;
    double lnM_cond;
    double sigma_cond;
    double corr_star;
    double corr_sfr;
    double corr_xray;
    double sampler_min_mass;
    double min_logprob;
    double halomass_correction;
    double parkinson_g0;
    double parkinson_y1;
    double parkinson_y2;
    double delta_c_sph;
    double delta_c_delos;
    double rhocrit_omm;
    double max_deltac_frac;
    double delta_min;
    double fract_float_err;
    double boxlen;
    double boxlen_z;
    int lo_dim[3];

    P21DevTable1Df sig_tbl;
    P21DevTable1Df dsig_tbl;
    P21DevTable1Dd nhalo_1d;
    P21DevTable1Dd fcoll_1d;
    P21DevTable2Dd nhalo_inv_2d;
    P21DevTable1Dd j_split_1d;
};

struct CondState {
    double M_cond;
    double lnM_cond;
    double sigma_cond;
    double delta;
    double delta_crit;
    double cond_val;
    double expected_N;
    double expected_M;
};

__device__ static inline void setup_cond_state(const SamplerKernelParams &p, double cond_val,
                                               CondState *cs) {
    if (p.from_catalog) {
        cs->M_cond = cond_val;
        cs->lnM_cond = log(cond_val);
        cs->sigma_cond = p21_eval_sigma_dev(cs->lnM_cond, p.sig_tbl);
        cs->cond_val = cs->lnM_cond;
        cs->delta =
            p21_get_delta_crit_dev(p.hmf, cs->sigma_cond, p.growth_in, p.delta_c_sph,
                                   p.delta_c_delos) /
            p.growth_in * p.growth_out;
    } else {
        cs->M_cond = p.M_cond;
        cs->lnM_cond = p.lnM_cond;
        cs->sigma_cond = p.sigma_cond;
        cs->delta = cond_val;
        cs->cond_val = cond_val;
    }
    cs->delta_crit =
        p21_get_delta_crit_dev(p.hmf, cs->sigma_cond, p.growth_out, p.delta_c_sph, p.delta_c_delos);

    if (cs->delta > p.max_deltac_frac * cs->delta_crit) {
        cs->expected_M = cs->M_cond;
        cs->expected_N = 1.0;
    } else if (cs->delta <= p.delta_min || cs->M_cond < p.M_min) {
        cs->expected_M = 0.0;
        cs->expected_N = 0.0;
    } else {
        double n_exp = p21_eval_rg1d_d(cs->cond_val, p.nhalo_1d);
        double f_exp = p21_eval_rg1d_d(cs->cond_val, p.fcoll_1d);
        cs->expected_N = n_exp * cs->M_cond / p.rhocrit_omm;
        cs->expected_M = f_exp * cs->M_cond;
    }
}

__device__ static inline double sample_dndm_inv_dev(const SamplerKernelParams &p,
                                                    const CondState &cs, P21PhiloxRng &rng) {
    double u = rng.uniform();
    double lnp = (u == 0.0) ? p.min_logprob : log(u);
    if (lnp < p.min_logprob) {
        lnp = p.min_logprob;
    }
    double r = p21_eval_rg2d_d(cs.cond_val, lnp, p.nhalo_inv_2d);
    r = fmin(1.0, fmax(0.0, r));
    return r * cs.M_cond;
}

__device__ static inline double eval_j_split_dev(double u_res, double gamma1,
                                                 const P21DevTable1Dd &js,
                                                 double fract_float_err) {
    if (fabs(gamma1) < fract_float_err) {
        return u_res;
    }
    if (u_res < js.x_min) {
        return pow(u_res, 1.0 - gamma1) / (1.0 - gamma1);
    }
    double u_max = js.x_min + (double)(js.nx - 1) * js.x_width;
    if (u_res > u_max) {
        return js.y_arr[js.nx - 1] + u_res - 0.5 * gamma1 * (1.0 / u_res - 1.0 / u_max);
    }
    return p21_eval_rg1d_d(u_res, js);
}

__device__ static inline bool stoc_sample_cond_dev(
    const SamplerKernelParams &p, const CondState &cs, P21PhiloxRng &rng, bool store_masses,
    float *M_out, int *kept_count_out, int *overflow_flag) {
    *kept_count_out = 0;

    if (cs.delta <= p.delta_min) {
        return true;
    }
    if (cs.delta >= p.max_deltac_frac * cs.delta_crit) {
        if ((float)cs.expected_M >= (float)p.sampler_min_mass) {
            if (store_masses) {
                M_out[0] = (float)cs.expected_M;
            }
            *kept_count_out = 1;
        }
        return true;
    }

    if (p.sample_method == SAMPLE_NUMBER_LIMITED || !p.from_catalog) {
        int nh = rng.poisson(cs.expected_N);
        int kept = 0;
        for (int i = 0; i < nh; ++i) {
            float m = (float)sample_dndm_inv_dev(p, cs, rng);
            if (m >= (float)p.sampler_min_mass) {
                if (store_masses) {
                    M_out[kept] = m;
                }
                ++kept;
            }
        }
        *kept_count_out = kept;
        return true;
    }

    if (p.sample_method == SAMPLE_MASS_LIMITED) {
        double exp_M = cs.expected_M * p.halomass_correction;
        if (cs.sigma_cond * 7.0 * p.growth_out < cs.delta_crit) {
            if ((float)exp_M >= (float)p.sampler_min_mass) {
                if (store_masses) {
                    M_out[0] = (float)exp_M;
                }
                *kept_count_out = 1;
            }
            return true;
        }
        float buf[kLocalCap];
        int n = 0;
        double m_prog = 0.0;
        while (m_prog < exp_M) {
            if (n >= kLocalCap) {
                atomicExch(overflow_flag, 1);
                return false;
            }
            float m = (float)sample_dndm_inv_dev(p, cs, rng);
            m_prog += (double)m;
            buf[n++] = m;
        }
        if (n > 0) {
            bool sel = (rng.next_u32() & 1U) != 0U;
            if (sel) {
                if (fabs(m_prog - (double)buf[n - 1] - exp_M) < fabs(m_prog - exp_M)) {
                    m_prog -= (double)buf[n - 1];
                    --n;
                }
            } else {
                int ridx = 0;
                int remaining = n;
                double last_del = 0.0;
                do {
                    do {
                        ridx = (int)(rng.next_u32() % (uint32_t)n);
                    } while (buf[ridx] == 0.0f);
                    last_del = (double)buf[ridx];
                    m_prog -= last_del;
                    buf[ridx] = 0.0f;
                    --remaining;
                } while (m_prog > exp_M && remaining > 0);
                if (fabs(m_prog + last_del - exp_M) < fabs(m_prog - exp_M)) {
                    buf[ridx] = (float)last_del;
                    m_prog += last_del;
                }
            }
        }
        int kept = 0;
        for (int i = 0; i < n; ++i) {
            if (buf[i] >= (float)p.sampler_min_mass) {
                if (store_masses) {
                    M_out[kept] = buf[i];
                }
                ++kept;
            }
        }
        *kept_count_out = kept;
        return true;
    }

    if (p.sample_method == SAMPLE_PARTITION) {
        double M_rem = cs.M_cond;
        double lnM_rem = log(M_rem);
        int kept = 0;
        int guard = 0;
        while (M_rem > p.sampler_min_mass) {
            if (++guard > 100000) {
                atomicExch(overflow_flag, 1);
                return false;
            }
            double sigma_r = p21_eval_sigma_dev(lnM_rem, p.sig_tbl);
            double d_curr =
                (p21_get_delta_crit_dev(p.hmf, sigma_r, p.growth_out, p.delta_c_sph,
                                        p.delta_c_delos) -
                 cs.delta) /
                (M_rem / cs.M_cond);
            double del_term = (d_curr * d_curr) / (p.growth_out * p.growth_out);
            double sig_diff_min = p.sigma_min * p.sigma_min - sigma_r * sigma_r;
            if (sig_diff_min <= 0.0) {
                break;
            }
            double nu_min = sqrt(del_term / sig_diff_min);
            double sigma_sample = 0.0;
            while (true) {
                double nu_s = rng.ugaussian_tail(nu_min) * p.halomass_correction;
                sigma_sample = sqrt(del_term / (nu_s * nu_s) + sigma_r * sigma_r);
                if (p.hmf == HMF_PS) {
                    break;
                }
                double t1 = p21_st_taylor_factor_dev(sigma_sample, sigma_r, p.growth_out,
                                                     p.delta_c_sph, nullptr) -
                            d_curr / p.growth_out;
                double t2 = p21_st_taylor_factor_dev(p.sigma_min, sigma_r, p.growth_out,
                                                     p.delta_c_sph, nullptr) -
                            d_curr / p.growth_out;
                double u_r = rng.uniform();
                if (u_r <= (t2 / t1)) {
                    break;
                }
            }
            double m_sample = exp(p21_eval_sigma_inverse_dev(sigma_sample, p.sig_tbl));
            if ((float)m_sample >= (float)p.sampler_min_mass) {
                if (store_masses) {
                    M_out[kept] = (float)m_sample;
                }
                ++kept;
            }
            M_rem -= m_sample;
            if (M_rem <= p.sampler_min_mass) {
                break;
            }
            lnM_rem = log(M_rem);
        }
        *kept_count_out = kept;
        return true;
    }

    if (p.sample_method == SAMPLE_BINARY_SPLIT) {
        float d_pts[kLocalCap];
        float m_pts[kLocalCap];
        d_pts[0] = (float)(p.delta_c_sph / p.growth_in);
        m_pts[0] = (float)cs.M_cond;
        double d_target = p.delta_c_sph / p.growth_out;
        double m_res = p.M_min;
        double sigma_res = p21_eval_sigma_dev(p.lnM_min, p.sig_tbl);
        double sigsq_res = sigma_res * sigma_res;
        double eps1 = 0.1, eps2 = 0.1;

        int idx = 0;
        int n_points = 1;
        int kept = 0;

        while (idx < n_points) {
            double d_start = (double)d_pts[idx];
            double m_start = (double)m_pts[idx];
            double lnm_start = log(m_start);
            double dd_target = d_target - d_start;
            int save = 0;

            double m_half = 0.5 * m_start;
            double lnm_half = log(m_half);
            double sigma_start = p21_eval_sigma_dev(lnm_start, p.sig_tbl);
            double sigsq_start = sigma_start * sigma_start;
            double sigma_half = p21_eval_sigma_dev(lnm_half, p.sig_tbl);
            double sigsq_half = sigma_half * sigma_half;

            double G1 = p.parkinson_g0 * pow(d_start / sigma_start, p.parkinson_y2);
            double q_res = m_res / m_start;
            double q = 0.0;
            double F = 0.0;

            if (q_res >= 0.5) {
                double dd = eps1 * sqrt(2.0) * sqrt(fmax(0.0, sigsq_half - sigsq_start));
                if (dd >= dd_target) {
                    dd = dd_target;
                    save = 1;
                }
                double u_res = sigma_start * pow(sigsq_res - sigsq_start, -0.5);
                F = sqrt(2.0 / M_PI) *
                    eval_j_split_dev(u_res, p.parkinson_y1, p.j_split_1d, p.fract_float_err) * G1 /
                    sigma_start * dd;
                if (!save) {
                    d_pts[idx] = (float)(dd + d_start);
                }
            } else {
                double alpha_half =
                    -m_half / (2.0 * sigsq_half) * p21_eval_dsigmasqdm_dev(lnm_half, p.dsig_tbl);
                double V_res = sigsq_res * pow(sigsq_res - sigsq_start, -1.5);
                double V_half = sigsq_half * pow(sigsq_half - sigsq_start, -1.5);
                double beta = log(V_res / V_half) / log(2.0 * q_res);
                double B = pow(2.0, beta) * V_half;
                double dd1 = eps1 * sqrt(2.0) * sqrt(fmax(0.0, sigsq_half - sigsq_start));
                double mu = (p.parkinson_y1 < 0.0) ? (-log(sigma_res / sigma_half) / log(2.0 * q_res))
                                                  : alpha_half;
                double eta = beta - 1.0 - p.parkinson_y1 * mu;
                double pow_diff = pow(0.5, eta) - pow(q_res, eta);
                double G2 =
                    G1 * pow(sigma_half / sigma_start, p.parkinson_y1) * pow(0.5, mu * p.parkinson_y1);
                double dN_dd = sqrt(2.0 / M_PI) * B * pow_diff / eta * alpha_half * G2;
                double dd2 = eps2 / dN_dd;
                double dd = fmin(dd1, dd2);
                if (dd >= dd_target) {
                    dd = dd_target;
                    save = 1;
                }
                double N_upper = dN_dd * dd;
                double u_res = sigma_start * pow(sigsq_res - sigsq_start, -0.5);
                F = sqrt(2.0 / M_PI) *
                    eval_j_split_dev(u_res, p.parkinson_y1, p.j_split_1d, p.fract_float_err) * G1 /
                    sigma_start * dd;

                if (rng.uniform() < N_upper) {
                    q = pow(pow(q_res, eta) + pow_diff * rng.uniform(), 1.0 / eta);
                    double m_q = q * m_start;
                    double lnm_q = log(m_q);
                    double sigma_q = p21_eval_sigma_dev(lnm_q, p.sig_tbl);
                    double sigsq_q = sigma_q * sigma_q;
                    double alpha_q =
                        -m_q / (2.0 * sigsq_q) * p21_eval_dsigmasqdm_dev(lnm_q, p.dsig_tbl);
                    double R_q = (alpha_q / alpha_half) *
                                 (sigsq_q * pow(sigsq_q - sigsq_start, -1.5) / (B * pow(q, beta)));
                    if (rng.uniform() > R_q) {
                        q = 0.0;
                    }
                }
                if (!save) {
                    d_pts[idx] = (float)(dd + d_start);
                }
            }

            double m_prog1 = (1.0 - F - q) * m_start;
            double m_prog2 = q * m_start;

            if (save) {
                if (m_prog1 > m_res && (float)m_prog1 >= (float)p.sampler_min_mass) {
                    if (store_masses) {
                        if (kept >= kLocalCap) {
                            atomicExch(overflow_flag, 1);
                            return false;
                        }
                        M_out[kept] = (float)m_prog1;
                    }
                    ++kept;
                }
                if (m_prog2 > m_res && (float)m_prog2 >= (float)p.sampler_min_mass) {
                    if (store_masses) {
                        if (kept >= kLocalCap) {
                            atomicExch(overflow_flag, 1);
                            return false;
                        }
                        M_out[kept] = (float)m_prog2;
                    }
                    ++kept;
                }
            } else {
                float d_next = d_pts[idx];
                if (m_prog1 > m_res) {
                    d_pts[idx] = d_next;
                    m_pts[idx] = (float)m_prog1;
                    --idx;
                }
                if (m_prog2 > m_res) {
                    if (n_points >= kLocalCap) {
                        atomicExch(overflow_flag, 1);
                        return false;
                    }
                    d_pts[n_points] = d_next;
                    m_pts[n_points++] = (float)m_prog2;
                }
            }
            ++idx;
        }
        *kept_count_out = kept;
        return true;
    }

    return false;
}

__device__ static inline void wrap_pos_dev(double pos[3], const double size[3]) {
    p21_wrap_position(pos, size);
}

__device__ static inline void rand_in_cell_dev(int x, int y, int z, double cell_len,
                                               P21PhiloxRng &rng, double pt[3]) {
    pt[0] = ((double)x + (rng.uniform() - 0.5)) * cell_len;
    pt[1] = ((double)y + (rng.uniform() - 0.5)) * cell_len;
    pt[2] = ((double)z + (rng.uniform() - 0.5)) * cell_len;
}

__device__ static inline void rand_in_sphere_dev(const double c[3], double radius,
                                                 P21PhiloxRng &rng, double pt[3]) {
    double x1 = 2.0 * rng.uniform() - 1.0;
    double y1 = 2.0 * rng.uniform() - 1.0;
    double z1 = 2.0 * rng.uniform() - 1.0;
    double d1 = sqrt(x1 * x1 + y1 * y1 + z1 * z1);
    double r1 = rng.uniform();
    if (d1 <= 0.0) {
        pt[0] = c[0];
        pt[1] = c[1];
        pt[2] = c[2];
        return;
    }
    pt[0] = c[0] + (radius * r1 * x1 / d1);
    pt[1] = c[1] + (radius * r1 * y1 / d1);
    pt[2] = c[2] + (radius * r1 * z1 / d1);
}

__device__ static inline void set_prop_rng_dev(P21PhiloxRng &rng, bool correlated, double c0,
                                               double c1, double c2, float in0, float in1,
                                               float in2, float *o0, float *o1, float *o2) {
    double r0 = rng.ugaussian();
    double r1 = rng.ugaussian();
    double r2 = rng.ugaussian();
    if (correlated) {
        r0 = sqrt(1.0 - c0 * c0) * r0 + c0 * (double)in0;
        r1 = sqrt(1.0 - c1 * c1) * r1 + c1 * (double)in1;
        r2 = sqrt(1.0 - c2 * c2) * r2 + c2 * (double)in2;
    }
    *o0 = (float)r0;
    *o1 = (float)r1;
    *o2 = (float)r2;
}

__global__ void sampler_grid_count_kernel(size_t n_cells, SamplerKernelParams p, uint64_t seed,
                                          const float *dens_field, const float *halo_overlap_box,
                                          uint64_t *counts_out, int *overflow_flag) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) {
        return;
    }

    CondState cs{};
    setup_cond_state(p, (double)dens_field[idx] * p.growth_out, &cs);
    double mass_defc = (double)halo_overlap_box[idx];
    cs.expected_M *= (1.0 - mass_defc);
    cs.expected_N *= (1.0 - mass_defc);

    P21PhiloxRng rng;
    rng.init(seed, (uint64_t)idx, 0U);
    int kept = 0;
    stoc_sample_cond_dev(p, cs, rng, false, nullptr, &kept, overflow_flag);
    counts_out[idx] = (uint64_t)kept;
}

__global__ void sampler_grid_emit_kernel(size_t n_cells, SamplerKernelParams p, uint64_t seed,
                                         uint64_t base_offset, const float *dens_field,
                                         const float *halo_overlap_box, const uint64_t *offsets,
                                         float *out_masses, float *out_coords, float *out_star_rng,
                                         float *out_sfr_rng, float *out_xray_rng) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) {
        return;
    }

    size_t yz = (size_t)p.lo_dim[1] * (size_t)p.lo_dim[2];
    int x = (int)(idx / yz);
    size_t rem = idx - (size_t)x * yz;
    int y = (int)(rem / (size_t)p.lo_dim[2]);
    int z = (int)(rem - (size_t)y * (size_t)p.lo_dim[2]);

    CondState cs{};
    setup_cond_state(p, (double)dens_field[idx] * p.growth_out, &cs);
    double mass_defc = (double)halo_overlap_box[idx];
    cs.expected_M *= (1.0 - mass_defc);
    cs.expected_N *= (1.0 - mass_defc);

    P21PhiloxRng rng;
    rng.init(seed, (uint64_t)idx, 0U);

    uint64_t write_pos = base_offset + offsets[idx];
    int kept = 0;
    const double cell_len = p.boxlen / (double)p.lo_dim[0];
    const double box_size[3] = {p.boxlen, p.boxlen, p.boxlen_z};

    if (cs.delta <= p.delta_min) {
        return;
    }
    if (cs.delta >= p.max_deltac_frac * cs.delta_crit) {
        if ((float)cs.expected_M >= (float)p.sampler_min_mass) {
            out_masses[write_pos] = (float)cs.expected_M;
            kept = 1;
        }
    } else if (p.sample_method == SAMPLE_NUMBER_LIMITED || !p.from_catalog) {
        int nh = rng.poisson(cs.expected_N);
        for (int i = 0; i < nh; ++i) {
            float m = (float)sample_dndm_inv_dev(p, cs, rng);
            if (m >= (float)p.sampler_min_mass) {
                out_masses[write_pos + (uint64_t)kept] = m;
                ++kept;
            }
        }
    }

    /* Use separate substream=1 for coordinates & property RNG so Pass 1 never needs them. */
    P21PhiloxRng prng;
    prng.init(seed, (uint64_t)idx, 1U);
    for (int i = 0; i < kept; ++i) {
        double pt[3];
        rand_in_cell_dev(x, y, z, cell_len, prng, pt);
        wrap_pos_dev(pt, box_size);
        uint64_t dst = write_pos + (uint64_t)i;
        out_coords[3 * dst + 0] = (float)pt[0];
        out_coords[3 * dst + 1] = (float)pt[1];
        out_coords[3 * dst + 2] = (float)pt[2];
        set_prop_rng_dev(prng, false, 0.0, 0.0, 0.0, 0.0f, 0.0f, 0.0f, &out_star_rng[dst],
                         &out_sfr_rng[dst], &out_xray_rng[dst]);
    }
}

__global__ void sampler_prog_count_kernel(size_t n_in, SamplerKernelParams p, uint64_t seed,
                                          const float *in_masses, uint64_t *counts_out,
                                          int *overflow_flag, int *bad_mass_flag) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_in) {
        return;
    }

    double m2 = (double)in_masses[idx];
    if (m2 < p.M_min || m2 > p.M_max_tables) {
        atomicExch(bad_mass_flag, 1);
        counts_out[idx] = 0ULL;
        return;
    }

    CondState cs{};
    setup_cond_state(p, m2, &cs);

    P21PhiloxRng rng;
    rng.init(seed, (uint64_t)idx, 0U);
    int kept = 0;
    stoc_sample_cond_dev(p, cs, rng, false, nullptr, &kept, overflow_flag);
    counts_out[idx] = (uint64_t)kept;
}

__global__ void sampler_prog_emit_kernel(size_t n_in, SamplerKernelParams p, uint64_t seed,
                                         double m_to_r_pref, const float *in_masses,
                                         const float *in_coords, const float *in_star_rng,
                                         const float *in_sfr_rng, const float *in_xray_rng,
                                         const uint64_t *offsets, float *out_masses,
                                         float *out_coords, float *out_star_rng, float *out_sfr_rng,
                                         float *out_xray_rng, int *overflow_flag) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_in) {
        return;
    }

    double m2 = (double)in_masses[idx];
    if (m2 < p.M_min || m2 > p.M_max_tables) {
        return;
    }

    CondState cs{};
    setup_cond_state(p, m2, &cs);

    P21PhiloxRng rng;
    rng.init(seed, (uint64_t)idx, 0U);

    uint64_t write_pos = offsets[idx];
    int kept = 0;

    if (p.sample_method == SAMPLE_NUMBER_LIMITED && cs.delta > p.delta_min &&
        cs.delta < p.max_deltac_frac * cs.delta_crit) {
        int nh = rng.poisson(cs.expected_N);
        for (int i = 0; i < nh; ++i) {
            float m = (float)sample_dndm_inv_dev(p, cs, rng);
            if (m >= (float)p.sampler_min_mass) {
                out_masses[write_pos + (uint64_t)kept] = m;
                ++kept;
            }
        }
    } else {
        if (p.sample_method == SAMPLE_PARTITION) {
            stoc_sample_cond_dev(p, cs, rng, true, out_masses + write_pos, &kept, overflow_flag);
        } else {
            float tmp_masses[kLocalCap];
            if (!stoc_sample_cond_dev(p, cs, rng, true, tmp_masses, &kept, overflow_flag)) {
                return;
            }
            for (int i = 0; i < kept; ++i) {
                out_masses[write_pos + (uint64_t)i] = tmp_masses[i];
            }
        }
    }

    double r2 = cbrt(m2 / m_to_r_pref);
    double c_desc[3] = {(double)in_coords[3 * idx + 0], (double)in_coords[3 * idx + 1],
                        (double)in_coords[3 * idx + 2]};
    float sr_in = in_star_rng[idx];
    float sfr_in = in_sfr_rng[idx];
    float xr_in = in_xray_rng[idx];
    const double box_size[3] = {p.boxlen, p.boxlen, p.boxlen_z};

    P21PhiloxRng prng;
    prng.init(seed, (uint64_t)idx, 1U);
    for (int i = 0; i < kept; ++i) {
        uint64_t dst = write_pos + (uint64_t)i;
        float m_prog = out_masses[dst];
        double r1 = cbrt((double)m_prog / m_to_r_pref);
        double pt[3];
        rand_in_sphere_dev(c_desc, r2 - r1, prng, pt);
        wrap_pos_dev(pt, box_size);
        out_coords[3 * dst + 0] = (float)pt[0];
        out_coords[3 * dst + 1] = (float)pt[1];
        out_coords[3 * dst + 2] = (float)pt[2];
        set_prop_rng_dev(prng, true, p.corr_star, p.corr_sfr, p.corr_xray, sr_in, sfr_in, xr_in,
                         &out_star_rng[dst], &out_sfr_rng[dst], &out_xray_rng[dst]);
    }
}

/* ==================== Block Prefix Scan Utilities ==================== */

/* Chunked in-order exclusive prefix sum across arbitrary length n. */
__global__ void scan_phase1_chunk_sums_kernel(size_t n, size_t chunk, const uint64_t *in,
                                              uint64_t *blk_sums) {
    __shared__ uint64_t sh[kBlockThreads];
    size_t tid = threadIdx.x;
    size_t base = (size_t)blockIdx.x * chunk;
    size_t limit = base + chunk;
    if (limit > n) limit = n;

    uint64_t acc = 0;
    for (size_t i = base + tid; i < limit; i += kBlockThreads) {
        acc += in[i];
    }
    sh[tid] = acc;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            sh[tid] += sh[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        blk_sums[blockIdx.x] = sh[0];
    }
}

__global__ void scan_phase2_blocks_kernel(int nblocks, const uint64_t *blk_sums,
                                          uint64_t *blk_offsets, uint64_t *total_out) {
    __shared__ uint64_t sh[kMaxScanBlocks];
    int tid = (int)threadIdx.x;
    sh[tid] = (tid < nblocks) ? blk_sums[tid] : 0ULL;
    __syncthreads();
    for (int s = 1; s < kMaxScanBlocks; s <<= 1) {
        uint64_t add = (tid >= s) ? sh[tid - s] : 0ULL;
        __syncthreads();
        sh[tid] += add;
        __syncthreads();
    }
    if (tid < nblocks) {
        blk_offsets[tid] = (tid == 0) ? 0ULL : sh[tid - 1];
    }
    if (tid == 0) {
        *total_out = (nblocks > 0) ? sh[nblocks - 1] : 0ULL;
    }
}

__global__ void scan_phase3_write_offsets_kernel(size_t n, size_t chunk, const uint64_t *in,
                                                 const uint64_t *blk_offsets, uint64_t *out) {
    __shared__ uint64_t sh[kBlockThreads];
    int tid = (int)threadIdx.x;
    size_t base = (size_t)blockIdx.x * chunk;
    size_t limit = base + chunk;
    if (limit > n) limit = n;

    uint64_t running = blk_offsets[blockIdx.x];
    for (size_t off = 0; base + off < limit; off += kBlockThreads) {
        size_t idx = base + off + (size_t)tid;
        uint64_t val = (idx < limit) ? in[idx] : 0ULL;
        sh[tid] = val;
        __syncthreads();
        for (int s = 1; s < kBlockThreads; s <<= 1) {
            uint64_t add = (tid >= s) ? sh[tid - s] : 0ULL;
            __syncthreads();
            sh[tid] += add;
            __syncthreads();
        }
        uint64_t excl = running + (sh[tid] - val);
        uint64_t tile_tot = sh[kBlockThreads - 1];
        if (idx < limit) {
            out[idx] = excl;
        }
        running += tile_tot;
        __syncthreads();
    }
}

bool exclusive_scan_u64(size_t n, const uint64_t *d_in, uint64_t *d_out, uint64_t *total_out,
                        cudaStream_t stream) {
    *total_out = 0;
    if (n == 0) {
        return true;
    }
    int max_needed = (int)((n + kBlockThreads - 1) / kBlockThreads);
    int nblocks = max_needed < kMaxScanBlocks ? max_needed : kMaxScanBlocks;
    size_t raw_chunk = (n + (size_t)nblocks - 1) / (size_t)nblocks;
    size_t chunk = ((raw_chunk + kBlockThreads - 1) / kBlockThreads) * kBlockThreads;

    uint64_t *d_blk_sums = (uint64_t *)p21_cuda_pool_alloc(kMaxScanBlocks * sizeof(uint64_t), stream);
    uint64_t *d_blk_off = (uint64_t *)p21_cuda_pool_alloc(kMaxScanBlocks * sizeof(uint64_t), stream);
    uint64_t *d_tot = (uint64_t *)p21_cuda_pool_alloc(sizeof(uint64_t), stream);
    if (!d_blk_sums || !d_blk_off || !d_tot) {
        p21_cuda_pool_free(d_blk_sums, stream);
        p21_cuda_pool_free(d_blk_off, stream);
        p21_cuda_pool_free(d_tot, stream);
        return false;
    }

    scan_phase1_chunk_sums_kernel<<<nblocks, kBlockThreads, 0, stream>>>(n, chunk, d_in, d_blk_sums);
    bool ok = P21_CUDA_OK(cudaGetLastError());
    if (ok) {
        scan_phase2_blocks_kernel<<<1, kMaxScanBlocks, 0, stream>>>(nblocks, d_blk_sums, d_blk_off,
                                                                    d_tot);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    if (ok) {
        scan_phase3_write_offsets_kernel<<<nblocks, kBlockThreads, 0, stream>>>(n, chunk, d_in,
                                                                                d_blk_off, d_out);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(total_out, d_tot, sizeof(uint64_t),
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }

    p21_cuda_pool_free(d_blk_sums, stream);
    p21_cuda_pool_free(d_blk_off, stream);
    p21_cuda_pool_free(d_tot, stream);
    return ok;
}

bool upload_sampler_kernel_params(const P21HaloSamplingParams *hsp, bool need_inv,
                                  bool need_jsplit, cudaStream_t stream,
                                  SamplerKernelParams *out) {
    P21SamplerTables st{};
    p21_get_sampler_tables(&st);
    if (!st.sigma_1d || !st.dsigmasqdm_1d || !st.nhalo_1d || !st.fcoll_1d) {
        return false;
    }
    if (need_inv && !st.nhalo_inv_2d) {
        return false;
    }
    if (need_jsplit && !st.j_split_1d) {
        return false;
    }

    memset(out, 0, sizeof(*out));
    out->from_catalog = hsp->from_catalog;
    out->hmf = matter_options_global->HMF;
    out->sample_method = matter_options_global->SAMPLE_METHOD;
    out->growth_in = hsp->growth_in;
    out->growth_out = hsp->growth_out;
    out->M_min = hsp->M_min;
    out->lnM_min = hsp->lnM_min;
    out->sigma_min = hsp->sigma_min;
    out->M_max_tables = hsp->M_max_tables;
    out->M_cond = hsp->M_cond;
    out->lnM_cond = hsp->lnM_cond;
    out->sigma_cond = hsp->sigma_cond;
    out->corr_star = hsp->corr_star;
    out->corr_sfr = hsp->corr_sfr;
    out->corr_xray = hsp->corr_xray;
    out->sampler_min_mass = simulation_options_global->SAMPLER_MIN_MASS;
    out->min_logprob = simulation_options_global->MIN_LOGPROB;
    out->halomass_correction = simulation_options_global->HALOMASS_CORRECTION;
    out->parkinson_g0 = simulation_options_global->PARKINSON_G0;
    out->parkinson_y1 = simulation_options_global->PARKINSON_y1;
    out->parkinson_y2 = simulation_options_global->PARKINSON_y2;
    out->delta_c_sph = physconst.delta_c_sph;
    out->delta_c_delos = physconst.delta_c_delos;
    out->rhocrit_omm = RHOcrit * cosmo_params_global->OMm;
    out->max_deltac_frac = MAX_DELTAC_FRAC;
    out->delta_min = DELTA_MIN;
    out->fract_float_err = FRACT_FLOAT_ERR;
    out->boxlen = simulation_options_global->BOX_LEN;
    out->boxlen_z = simulation_options_global->BOX_LEN * simulation_options_global->NON_CUBIC_FACTOR;
    out->lo_dim[0] = simulation_options_global->HII_DIM;
    out->lo_dim[1] = simulation_options_global->HII_DIM;
    out->lo_dim[2] = HII_D_PARA;

    bool ok = p21_upload_rg1d_f(st.sigma_1d, &out->sig_tbl, stream) &&
              p21_upload_rg1d_f(st.dsigmasqdm_1d, &out->dsig_tbl, stream) &&
              p21_upload_rg1d_d(st.nhalo_1d, &out->nhalo_1d, stream) &&
              p21_upload_rg1d_d(st.fcoll_1d, &out->fcoll_1d, stream);
    if (ok && need_inv) {
        ok = p21_upload_rg2d_d(st.nhalo_inv_2d, &out->nhalo_inv_2d, stream);
    }
    if (ok && need_jsplit) {
        ok = p21_upload_rg1d_d(st.j_split_1d, &out->j_split_1d, stream);
    }
    return ok;
}

void free_sampler_kernel_params(SamplerKernelParams *p, cudaStream_t stream) {
    p21_free_dev_table(&p->sig_tbl, stream);
    p21_free_dev_table(&p->dsig_tbl, stream);
    p21_free_dev_table(&p->nhalo_1d, stream);
    p21_free_dev_table(&p->fcoll_1d, stream);
    p21_free_dev_table(&p->nhalo_inv_2d, stream);
    p21_free_dev_table(&p->j_split_1d, stream);
}

void omp_static_bounds(size_t n, int t, int n_threads, size_t *start_out, size_t *end_out) {
    size_t base = n / (size_t)n_threads;
    size_t rem = n % (size_t)n_threads;
    size_t ut = (size_t)t;
    size_t s = ut * base + (ut < rem ? ut : rem);
    size_t len = base + (ut < rem ? 1ULL : 0ULL);
    *start_out = s;
    *end_out = s + len;
}

void zero_halo_catalog_tail(HaloCatalog *hc, size_t start_idx, size_t buf_size) {
    if (!hc || start_idx >= buf_size) {
        return;
    }
    size_t count = buf_size - start_idx;
    memset(hc->halo_masses + start_idx, 0, count * sizeof(float));
    memset(hc->halo_coords + 3ULL * start_idx, 0, 3ULL * count * sizeof(float));
    memset(hc->star_rng + start_idx, 0, count * sizeof(float));
    memset(hc->sfr_rng + start_idx, 0, count * sizeof(float));
    memset(hc->xray_rng + start_idx, 0, count * sizeof(float));
}

bool finalize_sampler_output(HaloCatalog *out, size_t total_halos, size_t buf_size, float *&d_out_m,
                             float *&d_out_c, float *&d_out_sr, float *&d_out_sfrr,
                             float *&d_out_xr, cudaStream_t stream) {
    const size_t out_sbytes = total_halos * sizeof(float);
    const size_t out_cbytes = 3ULL * out_sbytes;
    bool ok = P21_CUDA_OK(cudaMemcpyAsync(out->halo_masses, d_out_m, out_sbytes,
                                          cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaMemcpyAsync(out->halo_coords, d_out_c, out_cbytes,
                                          cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaMemcpyAsync(out->star_rng, d_out_sr, out_sbytes,
                                          cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaMemcpyAsync(out->sfr_rng, d_out_sfrr, out_sbytes,
                                          cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaMemcpyAsync(out->xray_rng, d_out_xr, out_sbytes,
                                          cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaStreamSynchronize(stream));
    if (!ok) {
        return false;
    }
    p21_cuda_stats_add_d2h(4ULL * out_sbytes + out_cbytes);
    out->n_halos = (unsigned long long int)total_halos;
    zero_halo_catalog_tail(out, total_halos, buf_size);

    auto own_or_free = [&](const void *hptr, float *&dbuf, size_t nbytes) {
        if (!dbuf) return;
        if (!hptr || !p21_cuda_registry_put_owned(hptr, dbuf, nbytes, stream)) {
            p21_cuda_pool_free(dbuf, stream);
        }
        dbuf = nullptr;
    };
    own_or_free(out->halo_masses, d_out_m, out_sbytes);
    own_or_free(out->halo_coords, d_out_c, out_cbytes);
    own_or_free(out->star_rng, d_out_sr, out_sbytes);
    own_or_free(out->sfr_rng, d_out_sfrr, out_sbytes);
    own_or_free(out->xray_rng, d_out_xr, out_sbytes);
    p21_cuda_stats_mut()->halo_sampler_runs++;
    return true;
}

}  // namespace

/* ==================== C API Entrypoints ==================== */

extern "C" int p21_cuda_fill_dndm_tables(int nx, double xmin, double xmax, double ymin, double ymax,
                                         double growth_out, double param, int from_catalog,
                                         double *nhalo_out, double *fcoll_out) {
    if (!p21_cuda_get_enabled() || nx < 2 || !nhalo_out || !fcoll_out) {
        return -1;
    }
    if (!uses_interpolation_tables(matter_options_global->USE_INTERPOLATION_TABLES)) {
        return -1;
    }
    P21SamplerTables st{};
    p21_get_sampler_tables(&st);
    if (!st.sigma_1d || !st.dsigmasqdm_1d) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    P21DevTable1Df sig_tbl{};
    P21DevTable1Df dsig_tbl{};
    size_t bytes = (size_t)nx * sizeof(double);
    double *d_nhalo = (double *)p21_cuda_pool_alloc(bytes, stream);
    double *d_fcoll = (double *)p21_cuda_pool_alloc(bytes, stream);

    bool ok = d_nhalo && d_fcoll && p21_upload_rg1d_f(st.sigma_1d, &sig_tbl, stream) &&
              p21_upload_rg1d_f(st.dsigmasqdm_1d, &dsig_tbl, stream);

    if (ok) {
        double rhocrit_omm = RHOcrit * cosmo_params_global->OMm;
        dndm_tables_kernel<<<nx, kBlockThreads, 0, stream>>>(
            nx, xmin, xmax, ymin, ymax, growth_out, param, from_catalog,
            matter_options_global->HMF, physconst.delta_c_sph, physconst.delta_c_delos, rhocrit_omm,
            MAX_DELTAC_FRAC, FRACT_FLOAT_ERR, sig_tbl, dsig_tbl, d_nhalo, d_fcoll);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(nhalo_out, d_nhalo, bytes, cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(fcoll_out, d_fcoll, bytes, cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }

    p21_free_dev_table(&sig_tbl, stream);
    p21_free_dev_table(&dsig_tbl, stream);
    p21_cuda_pool_free(d_nhalo, stream);
    p21_cuda_pool_free(d_fcoll, stream);

    if (!ok) {
        return -1;
    }
    p21_cuda_stats_mut()->dndm_table_runs++;
    return 0;
}

extern "C" int p21_cuda_fill_dndm_inverse_table(int nx, int np, double xmin, double xmax,
                                                double lnM_min, double growth_out, double param,
                                                int from_catalog, double min_lp,
                                                double **z_arr_out) {
    if (!p21_cuda_get_enabled() || nx < 2 || np < 2 || !z_arr_out) {
        return -1;
    }
    if (!uses_interpolation_tables(matter_options_global->USE_INTERPOLATION_TABLES)) {
        return -1;
    }
    P21SamplerTables st{};
    p21_get_sampler_tables(&st);
    if (!st.sigma_1d || !st.dsigmasqdm_1d) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    P21DevTable1Df sig_tbl{};
    P21DevTable1Df dsig_tbl{};
    size_t total = (size_t)nx * (size_t)np;
    size_t bytes = total * sizeof(double);
    double *d_z_flat = (double *)p21_cuda_pool_alloc(bytes, stream);
    double *h_z_flat = (double *)malloc(bytes);

    bool ok = d_z_flat && h_z_flat && p21_upload_rg1d_f(st.sigma_1d, &sig_tbl, stream) &&
              p21_upload_rg1d_f(st.dsigmasqdm_1d, &dsig_tbl, stream);

    if (ok) {
        dndm_inverse_kernel<<<nx, kBlockThreads, 0, stream>>>(
            nx, np, xmin, xmax, lnM_min, growth_out, param, from_catalog,
            matter_options_global->HMF, physconst.delta_c_sph, physconst.delta_c_delos,
            MAX_DELTAC_FRAC, min_lp, sig_tbl, dsig_tbl, d_z_flat);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(h_z_flat, d_z_flat, bytes, cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }
    if (ok) {
        for (int i = 0; i < nx; ++i) {
            memcpy(z_arr_out[i], h_z_flat + (size_t)i * (size_t)np, (size_t)np * sizeof(double));
        }
    }

    free(h_z_flat);
    p21_free_dev_table(&sig_tbl, stream);
    p21_free_dev_table(&dsig_tbl, stream);
    p21_cuda_pool_free(d_z_flat, stream);

    if (!ok) {
        return -1;
    }
    p21_cuda_stats_mut()->dndm_table_runs++;
    return 0;
}

extern "C" int p21_cuda_sample_halo_grids(unsigned long long int seed, double redshift,
                                          float *dens_field, float *halo_overlap_box,
                                          HaloCatalog *halofield_large, HaloCatalog *halofield_out,
                                          const P21HaloSamplingParams *hsp, int *overflow_out,
                                          unsigned long long int *thread_counts_out) {
    (void)redshift;
    if (!p21_cuda_get_enabled() || !p21_cuda_get_fast_halo_sampler() || !dens_field ||
        !halo_overlap_box || !halofield_large || !halofield_out || !hsp || !overflow_out ||
        !thread_counts_out) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    *overflow_out = 0;
    const int n_threads =
        simulation_options_global->N_THREADS > 0 ? simulation_options_global->N_THREADS : 1;
    const size_t nhalo_in = (size_t)halofield_large->n_halos;
    const size_t lo_nx = (size_t)simulation_options_global->HII_DIM;
    const size_t lo_nz = (size_t)HII_D_PARA;
    const size_t n_cells = lo_nx * lo_nx * lo_nz;
    const size_t lo_bytes = n_cells * sizeof(float);
    const size_t buf_size = (size_t)halofield_out->buffer_size;
    const size_t arraysize_local = buf_size / (size_t)n_threads;

    cudaStream_t stream = p21_cuda_stream();
    SamplerKernelParams skp{};
    if (!upload_sampler_kernel_params(hsp, true, false, stream, &skp)) {
        free_sampler_kernel_params(&skp, stream);
        return -1;
    }

    const float *d_dens = (const float *)p21_cuda_registry_get_or_upload(dens_field, lo_bytes, stream);
    const float *d_overlap =
        (const float *)p21_cuda_registry_get_or_upload(halo_overlap_box, lo_bytes, stream);
    uint64_t *d_counts = (uint64_t *)p21_cuda_pool_alloc(n_cells * sizeof(uint64_t), stream);
    uint64_t *d_offsets = (uint64_t *)p21_cuda_pool_alloc(n_cells * sizeof(uint64_t), stream);
    int *d_overflow = (int *)p21_cuda_pool_alloc(sizeof(int), stream);

    bool ok = d_dens && d_overlap && d_counts && d_offsets && d_overflow &&
              P21_CUDA_OK(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream));

    int h_local_overflow = 0;
    uint64_t n_sampled = 0;
    if (ok) {
        int blocks = (int)((n_cells + kBlockThreads - 1) / kBlockThreads);
        sampler_grid_count_kernel<<<blocks, kBlockThreads, 0, stream>>>(
            n_cells, skp, (uint64_t)seed, d_dens, d_overlap, d_counts, d_overflow);
        ok = P21_CUDA_OK(cudaGetLastError()) &&
             P21_CUDA_OK(cudaMemcpyAsync(&h_local_overflow, d_overflow, sizeof(int),
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }
    if (ok && h_local_overflow == 0) {
        ok = exclusive_scan_u64(n_cells, d_counts, d_offsets, &n_sampled, stream);
    }
    p21_cuda_pool_free(d_overflow, stream);

    if (!ok || h_local_overflow != 0) {
        p21_cuda_pool_free(d_counts, stream);
        p21_cuda_pool_free(d_offsets, stream);
        free_sampler_kernel_params(&skp, stream);
        return -1;
    }

    uint64_t total_halos = (uint64_t)nhalo_in + n_sampled;
    bool any_overflow = (total_halos > buf_size);

    for (int t = 0; t < n_threads; ++t) {
        size_t ls, le, xs, xe;
        omp_static_bounds(nhalo_in, t, n_threads, &ls, &le);
        omp_static_bounds(lo_nx, t, n_threads, &xs, &xe);
        uint64_t cnt = (uint64_t)(le - ls);
        if (xe > xs) {
            size_t c_start = xs * lo_nx * lo_nz;
            size_t c_end = xe * lo_nx * lo_nz;
            uint64_t off_s = 0, off_e = 0;
            ok = ok && P21_CUDA_OK(cudaMemcpyAsync(&off_s, d_offsets + c_start, sizeof(uint64_t),
                                                   cudaMemcpyDeviceToHost, stream));
            if (c_end < n_cells) {
                ok = ok && P21_CUDA_OK(cudaMemcpyAsync(&off_e, d_offsets + c_end, sizeof(uint64_t),
                                                       cudaMemcpyDeviceToHost, stream));
            } else {
                off_e = n_sampled;
            }
            if (ok) {
                ok = P21_CUDA_OK(cudaStreamSynchronize(stream));
                cnt += (off_e - off_s);
            }
        }
        thread_counts_out[t] = (unsigned long long int)cnt;
        if (cnt > arraysize_local) {
            any_overflow = true;
        }
    }

    if (!ok) {
        p21_cuda_pool_free(d_counts, stream);
        p21_cuda_pool_free(d_offsets, stream);
        free_sampler_kernel_params(&skp, stream);
        return -1;
    }

    if (any_overflow) {
        *overflow_out = 1;
        p21_cuda_pool_free(d_counts, stream);
        p21_cuda_pool_free(d_offsets, stream);
        free_sampler_kernel_params(&skp, stream);
        return 0;
    }

    if (total_halos == 0) {
        halofield_out->n_halos = 0;
        zero_halo_catalog_tail(halofield_out, 0, buf_size);
        p21_cuda_pool_free(d_counts, stream);
        p21_cuda_pool_free(d_offsets, stream);
        free_sampler_kernel_params(&skp, stream);
        p21_cuda_stats_mut()->halo_sampler_runs++;
        return 0;
    }

    if (nhalo_in > 0 && (!halofield_large->halo_masses || !halofield_large->halo_coords ||
                         !halofield_large->star_rng || !halofield_large->sfr_rng ||
                         !halofield_large->xray_rng)) {
        p21_cuda_pool_free(d_counts, stream);
        p21_cuda_pool_free(d_offsets, stream);
        free_sampler_kernel_params(&skp, stream);
        return -1;
    }

    const size_t out_sbytes = (size_t)total_halos * sizeof(float);
    const size_t out_cbytes = 3ULL * out_sbytes;
    float *d_out_m = (float *)p21_cuda_pool_alloc(out_sbytes, stream);
    float *d_out_c = (float *)p21_cuda_pool_alloc(out_cbytes, stream);
    float *d_out_sr = (float *)p21_cuda_pool_alloc(out_sbytes, stream);
    float *d_out_sfrr = (float *)p21_cuda_pool_alloc(out_sbytes, stream);
    float *d_out_xr = (float *)p21_cuda_pool_alloc(out_sbytes, stream);

    if (!d_out_m || !d_out_c || !d_out_sr || !d_out_sfrr || !d_out_xr) {
        ok = false;
    }

    if (ok && nhalo_in > 0) {
        const size_t in_sbytes = nhalo_in * sizeof(float);
        const size_t in_cbytes = 3ULL * in_sbytes;
        ok = P21_CUDA_OK(cudaMemcpyAsync(d_out_m, halofield_large->halo_masses, in_sbytes,
                                         cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(d_out_c, halofield_large->halo_coords, in_cbytes,
                                         cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(d_out_sr, halofield_large->star_rng, in_sbytes,
                                         cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(d_out_sfrr, halofield_large->sfr_rng, in_sbytes,
                                         cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(d_out_xr, halofield_large->xray_rng, in_sbytes,
                                         cudaMemcpyHostToDevice, stream));
    }

    if (ok && n_sampled > 0) {
        int blocks = (int)((n_cells + kBlockThreads - 1) / kBlockThreads);
        sampler_grid_emit_kernel<<<blocks, kBlockThreads, 0, stream>>>(
            n_cells, skp, (uint64_t)seed, (uint64_t)nhalo_in, d_dens, d_overlap, d_offsets,
            d_out_m, d_out_c, d_out_sr, d_out_sfrr, d_out_xr);
        ok = P21_CUDA_OK(cudaGetLastError());
    }

    p21_cuda_pool_free(d_counts, stream);
    p21_cuda_pool_free(d_offsets, stream);
    free_sampler_kernel_params(&skp, stream);

    if (ok && finalize_sampler_output(halofield_out, (size_t)total_halos, buf_size, d_out_m,
                                      d_out_c, d_out_sr, d_out_sfrr, d_out_xr, stream)) {
        return 0;
    }

    p21_cuda_pool_free(d_out_m, stream);
    p21_cuda_pool_free(d_out_c, stream);
    p21_cuda_pool_free(d_out_sr, stream);
    p21_cuda_pool_free(d_out_sfrr, stream);
    p21_cuda_pool_free(d_out_xr, stream);
    return -1;
}

extern "C" int p21_cuda_sample_halo_progenitors(unsigned long long int seed, double z_in,
                                                double z_out, HaloCatalog *halofield_in,
                                                HaloCatalog *halofield_out,
                                                const P21HaloSamplingParams *hsp,
                                                int *overflow_out,
                                                unsigned long long int *thread_counts_out) {
    if (!p21_cuda_get_enabled() || !p21_cuda_get_fast_halo_sampler() || !halofield_in ||
        !halofield_out || !hsp || !overflow_out || !thread_counts_out) {
        return -1;
    }
    if (z_in >= z_out) {
        return ValueError;
    }
    const int method = matter_options_global->SAMPLE_METHOD;
    const int hmf = matter_options_global->HMF;
    if (method == SAMPLE_PARTITION && hmf != HMF_PS && hmf != HMF_ST) {
        return ValueError;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    *overflow_out = 0;
    const size_t nhalo_in = (size_t)halofield_in->n_halos;
    const size_t buf_size = (size_t)halofield_out->buffer_size;
    if (nhalo_in == 0) {
        halofield_out->n_halos = 0;
        zero_halo_catalog_tail(halofield_out, 0, buf_size);
        p21_cuda_stats_mut()->halo_sampler_runs++;
        return 0;
    }

    const int n_threads =
        simulation_options_global->N_THREADS > 0 ? simulation_options_global->N_THREADS : 1;
    const size_t arraysize_local = buf_size / (size_t)n_threads;

    const bool need_inv =
        (method == SAMPLE_NUMBER_LIMITED || method == SAMPLE_MASS_LIMITED);
    const bool need_jsplit = (method == SAMPLE_BINARY_SPLIT);

    cudaStream_t stream = p21_cuda_stream();
    SamplerKernelParams skp{};
    if (!upload_sampler_kernel_params(hsp, need_inv, need_jsplit, stream, &skp)) {
        free_sampler_kernel_params(&skp, stream);
        return -1;
    }

    const size_t in_sbytes = nhalo_in * sizeof(float);
    const size_t in_cbytes = 3ULL * in_sbytes;
    const float *d_in_m = (const float *)p21_cuda_registry_get_or_upload(halofield_in->halo_masses,
                                                                        in_sbytes, stream);
    const float *d_in_c = (const float *)p21_cuda_registry_get_or_upload(halofield_in->halo_coords,
                                                                        in_cbytes, stream);
    const float *d_in_sr =
        (const float *)p21_cuda_registry_get_or_upload(halofield_in->star_rng, in_sbytes, stream);
    const float *d_in_sfrr =
        (const float *)p21_cuda_registry_get_or_upload(halofield_in->sfr_rng, in_sbytes, stream);
    const float *d_in_xr =
        (const float *)p21_cuda_registry_get_or_upload(halofield_in->xray_rng, in_sbytes, stream);

    uint64_t *d_counts = (uint64_t *)p21_cuda_pool_alloc(nhalo_in * sizeof(uint64_t), stream);
    uint64_t *d_offsets = (uint64_t *)p21_cuda_pool_alloc(nhalo_in * sizeof(uint64_t), stream);
    int *d_flags = (int *)p21_cuda_pool_alloc(2 * sizeof(int), stream);

    bool ok = d_in_m && d_in_c && d_in_sr && d_in_sfrr && d_in_xr && d_counts && d_offsets &&
              d_flags && P21_CUDA_OK(cudaMemsetAsync(d_flags, 0, 2 * sizeof(int), stream));

    int h_flags[2] = {0, 0};
    uint64_t total_halos = 0;
    if (ok) {
        int blocks = (int)((nhalo_in + kBlockThreads - 1) / kBlockThreads);
        sampler_prog_count_kernel<<<blocks, kBlockThreads, 0, stream>>>(
            nhalo_in, skp, (uint64_t)seed, d_in_m, d_counts, d_flags, d_flags + 1);
        ok = P21_CUDA_OK(cudaGetLastError()) &&
             P21_CUDA_OK(cudaMemcpyAsync(h_flags, d_flags, 2 * sizeof(int), cudaMemcpyDeviceToHost,
                                         stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }
    if (ok && h_flags[1] != 0) {
        p21_cuda_pool_free(d_counts, stream);
        p21_cuda_pool_free(d_offsets, stream);
        p21_cuda_pool_free(d_flags, stream);
        free_sampler_kernel_params(&skp, stream);
        return ValueError;
    }
    if (ok && h_flags[0] == 0) {
        ok = exclusive_scan_u64(nhalo_in, d_counts, d_offsets, &total_halos, stream);
    }

    if (!ok || h_flags[0] != 0) {
        p21_cuda_pool_free(d_counts, stream);
        p21_cuda_pool_free(d_offsets, stream);
        p21_cuda_pool_free(d_flags, stream);
        free_sampler_kernel_params(&skp, stream);
        return -1;
    }

    bool any_overflow = (total_halos > buf_size);
    for (int t = 0; t < n_threads; ++t) {
        size_t s, e;
        omp_static_bounds(nhalo_in, t, n_threads, &s, &e);
        uint64_t cnt = 0;
        if (e > s) {
            uint64_t off_s = 0, off_e = 0;
            ok = ok && P21_CUDA_OK(cudaMemcpyAsync(&off_s, d_offsets + s, sizeof(uint64_t),
                                                   cudaMemcpyDeviceToHost, stream));
            if (e < nhalo_in) {
                ok = ok && P21_CUDA_OK(cudaMemcpyAsync(&off_e, d_offsets + e, sizeof(uint64_t),
                                                       cudaMemcpyDeviceToHost, stream));
            } else {
                off_e = total_halos;
            }
            if (ok) {
                ok = P21_CUDA_OK(cudaStreamSynchronize(stream));
                cnt = off_e - off_s;
            }
        }
        thread_counts_out[t] = (unsigned long long int)cnt;
        if (cnt > arraysize_local) {
            any_overflow = true;
        }
    }

    if (!ok) {
        p21_cuda_pool_free(d_counts, stream);
        p21_cuda_pool_free(d_offsets, stream);
        p21_cuda_pool_free(d_flags, stream);
        free_sampler_kernel_params(&skp, stream);
        return -1;
    }

    if (any_overflow) {
        *overflow_out = 1;
        p21_cuda_pool_free(d_counts, stream);
        p21_cuda_pool_free(d_offsets, stream);
        p21_cuda_pool_free(d_flags, stream);
        free_sampler_kernel_params(&skp, stream);
        return 0;
    }

    if (total_halos == 0) {
        halofield_out->n_halos = 0;
        zero_halo_catalog_tail(halofield_out, 0, buf_size);
        p21_cuda_pool_free(d_counts, stream);
        p21_cuda_pool_free(d_offsets, stream);
        p21_cuda_pool_free(d_flags, stream);
        free_sampler_kernel_params(&skp, stream);
        p21_cuda_stats_mut()->halo_sampler_runs++;
        return 0;
    }

    const size_t out_sbytes = (size_t)total_halos * sizeof(float);
    const size_t out_cbytes = 3ULL * out_sbytes;
    float *d_out_m = (float *)p21_cuda_pool_alloc(out_sbytes, stream);
    float *d_out_c = (float *)p21_cuda_pool_alloc(out_cbytes, stream);
    float *d_out_sr = (float *)p21_cuda_pool_alloc(out_sbytes, stream);
    float *d_out_sfrr = (float *)p21_cuda_pool_alloc(out_sbytes, stream);
    float *d_out_xr = (float *)p21_cuda_pool_alloc(out_sbytes, stream);

    if (!d_out_m || !d_out_c || !d_out_sr || !d_out_sfrr || !d_out_xr) {
        ok = false;
    }

    if (ok) {
        const double m_to_r_pref =
            (matter_options_global->FILTER == FILTER_GAUSSIAN ? pow(2.0 * M_PI, 1.5)
                                                              : (4.0 * M_PI / 3.0)) *
            RHOcrit * cosmo_params_global->OMm;
        int blocks = (int)((nhalo_in + kBlockThreads - 1) / kBlockThreads);
        sampler_prog_emit_kernel<<<blocks, kBlockThreads, 0, stream>>>(
            nhalo_in, skp, (uint64_t)seed, m_to_r_pref, d_in_m, d_in_c, d_in_sr, d_in_sfrr,
            d_in_xr, d_offsets, d_out_m, d_out_c, d_out_sr, d_out_sfrr, d_out_xr, d_flags);
        ok = P21_CUDA_OK(cudaGetLastError()) &&
             P21_CUDA_OK(cudaMemcpyAsync(h_flags, d_flags, sizeof(int), cudaMemcpyDeviceToHost,
                                         stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }

    p21_cuda_pool_free(d_counts, stream);
    p21_cuda_pool_free(d_offsets, stream);
    p21_cuda_pool_free(d_flags, stream);
    free_sampler_kernel_params(&skp, stream);

    if (ok && h_flags[0] == 0 &&
        finalize_sampler_output(halofield_out, (size_t)total_halos, buf_size, d_out_m, d_out_c,
                                d_out_sr, d_out_sfrr, d_out_xr, stream)) {
        return 0;
    }

    p21_cuda_pool_free(d_out_m, stream);
    p21_cuda_pool_free(d_out_c, stream);
    p21_cuda_pool_free(d_out_sr, stream);
    p21_cuda_pool_free(d_out_sfrr, stream);
    p21_cuda_pool_free(d_out_xr, stream);
    return -1;
}
