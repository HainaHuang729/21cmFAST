#include <math.h>
#include <string.h>

#include "cuda_common.cuh"
#include "cuda_halo_props.cuh"
#include "cuda_tables.cuh"

struct P21CudaEmissivityCtx {
    size_t num_pixels = 0;
    const float *d_dens = nullptr;
    float *d_mturn_acg = nullptr;
    float *d_mturn_mcg = nullptr;
    cudaStream_t stream = nullptr;
};

struct P21CudaLagEmissCtx {
    int out_dim[3] = {0, 0, 0};
    int dens_dim[3] = {0, 0, 0};
    size_t out_pixels = 0;
    size_t dens_pixels = 0;
    int use_ts_fluct = 0;
    int use_mcgs = 0;
    int use_metallicity = 0;
    int use_reion_fb = 0;
    int uses_recomb = 0;
    int extra_fields = 0;
    int mass_dep = 0;
    const float *d_dens = nullptr;
    float *d_mturn_acg = nullptr;
    float *d_mturn_mcg = nullptr;
    double *d_acc_nion = nullptr;
    double *d_acc_sfrd_acg = nullptr;
    double *d_acc_sfrd_mcg = nullptr;
    double *d_acc_xray = nullptr;
    double *d_acc_fesc_wsfr = nullptr;
    double *d_acc_hn = nullptr;
    double *d_acc_hm = nullptr;
    double *d_acc_sa = nullptr;
    double *d_acc_sm = nullptr;
    cudaStream_t stream = nullptr;
};

namespace {

constexpr int kBlockThreads = 256;

struct EmissivityEvalParams {
    double redshift;
    double growth_factor;
    double mturn_acg_homogeneous;
    double atomic_cooling_threshold;
    double sfr_timescale;
    double lx_over_sfr_acg;
    double lx_over_sfr_mcg;
    double fstar_10_acg;
    double hii_eff_factor;
    double rhocrit_omm;
    double rhocrit_omb;
    double cell_volume;
    double fract_float_err;
    int use_ts_fluct;
    int use_mcgs;
    int use_metallicity;
    int use_reion_feedback;
    int mass_dep_zeta;
    int extra_fields;
    int fix_mean;

    P21DevTable1Df nion_1d_acg;
    P21DevTable2Df nion_2d_acg;
    P21DevTable2Df nion_2d_mcg;
    P21DevTable1Df sfrd_1d_acg;
    P21DevTable2Df sfrd_2d_acg;
    P21DevTable2Df sfrd_2d_mcg;
    P21DevTable1Df xray_1d_acg;
    P21DevTable2Df xray_2d_acg;
    P21DevTable2Df xray_2d_mcg;
    P21DevTable1Dd nhalo_1d;
    P21DevTable1Dd fcoll_1d;
};

__global__ void emissivity_mturn_kernel(size_t n, P21MturnParams mp, float vcb_const,
                                        int below_zheatmax, int vcb_flucts, const float *lowres_vcb,
                                        const float *j21_lw, const float *gamma12,
                                        const float *z_reion, float *mturn_acg_out,
                                        float *mturn_mcg_out, double *blk_acg, double *blk_mcg) {
    __shared__ double s_acg[kBlockThreads];
    __shared__ double s_mcg[kBlockThreads];

    size_t tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    double sum_acg = 0.0;
    double sum_mcg = 0.0;

    for (size_t i = idx; i < n; i += stride) {
        float j21_val = 0.0f;
        float g12_val = 0.0f;
        float zre_val = 0.0f;
        float curr_vcb = vcb_const;

        if (vcb_flucts && mp.use_mcgs) {
            curr_vcb = lowres_vcb[i];
        }
        if (below_zheatmax) {
            if (mp.use_mcgs) {
                j21_val = j21_lw[i];
            }
            if (mp.use_reion_feedback) {
                g12_val = gamma12[i];
                zre_val = z_reion[i];
            }
        }

        float mt_acg = 0.0f, mt_mcg = 0.0f;
        p21_compute_mturns_inhomogeneous(mp, j21_val, curr_vcb, g12_val, zre_val, &mt_acg, &mt_mcg);

        if (mp.use_reion_feedback) {
            double l10 = log10((double)mt_acg);
            mturn_acg_out[i] = (float)l10;
            sum_acg += l10;
        }
        if (mp.use_mcgs) {
            double l10 = log10((double)mt_mcg);
            mturn_mcg_out[i] = (float)l10;
            sum_mcg += l10;
        }
    }

    s_acg[tid] = sum_acg;
    s_mcg[tid] = sum_mcg;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            s_acg[tid] += s_acg[tid + s];
            s_mcg[tid] += s_mcg[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        blk_acg[blockIdx.x] = s_acg[0];
        blk_mcg[blockIdx.x] = s_mcg[0];
    }
}

__global__ void minmax_density_block_kernel(const float *dens, size_t n, double *blk_min,
                                            double *blk_max) {
    __shared__ double s_min[kBlockThreads];
    __shared__ double s_max[kBlockThreads];

    size_t tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    double lo = 0.0;
    double hi = 0.0;
    for (size_t i = idx; i < n; i += stride) {
        double v = (double)dens[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }

    s_min[tid] = lo;
    s_max[tid] = hi;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            if (s_min[tid + s] < s_min[tid]) s_min[tid] = s_min[tid + s];
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        blk_min[blockIdx.x] = s_min[0];
        blk_max[blockIdx.x] = s_max[0];
    }
}

__global__ void finish_pair_sum_kernel(const double *b0, const double *b1, int nblocks,
                                         double *out2) {
    __shared__ double s0[kBlockThreads];
    __shared__ double s1[kBlockThreads];
    size_t tid = threadIdx.x;
    double a0 = 0.0, a1 = 0.0;
    for (int i = (int)tid; i < nblocks; i += kBlockThreads) {
        a0 += b0[i];
        a1 += b1[i];
    }
    s0[tid] = a0;
    s1[tid] = a1;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            s0[tid] += s0[tid + s];
            s1[tid] += s1[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out2[0] = s0[0];
        out2[1] = s1[0];
    }
}

__global__ void finish_minmax_kernel(const double *bmin, const double *bmax, int nblocks,
                                     double *out2) {
    __shared__ double smin[kBlockThreads];
    __shared__ double smax[kBlockThreads];
    size_t tid = threadIdx.x;
    double lo = 0.0, hi = 0.0;
    for (int i = (int)tid; i < nblocks; i += kBlockThreads) {
        if (bmin[i] < lo) lo = bmin[i];
        if (bmax[i] > hi) hi = bmax[i];
    }
    smin[tid] = lo;
    smax[tid] = hi;
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            if (smin[tid + s] < smin[tid]) smin[tid] = smin[tid + s];
            if (smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out2[0] = smin[0];
        out2[1] = smax[0];
    }
}

__global__ void eulerian_emissivity_eval_kernel(
    size_t n, EmissivityEvalParams p, const float *dens, const float *mturn_acg,
    const float *mturn_mcg, float *out_nion, float *out_sfrd_acg, float *out_sfrd_mcg,
    float *out_xray, float *out_halo_num, float *out_halo_mass, float *out_star_acg,
    float *out_star_mcg, double *blk_sums /* 9 * nblocks */) {
    __shared__ double sh[9][kBlockThreads];

    size_t tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    double acc[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double l10_acg_hom = log10(p.mturn_acg_homogeneous);

    for (size_t i = idx; i < n; i += stride) {
        double d = (double)dens[i];
        double l10_acg = p.use_reion_feedback ? (double)mturn_acg[i] : l10_acg_hom;
        double l10_mcg = p.use_mcgs ? (double)mturn_mcg[i] : 0.0;
        bool allow_mcg = p.use_mcgs && (pow(10.0, l10_acg) <= p.atomic_cooling_threshold * 1.001);

        /* n_ion (unscaled by 1+d on Eulerian grid, matching EmissivityFields.c:380) */
        double n_ion = (p.use_reion_feedback && p.mass_dep_zeta)
                           ? exp(p21_eval_rg2d_f(d, l10_acg, p.nion_2d_acg))
                           : exp(p21_eval_rg1d_f(d, p.nion_1d_acg));
        if (allow_mcg) {
            n_ion += exp(p21_eval_rg2d_f(d, l10_mcg, p.nion_2d_mcg));
        }

        double sfrd_acg = 0.0, sfrd_mcg = 0.0;
        if (p.use_ts_fluct || (p.extra_fields && p.mass_dep_zeta)) {
            sfrd_acg = (p.use_reion_feedback && p.mass_dep_zeta)
                           ? exp(p21_eval_rg2d_f(d, l10_acg, p.sfrd_2d_acg))
                           : exp(p21_eval_rg1d_f(d, p.sfrd_1d_acg));
            if (allow_mcg) {
                sfrd_mcg = exp(p21_eval_rg2d_f(d, l10_mcg, p.sfrd_2d_mcg));
            }
        }

        double xray = 0.0;
        if (p.use_ts_fluct && p.use_metallicity) {
            xray = p.use_reion_feedback ? exp(p21_eval_rg2d_f(d, l10_acg, p.xray_2d_acg))
                                        : exp(p21_eval_rg1d_f(d, p.xray_1d_acg));
            if (allow_mcg) {
                xray += exp(p21_eval_rg2d_f(d, l10_mcg, p.xray_2d_mcg));
            }
        }

        double halo_num = 0.0, halo_mass = 0.0, star_acg = 0.0, star_mcg = 0.0;
        if (p.extra_fields) {
            halo_num = p21_eval_rg1d_d(d, p.nhalo_1d) * p.cell_volume;
            halo_mass = p21_eval_rg1d_d(d, p.fcoll_1d) * p.rhocrit_omm;
            if (p.mass_dep_zeta) {
                star_acg = sfrd_acg * p.sfr_timescale;
            } else {
                star_acg = n_ion * p.rhocrit_omb * p.fstar_10_acg / p.hii_eff_factor;
            }
            if (p.use_mcgs) {
                star_mcg = sfrd_mcg * p.sfr_timescale;
            }
        }

        /* Eulerian (1 + dens) factor for all fields except n_ion */
        double opd = 1.0 + d;
        float f_nion = (float)n_ion;
        out_nion[i] = f_nion;
        acc[6] += (double)f_nion;

        float factor = (1.0 + d > p.fract_float_err) ? (float)opd : 1.0f;

        if (p.use_ts_fluct) {
            float f_sfr_a = (float)(sfrd_acg * opd);
            out_sfrd_acg[i] = f_sfr_a;
            acc[4] += (double)f_sfr_a / (double)factor;

            float f_sfr_m = 0.0f;
            if (p.use_mcgs) {
                f_sfr_m = (float)(sfrd_mcg * opd);
                out_sfrd_mcg[i] = f_sfr_m;
                acc[5] += (double)f_sfr_m / (double)factor;
            }

            float f_xr;
            if (p.use_metallicity) {
                f_xr = (float)(xray * opd);
            } else {
                f_xr = (float)(p.lx_over_sfr_acg * (double)f_sfr_a +
                               (p.use_mcgs ? p.lx_over_sfr_mcg * (double)f_sfr_m : 0.0));
            }
            out_xray[i] = f_xr;
            acc[7] += (double)f_xr / (double)factor;
        }

        if (p.extra_fields) {
            float f_hn = (float)(halo_num * opd);
            float f_hm = (float)(halo_mass * opd);
            float f_sa = (float)(star_acg * opd);
            out_halo_num[i] = f_hn;
            out_halo_mass[i] = f_hm;
            out_star_acg[i] = f_sa;
            acc[0] += (double)f_hn / (double)factor;
            acc[1] += (double)f_hm / (double)factor;
            acc[2] += (double)f_sa / (double)factor;
            if (p.use_mcgs) {
                float f_sm = (float)(star_mcg * opd);
                out_star_mcg[i] = f_sm;
                acc[3] += (double)f_sm / (double)factor;
            }
        }
    }

    for (int k = 0; k < 9; ++k) {
        sh[k][tid] = acc[k];
    }
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            for (int k = 0; k < 9; ++k) {
                sh[k][tid] += sh[k][tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int k = 0; k < 9; ++k) {
            blk_sums[(size_t)k * gridDim.x + blockIdx.x] = sh[k][0];
        }
    }
}

__global__ void finish_9way_sums_kernel(const double *blk_sums, int nblocks, double *out9) {
    __shared__ double sh[kBlockThreads];
    int k = blockIdx.x;
    size_t tid = threadIdx.x;
    const double *row = blk_sums + (size_t)k * nblocks;
    double acc = 0.0;
    for (int i = (int)tid; i < nblocks; i += kBlockThreads) {
        acc += row[i];
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
        out9[k] = sh[0];
    }
}

__global__ void emissivity_mean_fix_kernel(
    size_t n, int use_ts_fluct, int use_mcgs, int extra_fields, double s_nion, double s_sfr_acg,
    double s_sfr_mcg, double s_xray, double s_hn, double s_hm, double s_sa, double s_sm,
    float *out_nion, float *out_sfrd_acg, float *out_sfrd_mcg, float *out_xray,
    float *out_halo_num, float *out_halo_mass, float *out_star_acg, float *out_star_mcg) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    out_nion[idx] = (float)((double)out_nion[idx] * s_nion);
    if (use_ts_fluct) {
        out_sfrd_acg[idx] = (float)((double)out_sfrd_acg[idx] * s_sfr_acg);
        out_xray[idx] = (float)((double)out_xray[idx] * s_xray);
        if (use_mcgs) {
            out_sfrd_mcg[idx] = (float)((double)out_sfrd_mcg[idx] * s_sfr_mcg);
        }
    }
    if (extra_fields) {
        out_halo_num[idx] = (float)((double)out_halo_num[idx] * s_hn);
        out_halo_mass[idx] = (float)((double)out_halo_mass[idx] * s_hm);
        out_star_acg[idx] = (float)((double)out_star_acg[idx] * s_sa);
        if (use_mcgs) {
            out_star_mcg[idx] = (float)((double)out_star_mcg[idx] * s_sm);
        }
    }
}

__global__ void lag_deposit_halos_kernel(
    size_t n_halos, P21HaloScalingParams p, double boxlen, double boxlen_z, double vdf,
    double vdf2, int vel_nx, int vel_ny, int vel_nz, int out_nx, int out_ny, int out_nz,
    int use_reion_fb, int uses_recomb, int extra_fields, int is_2lpt, const float *vx,
    const float *vy, const float *vz, const float *vx2, const float *vy2, const float *vz2,
    const float *mturn_acg_grid, const float *mturn_mcg_grid, const float *halo_masses,
    const float *halo_coords, const float *star_rng, const float *sfr_rng, const float *xray_rng,
    double *acc_nion, double *acc_sfrd_acg, double *acc_sfrd_mcg, double *acc_xray,
    double *acc_fesc_wsfr, double *acc_hn, double *acc_hm, double *acc_sa, double *acc_sm) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_halos) {
        return;
    }

    double hmass = (double)halo_masses[idx];
    if (hmass == 0.0) {
        return;
    }

    const double cell_size_inv_v = (double)vel_nx / boxlen;
    const int vel_dim[3] = {vel_nx, vel_ny, vel_nz};
    double pos[3] = {
        (double)halo_coords[3 * idx + 0],
        (double)halo_coords[3 * idx + 1],
        (double)halo_coords[3 * idx + 2],
    };
    int ipos[3];
    p21_pos_to_index(pos, cell_size_inv_v, ipos);
    for (int axis = 0; axis < 3; ++axis) {
        ipos[axis] = p21_wrap_coord(ipos[axis], vel_dim[axis]);
    }
    size_t vel_idx = p21_grid_idx(ipos[0], ipos[1], ipos[2], vel_ny, vel_nz);

    pos[0] += (double)vx[vel_idx] * vdf;
    pos[1] += (double)vy[vel_idx] * vdf;
    pos[2] += (double)vz[vel_idx] * vdf;
    if (is_2lpt) {
        pos[0] -= (double)vx2[vel_idx] * vdf2;
        pos[1] -= (double)vy2[vel_idx] * vdf2;
        pos[2] -= (double)vz2[vel_idx] * vdf2;
    }

    pos[0] = pos[0] * (double)out_nx / boxlen;
    pos[1] = pos[1] * (double)out_ny / boxlen;
    pos[2] = pos[2] * (double)out_nz / boxlen_z;

    double M_turn_acg = p.mturn_acg_homogeneous;
    double M_turn_mcg = 0.0;
    if (use_reion_fb) {
        M_turn_acg = pow(10.0, p21_cic_read_float(mturn_acg_grid, pos, out_nx, out_ny, out_nz));
    }
    if (p.use_mcgs) {
        M_turn_mcg = pow(10.0, p21_cic_read_float(mturn_mcg_grid, pos, out_nx, out_ny, out_nz));
    }

    P21HaloPropsOut props{};
    p21_set_halo_properties_dev(hmass, M_turn_acg, M_turn_mcg, p, (double)star_rng[idx],
                                (double)sfr_rng[idx], (double)xray_rng[idx], &props);

    P21CicStencils st;
    p21_cic_compute(pos, out_nx, out_ny, out_nz, &st);

    p21_cic_deposit_stenciled(acc_nion, st, props.n_ion);
    if (p.use_ts_fluct) {
        p21_cic_deposit_stenciled(acc_sfrd_acg, st, props.sfr_acg);
        p21_cic_deposit_stenciled(acc_xray, st, props.xray_luminosity);
        if (p.use_mcgs) {
            p21_cic_deposit_stenciled(acc_sfrd_mcg, st, props.sfr_mcg);
        }
    }
    if (uses_recomb) {
        p21_cic_deposit_stenciled(acc_fesc_wsfr, st, props.fesc_weighted_sfr);
    }
    if (extra_fields) {
        p21_cic_deposit_stenciled(acc_hn, st, 1.0);
        p21_cic_deposit_stenciled(acc_hm, st, props.halo_mass);
        p21_cic_deposit_stenciled(acc_sa, st, props.stellar_mass_acg);
        if (p.use_mcgs) {
            p21_cic_deposit_stenciled(acc_sm, st, props.stellar_mass_mcg);
        }
    }
}

__global__ void lag_scale_halos_kernel(size_t n, double cell_vol_inv, int use_ts_fluct,
                                         int use_mcgs, int uses_recomb, int extra_fields,
                                         double *acc_nion, double *acc_sfrd_acg,
                                         double *acc_sfrd_mcg, double *acc_xray,
                                         double *acc_fesc_wsfr, double *acc_hm, double *acc_sa,
                                         double *acc_sm) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    acc_nion[idx] *= cell_vol_inv;
    if (use_ts_fluct) {
        acc_sfrd_acg[idx] *= cell_vol_inv;
        acc_xray[idx] *= cell_vol_inv;
        if (use_mcgs) {
            acc_sfrd_mcg[idx] *= cell_vol_inv;
        }
    }
    if (uses_recomb) {
        acc_fesc_wsfr[idx] *= cell_vol_inv;
    }
    if (extra_fields) {
        acc_hm[idx] *= cell_vol_inv;
        acc_sa[idx] *= cell_vol_inv;
        if (use_mcgs) {
            acc_sm[idx] *= cell_vol_inv;
        }
    }
}

__global__ void lag_deposit_integral_kernel(
    size_t dens_n, int d_nx, int d_ny, int d_nz, int o_nx, int o_ny, int o_nz, double dim_ratio_out,
    double vol_ratio_out, double vdf0, double vdf1, double vdf2, double vdf2_0, double vdf2_1,
    double vdf2_2, int is_2lpt, EmissivityEvalParams p, const float *d_dens, const float *vx,
    const float *vy, const float *vz, const float *vx2, const float *vy2, const float *vz2,
    const float *mturn_acg_grid, const float *mturn_mcg_grid, double *acc_nion,
    double *acc_sfrd_acg, double *acc_sfrd_mcg, double *acc_xray, double *acc_hn, double *acc_hm,
    double *acc_sa, double *acc_sm) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dens_n) {
        return;
    }

    size_t yz = (size_t)d_ny * (size_t)d_nz;
    int ix = (int)(idx / yz);
    size_t rem = idx - (size_t)ix * yz;
    int iy = (int)(rem / (size_t)d_nz);
    int iz = (int)(rem - (size_t)iy * (size_t)d_nz);

    double d = (double)d_dens[idx] * p.growth_factor;

    double pos[3] = {
        ((double)ix + (double)vx[idx] * vdf0) * dim_ratio_out,
        ((double)iy + (double)vy[idx] * vdf1) * dim_ratio_out,
        ((double)iz + (double)vz[idx] * vdf2) * dim_ratio_out,
    };
    if (is_2lpt) {
        pos[0] -= (double)vx2[idx] * vdf2_0 * dim_ratio_out;
        pos[1] -= (double)vy2[idx] * vdf2_1 * dim_ratio_out;
        pos[2] -= (double)vz2[idx] * vdf2_2 * dim_ratio_out;
    }

    size_t mturn_idx = idx;
    if (d_nx != o_nx || d_ny != o_ny || d_nz != o_nz) {
        int mx = p21_wrap_coord(p21_resample_coord(ix, dim_ratio_out), o_nx);
        int my = p21_wrap_coord(p21_resample_coord(iy, dim_ratio_out), o_ny);
        int mz = p21_wrap_coord(p21_resample_coord(iz, dim_ratio_out), o_nz);
        mturn_idx = p21_grid_idx(mx, my, mz, o_ny, o_nz);
    }

    double l10_acg =
        p.use_reion_feedback ? (double)mturn_acg_grid[mturn_idx] : log10(p.mturn_acg_homogeneous);
    double l10_mcg = p.use_mcgs ? (double)mturn_mcg_grid[mturn_idx] : 0.0;
    bool allow_mcg = p.use_mcgs && (pow(10.0, l10_acg) <= p.atomic_cooling_threshold * 1.001);

    double n_ion = (p.use_reion_feedback && p.mass_dep_zeta)
                       ? exp(p21_eval_rg2d_f(d, l10_acg, p.nion_2d_acg))
                       : exp(p21_eval_rg1d_f(d, p.nion_1d_acg));
    if (allow_mcg) {
        n_ion += exp(p21_eval_rg2d_f(d, l10_mcg, p.nion_2d_mcg));
    }

    double sfrd_acg = 0.0, sfrd_mcg = 0.0;
    if (p.use_ts_fluct || (p.extra_fields && p.mass_dep_zeta)) {
        sfrd_acg = (p.use_reion_feedback && p.mass_dep_zeta)
                       ? exp(p21_eval_rg2d_f(d, l10_acg, p.sfrd_2d_acg))
                       : exp(p21_eval_rg1d_f(d, p.sfrd_1d_acg));
        if (allow_mcg) {
            sfrd_mcg = exp(p21_eval_rg2d_f(d, l10_mcg, p.sfrd_2d_mcg));
        }
    }

    double xray = 0.0;
    if (p.use_ts_fluct && p.use_metallicity) {
        xray = p.use_reion_feedback ? exp(p21_eval_rg2d_f(d, l10_acg, p.xray_2d_acg))
                                    : exp(p21_eval_rg1d_f(d, p.xray_1d_acg));
        if (allow_mcg) {
            xray += exp(p21_eval_rg2d_f(d, l10_mcg, p.xray_2d_mcg));
        }
    }

    double halo_num = 0.0, halo_mass = 0.0, star_acg = 0.0, star_mcg = 0.0;
    if (p.extra_fields) {
        halo_num = p21_eval_rg1d_d(d, p.nhalo_1d) * p.cell_volume;
        halo_mass = p21_eval_rg1d_d(d, p.fcoll_1d) * p.rhocrit_omm;
        if (p.mass_dep_zeta) {
            star_acg = sfrd_acg * p.sfr_timescale;
        } else {
            star_acg = n_ion * p.rhocrit_omb * p.fstar_10_acg / p.hii_eff_factor;
        }
        if (p.use_mcgs) {
            star_mcg = sfrd_mcg * p.sfr_timescale;
        }
    }

    P21CicStencils st;
    p21_cic_compute(pos, o_nx, o_ny, o_nz, &st);

    p21_cic_deposit_stenciled(acc_nion, st, n_ion * vol_ratio_out);
    if (p.use_ts_fluct) {
        p21_cic_deposit_stenciled(acc_sfrd_acg, st, sfrd_acg * vol_ratio_out);
        if (p.use_mcgs) {
            p21_cic_deposit_stenciled(acc_sfrd_mcg, st, sfrd_mcg * vol_ratio_out);
        }
        if (p.use_metallicity) {
            p21_cic_deposit_stenciled(acc_xray, st, xray * vol_ratio_out);
        }
    }
    if (p.extra_fields) {
        p21_cic_deposit_stenciled(acc_hn, st, halo_num);
        p21_cic_deposit_stenciled(acc_hm, st, halo_mass * vol_ratio_out);
        p21_cic_deposit_stenciled(acc_sa, st, star_acg * vol_ratio_out);
        if (p.use_mcgs) {
            p21_cic_deposit_stenciled(acc_sm, st, star_mcg * vol_ratio_out);
        }
    }
}

__global__ void lag_post_integral_kernel(size_t n, int use_ts_fluct, int use_mcgs,
                                           int use_metallicity, int uses_recomb, int mass_dep,
                                           double lx_acg, double lx_mcg, double rhocrit_omb,
                                           double sfr_timescale, double fstar_10_acg,
                                           double hii_eff_factor, const double *acc_nion,
                                           const double *acc_sfrd_acg, const double *acc_sfrd_mcg,
                                           double *acc_xray, double *acc_fesc_wsfr) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    if (use_ts_fluct && !use_metallicity) {
        double val = lx_acg * acc_sfrd_acg[idx];
        if (use_mcgs) {
            val += lx_mcg * acc_sfrd_mcg[idx];
        }
        acc_xray[idx] = val;
    }
    if (uses_recomb) {
        if (mass_dep) {
            acc_fesc_wsfr[idx] = acc_nion[idx] * rhocrit_omb / sfr_timescale;
        } else {
            acc_fesc_wsfr[idx] = acc_sfrd_acg[idx] / fstar_10_acg * hii_eff_factor;
        }
    }
}

__global__ void lag_9way_sums_kernel(size_t n, int use_ts_fluct, int use_mcgs, int uses_recomb,
                                       int extra_fields, const double *acc_nion,
                                       const double *acc_sfrd_acg, const double *acc_sfrd_mcg,
                                       const double *acc_xray, const double *acc_fesc_wsfr,
                                       const double *acc_hn, const double *acc_hm,
                                       const double *acc_sa, const double *acc_sm,
                                       double *blk_sums) {
    __shared__ double sh[9][kBlockThreads];

    size_t tid = threadIdx.x;
    size_t idx = (size_t)blockIdx.x * blockDim.x + tid;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    double acc[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (size_t i = idx; i < n; i += stride) {
        acc[6] += acc_nion[i];
        if (use_ts_fluct) {
            acc[4] += acc_sfrd_acg[i];
            acc[7] += acc_xray[i];
            if (use_mcgs) {
                acc[5] += acc_sfrd_mcg[i];
            }
        }
        if (uses_recomb) {
            acc[8] += acc_fesc_wsfr[i];
        }
        if (extra_fields) {
            acc[0] += acc_hn[i];
            acc[1] += acc_hm[i];
            acc[2] += acc_sa[i];
            if (use_mcgs) {
                acc[3] += acc_sm[i];
            }
        }
    }

    for (int k = 0; k < 9; ++k) {
        sh[k][tid] = acc[k];
    }
    __syncthreads();
    for (int s = kBlockThreads / 2; s > 0; s >>= 1) {
        if (tid < (size_t)s) {
            for (int k = 0; k < 9; ++k) {
                sh[k][tid] += sh[k][tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int k = 0; k < 9; ++k) {
            blk_sums[(size_t)k * gridDim.x + blockIdx.x] = sh[k][0];
        }
    }
}

__global__ void lag_mean_fix_scale_kernel(
    size_t n, int use_ts_fluct, int use_mcgs, int uses_recomb, int extra_fields, double s_nion,
    double s_sfr_a, double s_sfr_m, double s_xray, double s_fesc, double s_hn, double s_hm,
    double s_sa, double s_sm, double *acc_nion, double *acc_sfrd_acg, double *acc_sfrd_mcg,
    double *acc_xray, double *acc_fesc_wsfr, double *acc_hn, double *acc_hm, double *acc_sa,
    double *acc_sm) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    acc_nion[idx] *= s_nion;
    if (use_ts_fluct) {
        acc_sfrd_acg[idx] *= s_sfr_a;
        acc_xray[idx] *= s_xray;
        if (use_mcgs) {
            acc_sfrd_mcg[idx] *= s_sfr_m;
        }
    }
    if (uses_recomb) {
        acc_fesc_wsfr[idx] *= s_fesc;
    }
    if (extra_fields) {
        acc_hn[idx] *= s_hn;
        acc_hm[idx] *= s_hm;
        acc_sa[idx] *= s_sa;
        if (use_mcgs) {
            acc_sm[idx] *= s_sm;
        }
    }
}

__global__ void lag_cast_f64_to_f32_kernel(size_t n, int use_ts_fluct, int use_mcgs,
                                            int uses_recomb, int extra_fields,
                                            const double *acc_nion, const double *acc_sfrd_acg,
                                            const double *acc_sfrd_mcg, const double *acc_xray,
                                            const double *acc_fesc_wsfr, const double *acc_hn,
                                            const double *acc_hm, const double *acc_sa,
                                            const double *acc_sm, float *out_nion,
                                            float *out_sfrd_acg, float *out_sfrd_mcg,
                                            float *out_xray, float *out_fesc_wsfr, float *out_hn,
                                            float *out_hm, float *out_sa, float *out_sm) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    out_nion[idx] = (float)acc_nion[idx];
    if (use_ts_fluct) {
        out_sfrd_acg[idx] = (float)acc_sfrd_acg[idx];
        out_xray[idx] = (float)acc_xray[idx];
        if (use_mcgs) {
            out_sfrd_mcg[idx] = (float)acc_sfrd_mcg[idx];
        }
    }
    if (uses_recomb) {
        out_fesc_wsfr[idx] = (float)acc_fesc_wsfr[idx];
    }
    if (extra_fields) {
        out_hn[idx] = (float)acc_hn[idx];
        out_hm[idx] = (float)acc_hm[idx];
        out_sa[idx] = (float)acc_sa[idx];
        if (use_mcgs) {
            out_sm[idx] = (float)acc_sm[idx];
        }
    }
}

bool compute_min_max_dens(const float *d_dens, size_t total, cudaStream_t stream,
                          double *min_dens_out, double *max_dens_out) {
    int blocks = (int)((total + kBlockThreads - 1) / kBlockThreads);
    if (blocks > 1024) blocks = 1024;
    double *d_blk0 = (double *)p21_cuda_pool_alloc((size_t)blocks * sizeof(double), stream);
    double *d_blk1 = (double *)p21_cuda_pool_alloc((size_t)blocks * sizeof(double), stream);
    double *d_out2 = (double *)p21_cuda_pool_alloc(2 * sizeof(double), stream);
    if (!d_blk0 || !d_blk1 || !d_out2) {
        p21_cuda_pool_free(d_blk0, stream);
        p21_cuda_pool_free(d_blk1, stream);
        p21_cuda_pool_free(d_out2, stream);
        return false;
    }

    minmax_density_block_kernel<<<blocks, kBlockThreads, 0, stream>>>(d_dens, total, d_blk0,
                                                                      d_blk1);
    bool ok = P21_CUDA_OK(cudaGetLastError());
    if (ok) {
        finish_minmax_kernel<<<1, kBlockThreads, 0, stream>>>(d_blk0, d_blk1, blocks, d_out2);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    double h_mm[2] = {0.0, 0.0};
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(h_mm, d_out2, 2 * sizeof(double),
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }
    p21_cuda_pool_free(d_blk0, stream);
    p21_cuda_pool_free(d_blk1, stream);
    p21_cuda_pool_free(d_out2, stream);

    if (ok) {
        *min_dens_out = h_mm[0];
        *max_dens_out = h_mm[1];
    }
    return ok;
}

bool upload_cond_emissivity_params(const ScalingConstants *ev_consts, cudaStream_t stream,
                                   EmissivityEvalParams *p) {
    const int use_ts_fluct = astro_options_global->USE_TS_FLUCT ? 1 : 0;
    const int use_mcgs = astro_options_global->USE_MCGS ? 1 : 0;
    const int use_metallicity = astro_options_global->USE_METALLICITY ? 1 : 0;
    const int use_reion_fb = astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK ? 1 : 0;
    const int mass_dep =
        source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL) ? 1 : 0;
    const int extra_fields = config_settings.EXTRA_EMISSIVITY_FIELDS ? 1 : 0;

    P21ConditionalTables htbl{};
    p21_get_conditional_tables(&htbl);

    memset(p, 0, sizeof(*p));
    p->redshift = ev_consts->redshift;
    p->growth_factor = ev_consts->growth_factor;
    p->mturn_acg_homogeneous = ev_consts->mturn_acg_homogeneous;
    p->atomic_cooling_threshold = ev_consts->atomic_cooling_threshold;
    p->sfr_timescale = ev_consts->sfr_timescale;
    p->lx_over_sfr_acg = ev_consts->lx_over_sfr_acg;
    p->lx_over_sfr_mcg = ev_consts->lx_over_sfr_mcg;
    p->fstar_10_acg = ev_consts->fstar_10_acg;
    p->hii_eff_factor = astro_params_global->HII_EFF_FACTOR;
    p->rhocrit_omm = RHOcrit * cosmo_params_global->OMm;
    p->rhocrit_omb = RHOcrit * cosmo_params_global->OMb;
    p->cell_volume = ev_consts->M_cell / p->rhocrit_omm;
    p->fract_float_err = FRACT_FLOAT_ERR;
    p->use_ts_fluct = use_ts_fluct;
    p->use_mcgs = use_mcgs;
    p->use_metallicity = use_metallicity;
    p->use_reion_feedback = use_reion_fb;
    p->mass_dep_zeta = mass_dep;
    p->extra_fields = extra_fields;
    p->fix_mean = ev_consts->fix_mean ? 1 : 0;

    bool ok = true;
    if (use_reion_fb && mass_dep) {
        ok = ok && p21_upload_rg2d_f(htbl.nion_2d_acg, &p->nion_2d_acg, stream);
    } else {
        ok = ok && p21_upload_rg1d_f(htbl.nion_1d_acg, &p->nion_1d_acg, stream);
    }
    if (use_mcgs) {
        ok = ok && p21_upload_rg2d_f(htbl.nion_2d_mcg, &p->nion_2d_mcg, stream);
    }
    if (use_ts_fluct || (extra_fields && mass_dep)) {
        if (use_reion_fb && mass_dep) {
            ok = ok && p21_upload_rg2d_f(htbl.sfrd_2d_acg, &p->sfrd_2d_acg, stream);
        } else {
            ok = ok && p21_upload_rg1d_f(htbl.sfrd_1d_acg, &p->sfrd_1d_acg, stream);
        }
        if (use_mcgs) {
            ok = ok && p21_upload_rg2d_f(htbl.sfrd_2d_mcg, &p->sfrd_2d_mcg, stream);
        }
    }
    if (use_ts_fluct && use_metallicity) {
        if (use_reion_fb) {
            ok = ok && p21_upload_rg2d_f(htbl.xray_2d_acg, &p->xray_2d_acg, stream);
        } else {
            ok = ok && p21_upload_rg1d_f(htbl.xray_1d_acg, &p->xray_1d_acg, stream);
        }
        if (use_mcgs) {
            ok = ok && p21_upload_rg2d_f(htbl.xray_2d_mcg, &p->xray_2d_mcg, stream);
        }
    }
    if (extra_fields) {
        ok = ok && p21_upload_rg1d_d(htbl.nhalo_1d, &p->nhalo_1d, stream) &&
             p21_upload_rg1d_d(htbl.fcoll_1d, &p->fcoll_1d, stream);
    }
    return ok;
}

void free_cond_emissivity_params(EmissivityEvalParams *p, cudaStream_t stream) {
    p21_free_dev_table(&p->nion_1d_acg, stream);
    p21_free_dev_table(&p->nion_2d_acg, stream);
    p21_free_dev_table(&p->nion_2d_mcg, stream);
    p21_free_dev_table(&p->sfrd_1d_acg, stream);
    p21_free_dev_table(&p->sfrd_2d_acg, stream);
    p21_free_dev_table(&p->sfrd_2d_mcg, stream);
    p21_free_dev_table(&p->xray_1d_acg, stream);
    p21_free_dev_table(&p->xray_2d_acg, stream);
    p21_free_dev_table(&p->xray_2d_mcg, stream);
    p21_free_dev_table(&p->nhalo_1d, stream);
    p21_free_dev_table(&p->fcoll_1d, stream);
}

void free_lag_ctx_buffers(P21CudaLagEmissCtx *ctx) {
    if (!ctx) return;
    cudaStream_t stream = ctx->stream;
    p21_cuda_pool_free(ctx->d_mturn_acg, stream);
    p21_cuda_pool_free(ctx->d_mturn_mcg, stream);
    p21_cuda_pool_free(ctx->d_acc_nion, stream);
    p21_cuda_pool_free(ctx->d_acc_sfrd_acg, stream);
    p21_cuda_pool_free(ctx->d_acc_sfrd_mcg, stream);
    p21_cuda_pool_free(ctx->d_acc_xray, stream);
    p21_cuda_pool_free(ctx->d_acc_fesc_wsfr, stream);
    p21_cuda_pool_free(ctx->d_acc_hn, stream);
    p21_cuda_pool_free(ctx->d_acc_hm, stream);
    p21_cuda_pool_free(ctx->d_acc_sa, stream);
    p21_cuda_pool_free(ctx->d_acc_sm, stream);
    ctx->d_mturn_acg = nullptr;
    ctx->d_mturn_mcg = nullptr;
    ctx->d_acc_nion = nullptr;
    ctx->d_acc_sfrd_acg = nullptr;
    ctx->d_acc_sfrd_mcg = nullptr;
    ctx->d_acc_xray = nullptr;
    ctx->d_acc_fesc_wsfr = nullptr;
    ctx->d_acc_hn = nullptr;
    ctx->d_acc_hm = nullptr;
    ctx->d_acc_sa = nullptr;
    ctx->d_acc_sm = nullptr;
}

}  // namespace

bool p21_cuda_build_mturn_grids(const ScalingConstants *consts, InitialConditions *ini_boxes,
                                TsBox *previous_spin_temp, IonizedBox *previous_ionize_box,
                                cudaStream_t stream, float **d_mturn_acg_out,
                                float **d_mturn_mcg_out, double log10_mturn_averages_out[2]) {
    if (!consts || !d_mturn_acg_out || !d_mturn_mcg_out || !log10_mturn_averages_out) {
        return false;
    }
    *d_mturn_acg_out = nullptr;
    *d_mturn_mcg_out = nullptr;

    const int use_reion_fb = astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK ? 1 : 0;
    const int use_mcgs = astro_options_global->USE_MCGS ? 1 : 0;
    if (!use_reion_fb && !use_mcgs) {
        log10_mturn_averages_out[0] = log10(consts->mturn_acg_homogeneous);
        log10_mturn_averages_out[1] = 0.0;
        return true;
    }

    size_t total = (size_t)simulation_options_global->HII_DIM *
                   (size_t)simulation_options_global->HII_DIM * (size_t)HII_D_PARA;
    size_t fbytes = total * sizeof(float);
    int below_zheat = (consts->redshift < simulation_options_global->Z_HEAT_MAX) ? 1 : 0;
    int vcb_flucts = (matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS) ? 1 : 0;

    float *d_mturn_acg = use_reion_fb ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    float *d_mturn_mcg = use_mcgs ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    int blocks = (int)((total + kBlockThreads - 1) / kBlockThreads);
    if (blocks > 1024) blocks = 1024;
    double *d_blk0 = (double *)p21_cuda_pool_alloc((size_t)blocks * sizeof(double), stream);
    double *d_blk1 = (double *)p21_cuda_pool_alloc((size_t)blocks * sizeof(double), stream);
    double *d_out2 = (double *)p21_cuda_pool_alloc(2 * sizeof(double), stream);

    if ((use_reion_fb && !d_mturn_acg) || (use_mcgs && !d_mturn_mcg) || !d_blk0 || !d_blk1 ||
        !d_out2) {
        p21_cuda_pool_free(d_mturn_acg, stream);
        p21_cuda_pool_free(d_mturn_mcg, stream);
        p21_cuda_pool_free(d_blk0, stream);
        p21_cuda_pool_free(d_blk1, stream);
        p21_cuda_pool_free(d_out2, stream);
        return false;
    }

    bool ok = true;
    const float *d_vcb = nullptr, *d_j21 = nullptr, *d_g12 = nullptr, *d_zre = nullptr;
    if (vcb_flucts && use_mcgs) {
        if (!ini_boxes || !ini_boxes->lowres_vcb) ok = false;
        else d_vcb = (const float *)p21_cuda_registry_get_or_upload(ini_boxes->lowres_vcb,
                                                                    fbytes, stream);
        if (!d_vcb) ok = false;
    }
    if (ok && below_zheat) {
        if (use_mcgs) {
            if (!previous_spin_temp || !previous_spin_temp->J_21_LW) ok = false;
            else d_j21 = (const float *)p21_cuda_registry_get_or_upload(
                     previous_spin_temp->J_21_LW, fbytes, stream);
            if (!d_j21) ok = false;
        }
        if (ok && use_reion_fb) {
            if (!previous_ionize_box || !previous_ionize_box->ionisation_rate_G12 ||
                !previous_ionize_box->z_reion) {
                ok = false;
            } else {
                d_g12 = (const float *)p21_cuda_registry_get_or_upload(
                    previous_ionize_box->ionisation_rate_G12, fbytes, stream);
                d_zre = (const float *)p21_cuda_registry_get_or_upload(
                    previous_ionize_box->z_reion, fbytes, stream);
                if (!d_g12 || !d_zre) ok = false;
            }
        }
    }
    if (ok) {
        P21MturnParams mp{};
        mp.z = consts->redshift;
        mp.mturn_acg_homogeneous = consts->mturn_acg_homogeneous;
        mp.a_lw = astro_params_global->A_LW;
        mp.beta_lw = astro_params_global->BETA_LW;
        mp.a_vcb = astro_params_global->A_VCB;
        mp.beta_vcb = astro_params_global->BETA_VCB;
        mp.sigma_vcb = cosmo_tables_global->V_CB_AVG * sqrt(3.0 * M_PI / 8.0);
        mp.mturn_stellar_feedback = astro_params_global->M_TURN_STELLAR_FEEDBACK;
        mp.use_reion_feedback = use_reion_fb;
        mp.use_mcgs = use_mcgs;

        emissivity_mturn_kernel<<<blocks, kBlockThreads, 0, stream>>>(
            total, mp, (float)consts->vcb_const, below_zheat, vcb_flucts, d_vcb, d_j21, d_g12,
            d_zre, d_mturn_acg, d_mturn_mcg, d_blk0, d_blk1);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    if (ok) {
        finish_pair_sum_kernel<<<1, kBlockThreads, 0, stream>>>(d_blk0, d_blk1, blocks, d_out2);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    double h_sums[2] = {0.0, 0.0};
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(h_sums, d_out2, 2 * sizeof(double),
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }
    p21_cuda_pool_free(d_blk0, stream);
    p21_cuda_pool_free(d_blk1, stream);
    p21_cuda_pool_free(d_out2, stream);

    if (!ok) {
        p21_cuda_pool_free(d_mturn_acg, stream);
        p21_cuda_pool_free(d_mturn_mcg, stream);
        return false;
    }

    log10_mturn_averages_out[0] =
        use_reion_fb ? (h_sums[0] / (double)total) : log10(consts->mturn_acg_homogeneous);
    log10_mturn_averages_out[1] = use_mcgs ? (h_sums[1] / (double)total) : 0.0;
    *d_mturn_acg_out = d_mturn_acg;
    *d_mturn_mcg_out = d_mturn_mcg;
    return true;
}

extern "C" P21CudaEmissivityCtx *p21_cuda_emissivity_begin(
    double redshift, InitialConditions *ini_boxes, PerturbedField *perturbed_field,
    TsBox *previous_spin_temp, IonizedBox *previous_ionize_box, const ScalingConstants *consts,
    double log10_mturn_averages[2], double *min_dens_out, double *max_dens_out) {
    (void)redshift;
    if (!perturbed_field || !perturbed_field->density || !consts || !log10_mturn_averages ||
        !min_dens_out || !max_dens_out || !p21_cuda_get_enabled()) {
        return nullptr;
    }
    if (!source_model_uses_eulerian_grids(matter_options_global->SOURCE_MODEL) ||
        simulation_options_global->HII_DIM <= 1 ||
        !uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)) {
        return nullptr;
    }
    if (!p21_cuda_ensure_init()) {
        return nullptr;
    }

    size_t total = (size_t)simulation_options_global->HII_DIM *
                   (size_t)simulation_options_global->HII_DIM * (size_t)HII_D_PARA;
    cudaStream_t stream = p21_cuda_stream();

    const float *d_dens = (const float *)p21_cuda_registry_get_or_upload(
        perturbed_field->density, total * sizeof(float), stream);
    if (!d_dens) {
        return nullptr;
    }

    float *d_mturn_acg = nullptr, *d_mturn_mcg = nullptr;
    if (!p21_cuda_build_mturn_grids(consts, ini_boxes, previous_spin_temp, previous_ionize_box,
                                    stream, &d_mturn_acg, &d_mturn_mcg, log10_mturn_averages)) {
        return nullptr;
    }

    double min_d = 0.0, max_d = 0.0;
    if (!compute_min_max_dens(d_dens, total, stream, &min_d, &max_d)) {
        p21_cuda_pool_free(d_mturn_acg, stream);
        p21_cuda_pool_free(d_mturn_mcg, stream);
        return nullptr;
    }
    *min_dens_out = min_d * 1.001;
    *max_dens_out = max_d * 1.001;

    auto *ctx = new P21CudaEmissivityCtx;
    ctx->num_pixels = total;
    ctx->d_dens = d_dens;
    ctx->d_mturn_acg = d_mturn_acg;
    ctx->d_mturn_mcg = d_mturn_mcg;
    ctx->stream = stream;
    return ctx;
}

extern "C" int p21_cuda_emissivity_eval_and_finish(P21CudaEmissivityCtx *ctx,
                                                   const ScalingConstants *ev_consts,
                                                   const P21EmissivityAverages *uhmf_averages,
                                                   EmissivityFields *emissivity_fields) {
    if (!ctx || !ev_consts || !uhmf_averages || !emissivity_fields) {
        return -1;
    }

    cudaStream_t stream = ctx->stream;
    const size_t total = ctx->num_pixels;
    const size_t fbytes = total * sizeof(float);
    const int use_ts_fluct = astro_options_global->USE_TS_FLUCT ? 1 : 0;
    const int use_mcgs = astro_options_global->USE_MCGS ? 1 : 0;
    const int extra_fields = config_settings.EXTRA_EMISSIVITY_FIELDS ? 1 : 0;
    const int fix_mean = ev_consts->fix_mean ? 1 : 0;

    EmissivityEvalParams p{};
    bool ok = upload_cond_emissivity_params(ev_consts, stream, &p);

    float *d_nion = (float *)p21_cuda_pool_alloc(fbytes, stream);
    float *d_sfrd_acg = use_ts_fluct ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    float *d_sfrd_mcg =
        (use_ts_fluct && use_mcgs) ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    float *d_xray = use_ts_fluct ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    float *d_hn = extra_fields ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    float *d_hm = extra_fields ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    float *d_sa = extra_fields ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    float *d_sm =
        (extra_fields && use_mcgs) ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;

    int blocks = (int)((total + kBlockThreads - 1) / kBlockThreads);
    if (blocks > 1024) blocks = 1024;
    double *d_blk_sums = (double *)p21_cuda_pool_alloc(9ULL * blocks * sizeof(double), stream);
    double *d_out9 = (double *)p21_cuda_pool_alloc(9ULL * sizeof(double), stream);

    if (!d_nion || (use_ts_fluct && (!d_sfrd_acg || !d_xray || (use_mcgs && !d_sfrd_mcg))) ||
        (extra_fields && (!d_hn || !d_hm || !d_sa || (use_mcgs && !d_sm))) || !d_blk_sums ||
        !d_out9) {
        ok = false;
    }

    if (ok) {
        eulerian_emissivity_eval_kernel<<<blocks, kBlockThreads, 0, stream>>>(
            total, p, ctx->d_dens, ctx->d_mturn_acg, ctx->d_mturn_mcg, d_nion, d_sfrd_acg,
            d_sfrd_mcg, d_xray, d_hn, d_hm, d_sa, d_sm, d_blk_sums);
        ok = P21_CUDA_OK(cudaGetLastError());
    }

    if (ok && fix_mean) {
        finish_9way_sums_kernel<<<9, kBlockThreads, 0, stream>>>(d_blk_sums, blocks, d_out9);
        ok = P21_CUDA_OK(cudaGetLastError());
        double h_sums[9] = {};
        if (ok) {
            ok = P21_CUDA_OK(cudaMemcpyAsync(h_sums, d_out9, 9 * sizeof(double),
                                             cudaMemcpyDeviceToHost, stream)) &&
                 P21_CUDA_OK(cudaStreamSynchronize(stream));
        }
        if (ok) {
            auto ratio = [&](double g_val, double sum_val) -> double {
                double box_mean = sum_val / (double)total;
                return (box_mean != 0.0) ? (g_val / box_mean) : 1.0;
            };
            double s_nion = ratio(uhmf_averages->n_ion, h_sums[6]);
            double s_sfr_a = use_ts_fluct ? ratio(uhmf_averages->sfrd_acg, h_sums[4]) : 1.0;
            double s_sfr_m =
                (use_ts_fluct && use_mcgs) ? ratio(uhmf_averages->sfrd_mcg, h_sums[5]) : 1.0;
            double s_xray = use_ts_fluct ? ratio(uhmf_averages->xray_emissivity, h_sums[7]) : 1.0;
            double s_hn = extra_fields ? ratio(uhmf_averages->halo_number, h_sums[0]) : 1.0;
            double s_hm = extra_fields ? ratio(uhmf_averages->halo_mass_density, h_sums[1]) : 1.0;
            double s_sa =
                extra_fields ? ratio(uhmf_averages->stellar_mass_density_acg, h_sums[2]) : 1.0;
            double s_sm = (extra_fields && use_mcgs)
                              ? ratio(uhmf_averages->stellar_mass_density_mcg, h_sums[3])
                              : 1.0;

            int grid = (int)((total + kBlockThreads - 1) / kBlockThreads);
            emissivity_mean_fix_kernel<<<grid, kBlockThreads, 0, stream>>>(
                total, use_ts_fluct, use_mcgs, extra_fields, s_nion, s_sfr_a, s_sfr_m, s_xray,
                s_hn, s_hm, s_sa, s_sm, d_nion, d_sfrd_acg, d_sfrd_mcg, d_xray, d_hn, d_hm, d_sa,
                d_sm);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
    }

    free_cond_emissivity_params(&p, stream);
    p21_cuda_pool_free(d_blk_sums, stream);
    p21_cuda_pool_free(d_out9, stream);
    p21_cuda_pool_free(ctx->d_mturn_acg, stream);
    p21_cuda_pool_free(ctx->d_mturn_mcg, stream);
    ctx->d_mturn_acg = nullptr;
    ctx->d_mturn_mcg = nullptr;

    if (ok) {
        auto copy_d2h = [&](float *hptr, const float *dptr) {
            if (!ok || !hptr || !dptr) return;
            ok = P21_CUDA_OK(cudaMemcpyAsync(hptr, dptr, fbytes, cudaMemcpyDeviceToHost, stream));
            if (ok) p21_cuda_stats_add_d2h(fbytes);
        };
        copy_d2h(emissivity_fields->n_ion, d_nion);
        copy_d2h(emissivity_fields->sfrd_acg, d_sfrd_acg);
        copy_d2h(emissivity_fields->sfrd_mcg, d_sfrd_mcg);
        copy_d2h(emissivity_fields->xray_emissivity, d_xray);
        copy_d2h(emissivity_fields->halo_number, d_hn);
        copy_d2h(emissivity_fields->halo_mass_density, d_hm);
        copy_d2h(emissivity_fields->stellar_mass_density_acg, d_sa);
        copy_d2h(emissivity_fields->stellar_mass_density_mcg, d_sm);
        if (ok) ok = P21_CUDA_OK(cudaStreamSynchronize(stream));
    }

    if (ok) {
        auto own_or_free = [&](const void *hptr, float *&dbuf) {
            if (!dbuf) return;
            if (!hptr || !p21_cuda_registry_put_owned(hptr, dbuf, fbytes, stream)) {
                p21_cuda_pool_free(dbuf, stream);
            }
            dbuf = nullptr;
        };
        own_or_free(emissivity_fields->n_ion, d_nion);
        own_or_free(emissivity_fields->sfrd_acg, d_sfrd_acg);
        own_or_free(emissivity_fields->sfrd_mcg, d_sfrd_mcg);
        own_or_free(emissivity_fields->xray_emissivity, d_xray);
    }

    p21_cuda_pool_free(d_nion, stream);
    p21_cuda_pool_free(d_sfrd_acg, stream);
    p21_cuda_pool_free(d_sfrd_mcg, stream);
    p21_cuda_pool_free(d_xray, stream);
    p21_cuda_pool_free(d_hn, stream);
    p21_cuda_pool_free(d_hm, stream);
    p21_cuda_pool_free(d_sa, stream);
    p21_cuda_pool_free(d_sm, stream);
    delete ctx;

    if (!ok) {
        return -1;
    }
    p21_cuda_stats_mut()->emissivity_runs++;
    return 0;
}

extern "C" void p21_cuda_emissivity_abort(P21CudaEmissivityCtx *ctx) {
    if (!ctx) return;
    p21_cuda_pool_free(ctx->d_mturn_acg, ctx->stream);
    p21_cuda_pool_free(ctx->d_mturn_mcg, ctx->stream);
    delete ctx;
}

extern "C" P21CudaLagEmissCtx *p21_cuda_lag_emissivity_begin(
    double redshift, InitialConditions *ini_boxes, TsBox *previous_spin_temp,
    IonizedBox *previous_ionize_box, const ScalingConstants *consts, int need_integral,
    double log10_mturn_averages[2], double *min_dens_out, double *max_dens_out) {
    (void)redshift;
    if (!ini_boxes || !consts || !log10_mturn_averages || !min_dens_out || !max_dens_out ||
        !p21_cuda_get_enabled()) {
        return nullptr;
    }
    if (!source_model_uses_lagrangian_grids(matter_options_global->SOURCE_MODEL) ||
        simulation_options_global->HII_DIM <= 1 ||
        !uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)) {
        return nullptr;
    }
    if (!p21_cuda_ensure_init()) {
        return nullptr;
    }

    cudaStream_t stream = p21_cuda_stream();
    auto *ctx = new P21CudaLagEmissCtx;
    ctx->stream = stream;
    ctx->out_dim[0] = simulation_options_global->HII_DIM;
    ctx->out_dim[1] = simulation_options_global->HII_DIM;
    ctx->out_dim[2] = HII_D_PARA;
    ctx->out_pixels = (size_t)ctx->out_dim[0] * (size_t)ctx->out_dim[1] * (size_t)ctx->out_dim[2];

    const bool hires_dens = matter_options_global->PERTURB_ON_HIGH_RES;
    ctx->dens_dim[0] = hires_dens ? simulation_options_global->DIM : ctx->out_dim[0];
    ctx->dens_dim[1] = hires_dens ? simulation_options_global->DIM : ctx->out_dim[1];
    ctx->dens_dim[2] = hires_dens ? D_PARA : ctx->out_dim[2];
    ctx->dens_pixels =
        (size_t)ctx->dens_dim[0] * (size_t)ctx->dens_dim[1] * (size_t)ctx->dens_dim[2];

    ctx->use_ts_fluct = astro_options_global->USE_TS_FLUCT ? 1 : 0;
    ctx->use_mcgs = astro_options_global->USE_MCGS ? 1 : 0;
    ctx->use_metallicity = astro_options_global->USE_METALLICITY ? 1 : 0;
    ctx->use_reion_fb = astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK ? 1 : 0;
    ctx->uses_recomb = uses_recombination(astro_options_global->RECOMB_MODEL) ? 1 : 0;
    ctx->extra_fields = config_settings.EXTRA_EMISSIVITY_FIELDS ? 1 : 0;
    ctx->mass_dep = source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL) ? 1 : 0;

    if (!p21_cuda_build_mturn_grids(consts, ini_boxes, previous_spin_temp, previous_ionize_box,
                                    stream, &ctx->d_mturn_acg, &ctx->d_mturn_mcg,
                                    log10_mturn_averages)) {
        delete ctx;
        return nullptr;
    }

    *min_dens_out = 0.0;
    *max_dens_out = 0.0;
    if (need_integral) {
        const float *h_dens = hires_dens ? ini_boxes->hires_density : ini_boxes->lowres_density;
        if (!h_dens) {
            free_lag_ctx_buffers(ctx);
            delete ctx;
            return nullptr;
        }
        ctx->d_dens = (const float *)p21_cuda_registry_get_or_upload(
            h_dens, ctx->dens_pixels * sizeof(float), stream);
        if (!ctx->d_dens) {
            free_lag_ctx_buffers(ctx);
            delete ctx;
            return nullptr;
        }
        double min_d = 0.0, max_d = 0.0;
        if (!compute_min_max_dens(ctx->d_dens, ctx->dens_pixels, stream, &min_d, &max_d)) {
            free_lag_ctx_buffers(ctx);
            delete ctx;
            return nullptr;
        }
        *min_dens_out = min_d * consts->growth_factor * 1.001;
        *max_dens_out = max_d * consts->growth_factor * 1.001;
    }

    const size_t dbytes = ctx->out_pixels * sizeof(double);
    auto alloc_zero = [&](double **ptr) -> bool {
        *ptr = (double *)p21_cuda_pool_alloc(dbytes, stream);
        if (!*ptr) return false;
        return P21_CUDA_OK(cudaMemsetAsync(*ptr, 0, dbytes, stream));
    };

    bool ok = alloc_zero(&ctx->d_acc_nion);
    if (ok && ctx->use_ts_fluct) {
        ok = alloc_zero(&ctx->d_acc_sfrd_acg) && alloc_zero(&ctx->d_acc_xray);
        if (ok && ctx->use_mcgs) {
            ok = alloc_zero(&ctx->d_acc_sfrd_mcg);
        }
    }
    if (ok && ctx->uses_recomb) {
        ok = alloc_zero(&ctx->d_acc_fesc_wsfr);
    }
    if (ok && ctx->extra_fields) {
        ok = alloc_zero(&ctx->d_acc_hn) && alloc_zero(&ctx->d_acc_hm) &&
             alloc_zero(&ctx->d_acc_sa);
        if (ok && ctx->use_mcgs) {
            ok = alloc_zero(&ctx->d_acc_sm);
        }
    }
    if (!ok) {
        free_lag_ctx_buffers(ctx);
        delete ctx;
        return nullptr;
    }
    return ctx;
}

extern "C" int p21_cuda_lag_emissivity_add_halos(P21CudaLagEmissCtx *ctx,
                                                 InitialConditions *ini_boxes, HaloCatalog *halos,
                                                 const ScalingConstants *consts) {
    if (!ctx || !ini_boxes || !halos || !consts) {
        return -1;
    }
    if (halos->n_halos == 0) {
        return 0;
    }
    if (!halos->halo_masses || !halos->halo_coords || !halos->star_rng || !halos->sfr_rng ||
        !halos->xray_rng) {
        return -1;
    }

    cudaStream_t stream = ctx->stream;
    const bool on_hires = matter_options_global->PERTURB_ON_HIGH_RES;
    P21DevVelFields velf{};
    if (!p21_upload_vel_fields(ini_boxes, on_hires, stream, &velf)) {
        return -1;
    }

    const size_t nh = (size_t)halos->n_halos;
    const float *d_hm =
        (const float *)p21_cuda_registry_get_or_upload(halos->halo_masses, nh * sizeof(float),
                                                       stream);
    const float *d_hc = (const float *)p21_cuda_registry_get_or_upload(
        halos->halo_coords, 3ULL * nh * sizeof(float), stream);
    const float *d_sr =
        (const float *)p21_cuda_registry_get_or_upload(halos->star_rng, nh * sizeof(float), stream);
    const float *d_sfrr =
        (const float *)p21_cuda_registry_get_or_upload(halos->sfr_rng, nh * sizeof(float), stream);
    const float *d_xr =
        (const float *)p21_cuda_registry_get_or_upload(halos->xray_rng, nh * sizeof(float), stream);
    if (!d_hm || !d_hc || !d_sr || !d_sfrr || !d_xr) {
        return -1;
    }

    const double boxlen = simulation_options_global->BOX_LEN;
    const double boxlen_z = boxlen * simulation_options_global->NON_CUBIC_FACTOR;
    const double cell_size_inv_o = (double)ctx->out_dim[0] / boxlen;
    const double cell_vol_inv = cell_size_inv_o * cell_size_inv_o * cell_size_inv_o;

    const double growth_factor = consts->growth_factor;
    const double disp_2lpt = -(3.0 / 7.0) * growth_factor * growth_factor;
    const double init_gf = dicke(simulation_options_global->INITIAL_REDSHIFT);
    const double init_disp_2lpt = -(3.0 / 7.0) * init_gf * init_gf;
    const double vdf = growth_factor - init_gf;
    const double vdf2 = disp_2lpt - init_disp_2lpt;

    P21HaloScalingParams hp = p21_make_halo_scaling_params(consts);
    int grid_h = (int)((nh + kBlockThreads - 1) / kBlockThreads);
    lag_deposit_halos_kernel<<<grid_h, kBlockThreads, 0, stream>>>(
        nh, hp, boxlen, boxlen_z, vdf, vdf2, velf.dim[0], velf.dim[1], velf.dim[2],
        ctx->out_dim[0], ctx->out_dim[1], ctx->out_dim[2], ctx->use_reion_fb, ctx->uses_recomb,
        ctx->extra_fields, velf.is_2lpt ? 1 : 0, velf.v[0], velf.v[1], velf.v[2], velf.v2[0],
        velf.v2[1], velf.v2[2], ctx->d_mturn_acg, ctx->d_mturn_mcg, d_hm, d_hc, d_sr, d_sfrr,
        d_xr, ctx->d_acc_nion, ctx->d_acc_sfrd_acg, ctx->d_acc_sfrd_mcg, ctx->d_acc_xray,
        ctx->d_acc_fesc_wsfr, ctx->d_acc_hn, ctx->d_acc_hm, ctx->d_acc_sa, ctx->d_acc_sm);
    if (!P21_CUDA_OK(cudaGetLastError())) {
        return -1;
    }

    int grid_c = (int)((ctx->out_pixels + kBlockThreads - 1) / kBlockThreads);
    lag_scale_halos_kernel<<<grid_c, kBlockThreads, 0, stream>>>(
        ctx->out_pixels, cell_vol_inv, ctx->use_ts_fluct, ctx->use_mcgs, ctx->uses_recomb,
        ctx->extra_fields, ctx->d_acc_nion, ctx->d_acc_sfrd_acg, ctx->d_acc_sfrd_mcg,
        ctx->d_acc_xray, ctx->d_acc_fesc_wsfr, ctx->d_acc_hm, ctx->d_acc_sa, ctx->d_acc_sm);
    if (!P21_CUDA_OK(cudaGetLastError())) {
        return -1;
    }
    return 0;
}

extern "C" int p21_cuda_lag_emissivity_add_integral(P21CudaLagEmissCtx *ctx,
                                                    InitialConditions *ini_boxes,
                                                    const ScalingConstants *ev_consts,
                                                    const P21EmissivityAverages *uhmf_averages) {
    if (!ctx || !ini_boxes || !ev_consts || !uhmf_averages || !ctx->d_dens) {
        return -1;
    }

    cudaStream_t stream = ctx->stream;
    const bool on_hires = matter_options_global->PERTURB_ON_HIGH_RES;
    P21DevVelFields velf{};
    if (!p21_upload_vel_fields(ini_boxes, on_hires, stream, &velf)) {
        return -1;
    }

    EmissivityEvalParams p{};
    bool ok = upload_cond_emissivity_params(ev_consts, stream, &p);

    const double boxlen = simulation_options_global->BOX_LEN;
    const double box_size[3] = {boxlen, boxlen, boxlen * simulation_options_global->NON_CUBIC_FACTOR};
    const double dim_ratio_out = (double)ctx->out_dim[0] / (double)ctx->dens_dim[0];
    const double vol_ratio_out =
        (double)ctx->out_pixels / (double)ctx->dens_pixels;

    const double growth_factor = ev_consts->growth_factor;
    const double disp_2lpt = -(3.0 / 7.0) * growth_factor * growth_factor;
    const double init_gf = dicke(simulation_options_global->INITIAL_REDSHIFT);
    const double init_disp_2lpt = -(3.0 / 7.0) * init_gf * init_gf;

    const double vdf0 = (growth_factor - init_gf) / box_size[0] * (double)ctx->dens_dim[0];
    const double vdf1 = (growth_factor - init_gf) / box_size[1] * (double)ctx->dens_dim[1];
    const double vdf2 = (growth_factor - init_gf) / box_size[2] * (double)ctx->dens_dim[2];
    const double vdf2_0 =
        (disp_2lpt - init_disp_2lpt) / box_size[0] * (double)ctx->dens_dim[0];
    const double vdf2_1 =
        (disp_2lpt - init_disp_2lpt) / box_size[1] * (double)ctx->dens_dim[1];
    const double vdf2_2 =
        (disp_2lpt - init_disp_2lpt) / box_size[2] * (double)ctx->dens_dim[2];

    if (ok) {
        int grid_d = (int)((ctx->dens_pixels + kBlockThreads - 1) / kBlockThreads);
        lag_deposit_integral_kernel<<<grid_d, kBlockThreads, 0, stream>>>(
            ctx->dens_pixels, ctx->dens_dim[0], ctx->dens_dim[1], ctx->dens_dim[2], ctx->out_dim[0],
            ctx->out_dim[1], ctx->out_dim[2], dim_ratio_out, vol_ratio_out, vdf0, vdf1, vdf2,
            vdf2_0, vdf2_1, vdf2_2, velf.is_2lpt ? 1 : 0, p, ctx->d_dens, velf.v[0], velf.v[1],
            velf.v[2], velf.v2[0], velf.v2[1], velf.v2[2], ctx->d_mturn_acg, ctx->d_mturn_mcg,
            ctx->d_acc_nion, ctx->d_acc_sfrd_acg, ctx->d_acc_sfrd_mcg, ctx->d_acc_xray,
            ctx->d_acc_hn, ctx->d_acc_hm, ctx->d_acc_sa, ctx->d_acc_sm);
        ok = P21_CUDA_OK(cudaGetLastError());
    }

    if (ok && ((ctx->use_ts_fluct && !ctx->use_metallicity) || ctx->uses_recomb)) {
        int grid_o = (int)((ctx->out_pixels + kBlockThreads - 1) / kBlockThreads);
        lag_post_integral_kernel<<<grid_o, kBlockThreads, 0, stream>>>(
            ctx->out_pixels, ctx->use_ts_fluct, ctx->use_mcgs, ctx->use_metallicity,
            ctx->uses_recomb, ctx->mass_dep, p.lx_over_sfr_acg, p.lx_over_sfr_mcg, p.rhocrit_omb,
            p.sfr_timescale, p.fstar_10_acg, p.hii_eff_factor, ctx->d_acc_nion,
            ctx->d_acc_sfrd_acg, ctx->d_acc_sfrd_mcg, ctx->d_acc_xray, ctx->d_acc_fesc_wsfr);
        ok = P21_CUDA_OK(cudaGetLastError());
    }

    if (ok && ev_consts->fix_mean) {
        int blocks = (int)((ctx->out_pixels + kBlockThreads - 1) / kBlockThreads);
        if (blocks > 1024) blocks = 1024;
        double *d_blk_sums = (double *)p21_cuda_pool_alloc(9ULL * blocks * sizeof(double), stream);
        double *d_out9 = (double *)p21_cuda_pool_alloc(9ULL * sizeof(double), stream);
        if (!d_blk_sums || !d_out9) {
            ok = false;
        }
        if (ok) {
            lag_9way_sums_kernel<<<blocks, kBlockThreads, 0, stream>>>(
                ctx->out_pixels, ctx->use_ts_fluct, ctx->use_mcgs, ctx->uses_recomb,
                ctx->extra_fields, ctx->d_acc_nion, ctx->d_acc_sfrd_acg, ctx->d_acc_sfrd_mcg,
                ctx->d_acc_xray, ctx->d_acc_fesc_wsfr, ctx->d_acc_hn, ctx->d_acc_hm, ctx->d_acc_sa,
                ctx->d_acc_sm, d_blk_sums);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
        if (ok) {
            finish_9way_sums_kernel<<<9, kBlockThreads, 0, stream>>>(d_blk_sums, blocks, d_out9);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
        double h_sums[9] = {};
        if (ok) {
            ok = P21_CUDA_OK(cudaMemcpyAsync(h_sums, d_out9, 9 * sizeof(double),
                                             cudaMemcpyDeviceToHost, stream)) &&
                 P21_CUDA_OK(cudaStreamSynchronize(stream));
        }
        if (ok) {
            auto ratio = [&](double g_val, double sum_val) -> double {
                double box_mean = sum_val / (double)ctx->out_pixels;
                return (box_mean != 0.0) ? (g_val / box_mean) : 1.0;
            };
            double s_nion = ratio(uhmf_averages->n_ion, h_sums[6]);
            double s_sfr_a = ctx->use_ts_fluct ? ratio(uhmf_averages->sfrd_acg, h_sums[4]) : 1.0;
            double s_sfr_m =
                (ctx->use_ts_fluct && ctx->use_mcgs) ? ratio(uhmf_averages->sfrd_mcg, h_sums[5]) : 1.0;
            double s_xray =
                ctx->use_ts_fluct ? ratio(uhmf_averages->xray_emissivity, h_sums[7]) : 1.0;
            double s_fesc =
                ctx->uses_recomb ? ratio(uhmf_averages->fesc_weighted_sfrd, h_sums[8]) : 1.0;
            double s_hn = ctx->extra_fields ? ratio(uhmf_averages->halo_number, h_sums[0]) : 1.0;
            double s_hm =
                ctx->extra_fields ? ratio(uhmf_averages->halo_mass_density, h_sums[1]) : 1.0;
            double s_sa =
                ctx->extra_fields ? ratio(uhmf_averages->stellar_mass_density_acg, h_sums[2]) : 1.0;
            double s_sm = (ctx->extra_fields && ctx->use_mcgs)
                              ? ratio(uhmf_averages->stellar_mass_density_mcg, h_sums[3])
                              : 1.0;

            int grid_o = (int)((ctx->out_pixels + kBlockThreads - 1) / kBlockThreads);
            lag_mean_fix_scale_kernel<<<grid_o, kBlockThreads, 0, stream>>>(
                ctx->out_pixels, ctx->use_ts_fluct, ctx->use_mcgs, ctx->uses_recomb,
                ctx->extra_fields, s_nion, s_sfr_a, s_sfr_m, s_xray, s_fesc, s_hn, s_hm, s_sa,
                s_sm, ctx->d_acc_nion, ctx->d_acc_sfrd_acg, ctx->d_acc_sfrd_mcg, ctx->d_acc_xray,
                ctx->d_acc_fesc_wsfr, ctx->d_acc_hn, ctx->d_acc_hm, ctx->d_acc_sa, ctx->d_acc_sm);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
        p21_cuda_pool_free(d_blk_sums, stream);
        p21_cuda_pool_free(d_out9, stream);
    }

    free_cond_emissivity_params(&p, stream);
    return ok ? 0 : -1;
}

extern "C" int p21_cuda_lag_emissivity_finish(P21CudaLagEmissCtx *ctx,
                                              EmissivityFields *emissivity_fields) {
    if (!ctx || !emissivity_fields) {
        return -1;
    }

    cudaStream_t stream = ctx->stream;
    const size_t total = ctx->out_pixels;
    const size_t fbytes = total * sizeof(float);

    float *d_nion = (float *)p21_cuda_pool_alloc(fbytes, stream);
    float *d_sfrd_acg = ctx->use_ts_fluct ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    float *d_sfrd_mcg = (ctx->use_ts_fluct && ctx->use_mcgs)
                            ? (float *)p21_cuda_pool_alloc(fbytes, stream)
                            : nullptr;
    float *d_xray = ctx->use_ts_fluct ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    float *d_fesc = ctx->uses_recomb ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    float *d_hn = ctx->extra_fields ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    float *d_hm = ctx->extra_fields ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    float *d_sa = ctx->extra_fields ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    float *d_sm = (ctx->extra_fields && ctx->use_mcgs)
                          ? (float *)p21_cuda_pool_alloc(fbytes, stream)
                          : nullptr;

    bool ok = true;
    if (!d_nion ||
        (ctx->use_ts_fluct && (!d_sfrd_acg || !d_xray || (ctx->use_mcgs && !d_sfrd_mcg))) ||
        (ctx->uses_recomb && !d_fesc) ||
        (ctx->extra_fields && (!d_hn || !d_hm || !d_sa || (ctx->use_mcgs && !d_sm)))) {
        ok = false;
    }

    if (ok) {
        int grid_o = (int)((total + kBlockThreads - 1) / kBlockThreads);
        lag_cast_f64_to_f32_kernel<<<grid_o, kBlockThreads, 0, stream>>>(
            total, ctx->use_ts_fluct, ctx->use_mcgs, ctx->uses_recomb, ctx->extra_fields,
            ctx->d_acc_nion, ctx->d_acc_sfrd_acg, ctx->d_acc_sfrd_mcg, ctx->d_acc_xray,
            ctx->d_acc_fesc_wsfr, ctx->d_acc_hn, ctx->d_acc_hm, ctx->d_acc_sa, ctx->d_acc_sm,
            d_nion, d_sfrd_acg, d_sfrd_mcg, d_xray, d_fesc, d_hn, d_hm, d_sa, d_sm);
        ok = P21_CUDA_OK(cudaGetLastError());
    }

    if (ok) {
        auto copy_d2h = [&](float *hptr, const float *dptr) {
            if (!ok || !hptr || !dptr) return;
            ok = P21_CUDA_OK(cudaMemcpyAsync(hptr, dptr, fbytes, cudaMemcpyDeviceToHost, stream));
            if (ok) p21_cuda_stats_add_d2h(fbytes);
        };
        copy_d2h(emissivity_fields->n_ion, d_nion);
        copy_d2h(emissivity_fields->sfrd_acg, d_sfrd_acg);
        copy_d2h(emissivity_fields->sfrd_mcg, d_sfrd_mcg);
        copy_d2h(emissivity_fields->xray_emissivity, d_xray);
        copy_d2h(emissivity_fields->fesc_weighted_sfrd, d_fesc);
        copy_d2h(emissivity_fields->halo_number, d_hn);
        copy_d2h(emissivity_fields->halo_mass_density, d_hm);
        copy_d2h(emissivity_fields->stellar_mass_density_acg, d_sa);
        copy_d2h(emissivity_fields->stellar_mass_density_mcg, d_sm);
        if (ok) ok = P21_CUDA_OK(cudaStreamSynchronize(stream));
    }

    if (ok) {
        auto own_or_free = [&](const void *hptr, float *&dbuf) {
            if (!dbuf) return;
            if (!hptr || !p21_cuda_registry_put_owned(hptr, dbuf, fbytes, stream)) {
                p21_cuda_pool_free(dbuf, stream);
            }
            dbuf = nullptr;
        };
        own_or_free(emissivity_fields->n_ion, d_nion);
        own_or_free(emissivity_fields->sfrd_acg, d_sfrd_acg);
        own_or_free(emissivity_fields->sfrd_mcg, d_sfrd_mcg);
        own_or_free(emissivity_fields->xray_emissivity, d_xray);
        own_or_free(emissivity_fields->fesc_weighted_sfrd, d_fesc);
    }

    p21_cuda_pool_free(d_nion, stream);
    p21_cuda_pool_free(d_sfrd_acg, stream);
    p21_cuda_pool_free(d_sfrd_mcg, stream);
    p21_cuda_pool_free(d_xray, stream);
    p21_cuda_pool_free(d_fesc, stream);
    p21_cuda_pool_free(d_hn, stream);
    p21_cuda_pool_free(d_hm, stream);
    p21_cuda_pool_free(d_sa, stream);
    p21_cuda_pool_free(d_sm, stream);
    free_lag_ctx_buffers(ctx);
    delete ctx;

    if (!ok) {
        return -1;
    }
    p21_cuda_stats_mut()->emissivity_runs++;
    return 0;
}

extern "C" void p21_cuda_lag_emissivity_abort(P21CudaLagEmissCtx *ctx) {
    if (!ctx) return;
    free_lag_ctx_buffers(ctx);
    delete ctx;
}
