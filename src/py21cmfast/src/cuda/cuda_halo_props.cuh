#ifndef P21_CUDA_HALO_PROPS_CUH
#define P21_CUDA_HALO_PROPS_CUH

#include "cuda_common.cuh"

/* Host-to-device parameter bundle matching ScalingConstants + scaling_relations.c. */
struct P21HaloScalingParams {
    double redshift;
    double growth_factor;
    double fstar_10_acg;
    double alpha_star_acg;
    double sigma_star;
    double alpha_upper;
    double pivot_upper;
    double upper_pivot_ratio;
    double fstar_7_mcg;
    double alpha_star_mcg;
    double sfr_timescale;
    double sigma_sfr_lim;
    double sigma_sfr_idx;
    double lx_over_sfr_acg;
    double lx_over_sfr_mcg;
    double sigma_xray;
    double fesc_10_acg;
    double alpha_esc;
    double fesc_7_mcg;
    double pop2_ion;
    double pop3_ion;
    double atomic_cooling_threshold;
    double mturn_acg_homogeneous;
    double omb_over_omm;
    double rhocrit_omb;
    double s_per_yr;
    int scaling_median;
    int use_upper_stellar_turnover;
    int use_mcgs;
    int use_ts_fluct;
    int use_metallicity;
};

struct P21HaloPropsOut {
    double halo_mass;
    double stellar_mass_acg;
    double stellar_mass_mcg;
    double sfr_acg;
    double sfr_mcg;
    double fesc_weighted_sfr;
    double n_ion;
    double xray_luminosity;
    double metallicity_acg;
    double metallicity_mcg;
};

static inline P21HaloScalingParams p21_make_halo_scaling_params(const ScalingConstants *sc) {
    P21HaloScalingParams p{};
    p.redshift = sc->redshift;
    p.growth_factor = sc->growth_factor;
    p.fstar_10_acg = sc->fstar_10_acg;
    p.alpha_star_acg = sc->alpha_star_acg;
    p.sigma_star = sc->sigma_star;
    p.alpha_upper = sc->alpha_upper;
    p.pivot_upper = sc->pivot_upper;
    p.upper_pivot_ratio = sc->upper_pivot_ratio;
    p.fstar_7_mcg = sc->fstar_7_mcg;
    p.alpha_star_mcg = sc->alpha_star_mcg;
    p.sfr_timescale = sc->sfr_timescale;
    p.sigma_sfr_lim = sc->sigma_sfr_lim;
    p.sigma_sfr_idx = sc->sigma_sfr_idx;
    p.lx_over_sfr_acg = sc->lx_over_sfr_acg;
    p.lx_over_sfr_mcg = sc->lx_over_sfr_mcg;
    p.sigma_xray = sc->sigma_xray;
    p.fesc_10_acg = sc->fesc_10_acg;
    p.alpha_esc = sc->alpha_esc;
    p.fesc_7_mcg = sc->fesc_7_mcg;
    p.pop2_ion = sc->pop2_ion;
    p.pop3_ion = sc->pop3_ion;
    p.atomic_cooling_threshold = sc->atomic_cooling_threshold;
    p.mturn_acg_homogeneous = sc->mturn_acg_homogeneous;
    p.omb_over_omm = (double)cosmo_params_global->OMb / (double)cosmo_params_global->OMm;
    p.rhocrit_omb = RHOcrit * (double)cosmo_params_global->OMb;
    p.s_per_yr = physconst.s_per_yr;
    p.scaling_median = sc->scaling_median ? 1 : 0;
    p.use_upper_stellar_turnover = astro_options_global->USE_UPPER_STELLAR_TURNOVER ? 1 : 0;
    p.use_mcgs = astro_options_global->USE_MCGS ? 1 : 0;
    p.use_ts_fluct = astro_options_global->USE_TS_FLUCT ? 1 : 0;
    p.use_metallicity = astro_options_global->USE_METALLICITY ? 1 : 0;
    return p;
}

__host__ __device__ static inline void p21_pos_to_index(const double pos[3], double cell_size_inv,
                                                        int idx[3]) {
    idx[0] = (int)(pos[0] * cell_size_inv + 0.5);
    idx[1] = (int)(pos[1] * cell_size_inv + 0.5);
    idx[2] = (int)(pos[2] * cell_size_inv + 0.5);
}

__host__ __device__ static inline void p21_wrap_position(double pos[3], const double size[3]) {
    for (int axis = 0; axis < 3; ++axis) {
        double s = size[axis];
        while (pos[axis] >= s) {
            pos[axis] -= s;
        }
        while (pos[axis] < 0.0) {
            pos[axis] += s;
        }
    }
}

__device__ static inline double p21_cic_read_float(const float *box, const double pos[3],
                                                   int nx, int ny, int nz) {
    int ip0[3], ip1[3];
    double dist[3];
    int dims[3] = {nx, ny, nz};
    for (int axis = 0; axis < 3; ++axis) {
        int i0 = (int)floor(pos[axis]);
        dist[axis] = pos[axis] - (double)i0;
        ip0[axis] = p21_wrap_coord(i0, dims[axis]);
        ip1[axis] = p21_wrap_coord(i0 + 1, dims[axis]);
    }
    const double wx0 = 1.0 - dist[0], wx1 = dist[0];
    const double wy0 = 1.0 - dist[1], wy1 = dist[1];
    const double wz0 = 1.0 - dist[2], wz1 = dist[2];

    double sum = 0.0;
    sum += wx0 * wy0 * wz0 * (double)box[p21_grid_idx(ip0[0], ip0[1], ip0[2], ny, nz)];
    sum += wx1 * wy0 * wz0 * (double)box[p21_grid_idx(ip1[0], ip0[1], ip0[2], ny, nz)];
    sum += wx0 * wy1 * wz0 * (double)box[p21_grid_idx(ip0[0], ip1[1], ip0[2], ny, nz)];
    sum += wx1 * wy1 * wz0 * (double)box[p21_grid_idx(ip1[0], ip1[1], ip0[2], ny, nz)];
    sum += wx0 * wy0 * wz1 * (double)box[p21_grid_idx(ip0[0], ip0[1], ip1[2], ny, nz)];
    sum += wx1 * wy0 * wz1 * (double)box[p21_grid_idx(ip1[0], ip0[1], ip1[2], ny, nz)];
    sum += wx0 * wy1 * wz1 * (double)box[p21_grid_idx(ip0[0], ip1[1], ip1[2], ny, nz)];
    sum += wx1 * wy1 * wz1 * (double)box[p21_grid_idx(ip1[0], ip1[1], ip1[2], ny, nz)];
    return sum;
}

struct P21CicStencils {
    size_t idx[8];
    double w[8];
};

__device__ static inline void p21_cic_compute(const double pos[3], int nx, int ny, int nz,
                                              P21CicStencils *out) {
    int ip0[3], ip1[3];
    double dist[3];
    int dims[3] = {nx, ny, nz};
    for (int axis = 0; axis < 3; ++axis) {
        int i0 = (int)floor(pos[axis]);
        dist[axis] = pos[axis] - (double)i0;
        ip0[axis] = p21_wrap_coord(i0, dims[axis]);
        ip1[axis] = p21_wrap_coord(i0 + 1, dims[axis]);
    }
    const double wx0 = 1.0 - dist[0], wx1 = dist[0];
    const double wy0 = 1.0 - dist[1], wy1 = dist[1];
    const double wz0 = 1.0 - dist[2], wz1 = dist[2];

    out->idx[0] = p21_grid_idx(ip0[0], ip0[1], ip0[2], ny, nz);
    out->idx[1] = p21_grid_idx(ip1[0], ip0[1], ip0[2], ny, nz);
    out->idx[2] = p21_grid_idx(ip0[0], ip1[1], ip0[2], ny, nz);
    out->idx[3] = p21_grid_idx(ip1[0], ip1[1], ip0[2], ny, nz);
    out->idx[4] = p21_grid_idx(ip0[0], ip0[1], ip1[2], ny, nz);
    out->idx[5] = p21_grid_idx(ip1[0], ip0[1], ip1[2], ny, nz);
    out->idx[6] = p21_grid_idx(ip0[0], ip1[1], ip1[2], ny, nz);
    out->idx[7] = p21_grid_idx(ip1[0], ip1[1], ip1[2], ny, nz);

    out->w[0] = wx0 * wy0 * wz0;
    out->w[1] = wx1 * wy0 * wz0;
    out->w[2] = wx0 * wy1 * wz0;
    out->w[3] = wx1 * wy1 * wz0;
    out->w[4] = wx0 * wy0 * wz1;
    out->w[5] = wx1 * wy0 * wz1;
    out->w[6] = wx0 * wy1 * wz1;
    out->w[7] = wx1 * wy1 * wz1;
}

__device__ static inline void p21_cic_deposit_stenciled(double *box, const P21CicStencils &s,
                                                        double val) {
    if (box == nullptr) {
        return;
    }
    for (int k = 0; k < 8; ++k) {
        p21_atomic_add_double(box + s.idx[k], val * s.w[k]);
    }
}

__device__ static inline double p21_scaling_single_PL(double M, double alpha, double pivot) {
    return pow(M / pivot, alpha);
}

__device__ static inline double p21_scaling_double_PL(double M, double alpha_lo, double pivot_ratio,
                                                      double alpha_hi, double pivot_hi) {
    return pivot_ratio / (pow(M / pivot_hi, -alpha_lo) + pow(M / pivot_hi, -alpha_hi));
}

__device__ static inline void p21_set_halo_properties_dev(double halo_mass, double M_turn_acg,
                                                          double M_turn_mcg,
                                                          const P21HaloScalingParams &p,
                                                          double star_rng, double sfr_rng,
                                                          double xray_rng, P21HaloPropsOut *out) {
    double stoc_star_adj = p.scaling_median ? 0.0 : (p.sigma_star * p.sigma_star * 0.5);

    double mu_fstar_acg;
    if (p.use_upper_stellar_turnover && (p.alpha_star_acg > p.alpha_upper)) {
        mu_fstar_acg =
            p.fstar_10_acg * p21_scaling_double_PL(halo_mass, p.alpha_star_acg, p.upper_pivot_ratio,
                                                   p.alpha_upper, p.pivot_upper);
    } else {
        mu_fstar_acg = p.fstar_10_acg * p21_scaling_single_PL(halo_mass, p.alpha_star_acg, 1e10);
    }
    double f_sample_acg =
        mu_fstar_acg * exp(-M_turn_acg / halo_mass + star_rng * p.sigma_star - stoc_star_adj);
    if (f_sample_acg > 1.0) {
        f_sample_acg = 1.0;
    }
    double star_acg = f_sample_acg * halo_mass * p.omb_over_omm;

    double star_mcg = 0.0;
    if (p.use_mcgs) {
        double f_sample_mcg;
        if (M_turn_acg > p.atomic_cooling_threshold * 1.001) {
            f_sample_mcg = 0.0;
        } else {
            double mu_fstar_mcg =
                p.fstar_7_mcg * p21_scaling_single_PL(halo_mass, p.alpha_star_mcg, 1e7);
            f_sample_mcg = mu_fstar_mcg *
                           exp(-M_turn_mcg / halo_mass - halo_mass / p.atomic_cooling_threshold +
                               star_rng * p.sigma_star - stoc_star_adj);
        }
        if (f_sample_mcg > 1.0) {
            f_sample_mcg = 1.0;
        }
        star_mcg = f_sample_mcg * halo_mass * p.omb_over_omm;
    }

    double sigma_sfr = 0.0;
    if (p.sigma_sfr_lim > 0.0) {
        sigma_sfr =
            p.sigma_sfr_idx * log10((star_acg + star_mcg) / 1e10) + p.sigma_sfr_lim;
        if (sigma_sfr < p.sigma_sfr_lim) {
            sigma_sfr = p.sigma_sfr_lim;
        }
    }
    double stoc_sfr_adj = p.scaling_median ? 0.0 : (sigma_sfr * sigma_sfr * 0.5);
    double sfr_acg = (star_acg / p.sfr_timescale) * exp(sfr_rng * sigma_sfr - stoc_sfr_adj);
    double sfr_mcg = 0.0;
    if (p.use_mcgs) {
        sfr_mcg = (star_mcg / p.sfr_timescale) * exp(sfr_rng * sigma_sfr - stoc_sfr_adj);
    }

    double met_acg = 0.0;
    double met_mcg = 0.0;
    double xray_lum = 0.0;
    if (p.use_ts_fluct) {
        if (p.use_metallicity) {
            auto calc_met = [&](double sfr, double stellar) -> double {
                double redshift_scaling = pow(10.0, -0.056 * p.redshift + 0.064);
                double stellar_term = 1.0;
                if (stellar > 0.0 && sfr > 0.0) {
                    double M0 = 1.28825e10 * pow(sfr * p.s_per_yr, 0.56);
                    stellar_term = pow(1.0 + pow(stellar / M0, -2.1), -0.148);
                }
                return 1.23 * stellar_term * redshift_scaling;
            };
            met_acg = calc_met(sfr_acg, star_acg);
            if (p.use_mcgs) {
                met_mcg = calc_met(sfr_mcg, star_mcg);
            }
        }
        auto calc_lx_on_sfr = [&](double metallicity, double lx_const) -> double {
            if (p.use_metallicity) {
                return lx_const * p21_scaling_double_PL(metallicity, 0.0, 1.0, -0.64, 0.05);
            }
            return lx_const;
        };
        double mu_x = calc_lx_on_sfr(met_acg, p.lx_over_sfr_acg) * sfr_acg;
        if (p.use_mcgs) {
            mu_x += calc_lx_on_sfr(met_mcg, p.lx_over_sfr_mcg) * sfr_mcg;
        }
        double stoc_xray_adj = p.scaling_median ? 0.0 : (p.sigma_xray * p.sigma_xray * 0.5);
        xray_lum = mu_x * exp(xray_rng * p.sigma_xray - stoc_xray_adj);
    }

    double fesc_acg = fmin(p.fesc_10_acg * pow(halo_mass / 1e10, p.alpha_esc), 1.0);
    double fesc_mcg = 0.0;
    if (p.use_mcgs) {
        fesc_mcg = fmin(p.fesc_7_mcg * pow(halo_mass / 1e7, p.alpha_esc), 1.0);
    }

    out->halo_mass = halo_mass;
    out->stellar_mass_acg = star_acg;
    out->stellar_mass_mcg = star_mcg;
    out->sfr_acg = sfr_acg;
    out->sfr_mcg = sfr_mcg;
    out->n_ion = (star_acg * p.pop2_ion * fesc_acg + star_mcg * p.pop3_ion * fesc_mcg) /
                 p.rhocrit_omb;
    out->fesc_weighted_sfr = sfr_acg * p.pop2_ion * fesc_acg + sfr_mcg * p.pop3_ion * fesc_mcg;
    out->metallicity_acg = met_acg;
    out->metallicity_mcg = met_mcg;
    out->xray_luminosity = xray_lum;
}

struct P21DevVelFields {
    const float *v[3];
    const float *v2[3];
    int dim[3];
    bool is_2lpt;
};

static inline bool p21_upload_vel_fields(InitialConditions *ini_boxes, bool on_hires,
                                         cudaStream_t stream, P21DevVelFields *out) {
    memset(out, 0, sizeof(*out));
    if (!ini_boxes) {
        return false;
    }
    int nx = on_hires ? simulation_options_global->DIM : simulation_options_global->HII_DIM;
    int nz = on_hires ? D_PARA : HII_D_PARA;
    out->dim[0] = nx;
    out->dim[1] = nx;
    out->dim[2] = nz;
    out->is_2lpt = (matter_options_global->PERTURB_ALGORITHM == PERTURB_ALGORITHM_2LPT);

    size_t nbytes = (size_t)nx * (size_t)nx * (size_t)nz * sizeof(float);
    const float *hv[3] = {
        on_hires ? ini_boxes->hires_vx : ini_boxes->lowres_vx,
        on_hires ? ini_boxes->hires_vy : ini_boxes->lowres_vy,
        on_hires ? ini_boxes->hires_vz : ini_boxes->lowres_vz,
    };
    const float *hv2[3] = {
        on_hires ? ini_boxes->hires_vx_2LPT : ini_boxes->lowres_vx_2LPT,
        on_hires ? ini_boxes->hires_vy_2LPT : ini_boxes->lowres_vy_2LPT,
        on_hires ? ini_boxes->hires_vz_2LPT : ini_boxes->lowres_vz_2LPT,
    };

    for (int axis = 0; axis < 3; ++axis) {
        if (!hv[axis]) {
            return false;
        }
        out->v[axis] = (const float *)p21_cuda_registry_get_or_upload(hv[axis], nbytes, stream);
        if (!out->v[axis]) {
            return false;
        }
        if (out->is_2lpt) {
            if (!hv2[axis]) {
                return false;
            }
            out->v2[axis] =
                (const float *)p21_cuda_registry_get_or_upload(hv2[axis], nbytes, stream);
            if (!out->v2[axis]) {
                return false;
            }
        }
    }
    return true;
}

#endif /* P21_CUDA_HALO_PROPS_CUH */
