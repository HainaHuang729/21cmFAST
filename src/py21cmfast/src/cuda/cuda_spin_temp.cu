#include <math.h>
#include <mutex>

#include "cuda_common.cuh"

namespace {

constexpr int kBlockThreads = 256;
constexpr int kKappaPts = 30;
constexpr float kMaxTk = 5.0e4f;

__constant__ double c_kappa10_t[kKappaPts] = {
    0.0,           0.317597943861, 0.635195887722, 0.952793831584, 1.27039177544,
    1.58798971931, 1.90558766317,  2.22318560703,  2.54078355089,  2.85838149475,
    3.17597943861, 3.49357738247,  3.81117532633,  4.1287732702,   4.44637121406,
    4.76396915792, 5.08156710178,  5.39916504564,  5.7167629895,   6.03436093336,
    6.35195887722, 6.66955682109,  6.98715476495,  7.30475270881,  7.62235065267,
    7.93994859653, 8.25754654039,  8.57514448425,  8.89274242811,  9.21034037198,
};
__constant__ double c_kappa10_k[kKappaPts] = {
    -29.6115227098, -29.6228184691, -29.5917673123, -29.4469989515, -29.1171430989,
    -28.5382192456, -27.7424388865, -26.8137036254, -25.8749225449, -25.0548322235,
    -24.4005076336, -23.8952335377, -23.5075651004, -23.201637629,  -22.9593758343,
    -22.7534867461, -22.5745752086, -22.4195690855, -22.2833176123, -22.1549519419,
    -22.0323282988, -21.9149994721, -21.800121439,  -21.6839502137, -21.5662434981,
    -21.4473595491, -21.3279560712, -21.2067614838, -21.0835560288, -20.9627928675,
};

__constant__ double c_kappa_ph_t[kKappaPts] = {
    0.0,           0.341499570777, 0.682999141554, 1.02449871233, 1.36599828311,
    1.70749785389, 2.04899742466,  2.39049699544,  2.73199656622, 3.07349613699,
    3.41499570777, 3.75649527855,  4.09799484933,  4.4394944201,  4.78099399088,
    5.12249356166, 5.46399313243,  5.80549270321,  6.14699227399, 6.48849184477,
    6.82999141554, 7.17149098632,  7.5129905571,   7.85449012787, 8.19598969865,
    8.53748926943, 8.8789888402,   9.22048841098,  9.56198798176, 9.90348755254,
};
__constant__ double c_kappa_ph_k[kKappaPts] = {
    -21.6395565688, -21.5641675629, -21.5225112028, -21.5130514508, -21.5342522691,
    -21.5845293039, -21.6581396414, -21.7420392948, -21.8221380683, -21.8837908896,
    -21.9167553997, -21.9200173678, -21.8938574675, -21.8414464728, -21.7684762963,
    -21.6796222358, -21.5784701374, -21.4679438133, -21.3503236936, -21.2277666787,
    -21.1017425964, -20.9733966978, -20.8437244283, -20.7135746917, -20.583135408,
    -20.4523507819, -20.3215504736, -20.1917429161, -20.0629513946, -19.9343540344,
};

__constant__ double c_kappa_el_t[kKappaPts] = {
    0.0,           0.396997429827, 0.793994859653, 1.19099228948, 1.58798971931,
    1.98498714913, 2.38198457896,  2.77898200879,  3.17597943861, 3.57297686844,
    3.96997429827, 4.36697172809,  4.76396915792,  5.16096658775, 5.55796401757,
    5.9549614474,  6.35195887722,  6.74895630705,  7.14595373688, 7.5429511667,
    7.93994859653, 8.33694602636,  8.73394345618,  9.13094088601, 9.52793831584,
    9.92493574566, 10.3219331755,  10.7189306053,  11.1159280351, 11.512925465,
};
__constant__ double c_kappa_el_k[kKappaPts] = {
    -22.1549007191, -21.9576919899, -21.760758435,  -21.5641795674, -21.3680349001,
    -21.1724124486, -20.9774403051, -20.78327367,   -20.5901042551, -20.3981934669,
    -20.2078762485, -20.0195787458, -19.8339587914, -19.6518934427, -19.4745894649,
    -19.3043925781, -19.1444129787, -18.9986014565, -18.8720602784, -18.768679825,
    -18.6909581885, -18.6387511068, -18.6093755705, -18.5992098958, -18.6050625357,
    -18.6319366207, -18.7017996535, -18.8477153986, -19.0813436512, -19.408859606,
};

std::mutex g_lya_mu;
double *g_d_dEC = nullptr;
double *g_d_dEI = nullptr;
int g_lya_nT = 0;
int g_lya_ngp = 0;
int g_lya_gen = -1;

struct SpinTempParams {
    float zp;
    double dzp;
    double Trad;
    double Trad_inv;
    double Ts_prefactor;
    double xa_tilde_prefactor;
    double xc_inverse;
    double dcomp_dzp_prefactor;
    double Nb_zp;
    double N_zp;
    double hubble_zp;
    double growth_zp;
    double dgrowth_dzp;
    double dt_dzp;
    double c21_tau_pref; /* 3*h_p*A10*c*lambda_21^2 / (32*pi*k_B) */
    double clumping_factor;
    double h_frac;
    double he_frac;
    double No_val;
    double Nb0_val;
    double hp_val;
    double kB_val;
    double mp_val;
    double c_cms_val;
    double A10_val;
    double T21_val;
    double lambda21_val;
    double nu_lya_val;
    double fract_float_err;
    int use_xray_heating;
    int use_cmb_heating;
    int use_lya_heating;
    int use_mcgs;
    const double *d_dEC;
    const double *d_dEI;
    int lya_nT;
    int lya_ngp;
};

__device__ static inline double p21_kappa_10(double TK) {
    constexpr double kBinWidth = 0.317597943861;
    constexpr double kInvBinWidth = 1.0 / 0.317597943861;
    double lk = log(TK);
    double ans;
    if (lk < c_kappa10_t[0]) {
        ans = c_kappa10_k[0];
    } else if (lk > c_kappa10_t[kKappaPts - 1]) {
        ans = log(exp(c_kappa10_k[kKappaPts - 1]) *
                  pow(exp(lk) / exp(c_kappa10_t[kKappaPts - 1]), 0.381));
    } else {
        int idx = (int)floor((lk - c_kappa10_t[0]) * kInvBinWidth);
        if (idx >= kKappaPts - 1) idx = kKappaPts - 2;
        ans = c_kappa10_k[idx] +
              (lk - (c_kappa10_t[0] + kBinWidth * (float)idx)) *
                  (c_kappa10_k[idx + 1] - c_kappa10_k[idx]) * kInvBinWidth;
    }
    return exp(ans);
}

__device__ static inline double p21_kappa_10_pH(double T) {
    constexpr double kBinWidth = 0.341499570777;
    constexpr double kInvBinWidth = 1.0 / 0.341499570777;
    double lk = log(T);
    double ans;
    if (lk < c_kappa_ph_t[0]) {
        ans = c_kappa_ph_k[0];
    } else if (lk > c_kappa_ph_t[kKappaPts - 1]) {
        ans = c_kappa_ph_k[kKappaPts - 1] +
              ((c_kappa_ph_k[kKappaPts - 1] - c_kappa_ph_k[kKappaPts - 2]) /
               (c_kappa_ph_t[kKappaPts - 1] - c_kappa_ph_t[kKappaPts - 2]) *
               (lk - c_kappa_ph_t[kKappaPts - 1]));
    } else {
        int idx = (int)floor((lk - c_kappa_ph_t[0]) * kInvBinWidth);
        if (idx >= kKappaPts - 1) idx = kKappaPts - 2;
        ans = c_kappa_ph_k[idx] +
              (lk - (c_kappa_ph_t[0] + kBinWidth * (double)idx)) *
                  (c_kappa_ph_k[idx + 1] - c_kappa_ph_k[idx]) * kInvBinWidth;
    }
    return exp(ans);
}

__device__ static inline double p21_kappa_10_elec(double T) {
    constexpr double kBinWidth = 0.396997429827;
    constexpr double kInvBinWidth = 1.0 / 0.396997429827;
    double lk = log(T);
    double ans;
    if (lk < c_kappa_el_t[0]) {
        ans = c_kappa_el_k[0];
    } else if (lk > c_kappa_el_t[kKappaPts - 1]) {
        ans = c_kappa_el_k[kKappaPts - 1] +
              ((c_kappa_el_k[kKappaPts - 1] - c_kappa_el_k[kKappaPts - 2]) /
               (c_kappa_el_t[kKappaPts - 1] - c_kappa_el_t[kKappaPts - 2]) *
               (lk - c_kappa_el_t[kKappaPts - 1]));
    } else {
        int idx = (int)floor((lk - c_kappa_el_t[0]) * kInvBinWidth);
        if (idx >= kKappaPts - 1) idx = kKappaPts - 2;
        ans = c_kappa_el_k[idx] +
              (lk - (c_kappa_el_t[0] + kBinWidth * (float)idx)) *
                  (c_kappa_el_k[idx + 1] - c_kappa_el_k[idx]) * kInvBinWidth;
    }
    return exp(ans);
}

__device__ static inline double p21_alpha_A(double T) {
    double logT = log(T / 1.1604505e4);
    double l2 = logT * logT;
    double l3 = l2 * logT;
    double l4 = l2 * l2;
    double l5 = l4 * logT;
    double l6 = l3 * l3;
    double l7 = l6 * logT;
    double l8 = l4 * l4;
    double l9 = l8 * logT;
    return exp(-28.6130338 - 0.72411256 * logT - 2.02604473e-2 * l2 - 2.38086188e-3 * l3 -
               3.21260521e-4 * l4 - 1.42150291e-5 * l5 + 4.98910892e-6 * l6 +
               5.75561414e-7 * l7 - 1.85676704e-8 * l8 - 3.07113524e-9 * l9);
}

__device__ static inline int p21_find_nearest_point(double min_v, double max_v, int n, double val) {
    double dn = (max_v - min_v) / (n - 1);
    if (val <= (min_v + dn)) {
        return 0;
    }
    if (val >= max_v) {
        return n - 2;
    }
    int pos = (int)floor((val - min_v) / dn);
    if (pos < 0) return 0;
    if (pos > n - 2) return n - 2;
    return pos;
}

__device__ static inline double p21_interp_lya_heating(double tk, double ts, double taugp_val,
                                                       const double *arrE, int nT, int ngp) {
    constexpr double kTkMin = -1.0, kTkMax = 3.0;
    constexpr double kTsMin = -1.0, kTsMax = 3.0;
    constexpr double kTgpMin = 1.0, kTgpMax = 7.0;

    tk = fmin(fmax(log10(tk), kTkMin), kTkMax);
    ts = fmin(fmax(log10(ts), kTsMin), kTsMax);
    taugp_val = fmin(fmax(log10(taugp_val), kTgpMin), kTgpMax);

    int itk = p21_find_nearest_point(kTkMin, kTkMax, nT, tk);
    int its = p21_find_nearest_point(kTsMin, kTsMax, nT, ts);
    int itgp = p21_find_nearest_point(kTgpMin, kTgpMax, ngp, taugp_val);

    double dT = (kTkMax - kTkMin) / (nT - 1);
    double dS = (kTsMax - kTsMin) / (nT - 1);
    double dG = (kTgpMax - kTgpMin) / (ngp - 1);

    double x0 = kTkMin + itk * dT;
    double y0 = kTsMin + its * dS;
    double z0 = kTgpMin + itgp * dG;

    double xd = (tk - x0) / dT;
    double yd = (ts - y0) / dS;
    double zd = (taugp_val - z0) / dG;

    auto at = [&](int ix, int iy, int iz) -> double {
        return arrE[(ix * nT + iy) * ngp + iz];
    };

    double c00 = at(itk, its, itgp) * (1.0 - xd) + at(itk + 1, its, itgp) * xd;
    double c01 = at(itk, its, itgp + 1) * (1.0 - xd) + at(itk + 1, its, itgp + 1) * xd;
    double c10 = at(itk, its + 1, itgp) * (1.0 - xd) + at(itk + 1, its + 1, itgp) * xd;
    double c11 = at(itk, its + 1, itgp + 1) * (1.0 - xd) + at(itk + 1, its + 1, itgp + 1) * xd;

    double c0 = c00 * (1.0 - yd) + c10 * yd;
    double c1 = c01 * (1.0 - yd) + c11 * yd;
    return c0 * (1.0 - zd) + c1 * zd;
}

__global__ void init_first_ts_kernel(size_t n, const float *dens, float z, double xe, double TK,
                                     double cT_ad, double Trad, double xcoll_pref, double No_val,
                                     double Nb0_val, double z1_3, double kap_hi, double kap_el,
                                     double kap_ph, float *tk_out, float *xe_out, float *ts_out) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    (void)z;
    double gdens = (double)dens[idx];
    tk_out[idx] = (float)(TK * (1.0 + cT_ad * gdens));
    xe_out[idx] = (float)xe;

    double dens_term = z1_3 * (1.0 + gdens);
    double xc = xcoll_pref * dens_term *
                ((1.0 - xe) * No_val * kap_hi + xe * Nb0_val * kap_el + xe * No_val * kap_ph);
    double ts = (1.0 + xc) / (1.0 / Trad + xc / TK);
    ts_out[idx] = (float)ts;
}

__global__ void compute_ts_fast_kernel(
    size_t n, SpinTempParams p, const float *density, const double *xray_heat,
    const double *xray_ion, const double *xray_lya, const double *lya_cont_inj,
    const double *lya_cont, const double *lya_inj, const double *lyw, const float *prev_ts,
    const float *prev_tk, const float *prev_xe, float *ts_out, float *tk_out, float *xe_out,
    float *j21_out, int *nonfinite_flag) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    double curr_delta = (double)density[idx];
    if (curr_delta <= -1.0) {
        curr_delta = -1.0 + p.fract_float_err;
    }

    double r_prev_Ts = (double)prev_ts[idx];
    double r_prev_Tk = (double)prev_tk[idx];
    double r_prev_xe = (double)prev_xe[idx];

    double r_xheat = p.use_xray_heating ? xray_heat[idx] : 0.0;
    double r_xion = xray_ion[idx];
    double r_xlya = xray_lya[idx];
    double r_lyw = p.use_mcgs ? lyw[idx] : 0.0;

    double tau21 = p.c21_tau_pref * ((1.0 - r_prev_xe) * p.N_zp) / r_prev_Ts / p.hubble_zp;
    double xCMB = (tau21 > 1e-8)
                      ? ((1.0 - exp(-tau21)) / tau21)
                      : (1.0 - tau21 / 2.0 * (1.0 - tau21 / 3.0 * (1.0 - tau21 / 4.0)));

    double dxion_sink_dt = p21_alpha_A(r_prev_Tk) * p.clumping_factor * r_prev_xe * r_prev_xe *
                           p.h_frac * p.Nb_zp * (1.0 + curr_delta);
    double dxe_dzp = p.dt_dzp * (r_xion - dxion_sink_dt);

    double dadia_dzp = 3.0 / (1.0 + (double)p.zp);
    if (fabs(curr_delta) > p.fract_float_err) {
        dadia_dzp += p.dgrowth_dzp / (p.growth_zp * (1.0 / curr_delta + 1.0));
    }
    dadia_dzp *= (2.0 / 3.0) * r_prev_Tk;

    double dspec_dzp = -dxe_dzp * r_prev_Tk / (1.0 + r_prev_xe);
    double dcomp_dzp =
        p.dcomp_dzp_prefactor * (r_prev_xe / (1.0 + r_prev_xe + p.he_frac)) * (p.Trad - r_prev_Tk);
    double dxheat_dzp = p.use_xray_heating ? (r_xheat * p.dt_dzp) : 0.0;

    double dCMBheat_dzp = 0.0;
    if (p.use_cmb_heating) {
        double l2 = p.lambda21_val * p.lambda21_val;
        double eps_CMB = (3.0 / 4.0) * (p.Trad / p.T21_val) * p.A10_val * p.h_frac *
                         (p.hp_val * p.hp_val / l2 / p.mp_val) *
                         (1.0 + 2.0 * r_prev_Tk / p.T21_val);
        dCMBheat_dzp =
            -eps_CMB * (2.0 / 3.0 / p.kB_val / (1.0 + r_prev_xe)) / p.hubble_zp / (1.0 + p.zp);
    }

    double eps_Lya_cont = 0.0;
    double eps_Lya_inj = 0.0;
    double r_lya_c = 0.0, r_lya_i = 0.0, r_lya_ci = 0.0;
    if (p.use_lya_heating) {
        r_lya_c = lya_cont[idx];
        r_lya_i = lya_inj[idx];
        double zp1 = 1.0 + (double)p.zp;
        double tgp = 1.342881e-7 / p.hubble_zp * p.No_val * (zp1 * zp1 * zp1) *
                     (1.0 + curr_delta) * (1.0 - r_prev_xe);
        double E_cont =
            p21_interp_lya_heating(r_prev_Tk, r_prev_Ts, tgp, p.d_dEC, p.lya_nT, p.lya_ngp);
        double E_inj =
            p21_interp_lya_heating(r_prev_Tk, r_prev_Ts, tgp, p.d_dEI, p.lya_nT, p.lya_ngp);
        if (!isfinite(E_cont)) E_cont = 0.0;
        if (!isfinite(E_inj)) E_inj = 0.0;
        double ndot_pref =
            (4.0 * M_PI * p.nu_lya_val) / (p.Nb_zp * (1.0 + curr_delta)) / zp1 / p.c_cms_val;
        double norm_heat = 2.0 / 3.0 / p.kB_val / (1.0 + r_prev_xe);
        eps_Lya_cont = -(ndot_pref * r_lya_c) * E_cont * norm_heat;
        eps_Lya_inj = -(ndot_pref * r_lya_i) * E_inj * norm_heat;
    } else {
        r_lya_ci = lya_cont_inj[idx];
    }

    double x_e = r_prev_xe + dxe_dzp * p.dzp;
    if (x_e > 1.0) {
        x_e = 1.0 - p.fract_float_err;
    } else if (x_e < 0.0) {
        x_e = 0.0;
    }

    double Tk = r_prev_Tk;
    if (Tk < (double)kMaxTk) {
        Tk += (dxheat_dzp + dcomp_dzp + dspec_dzp + dadia_dzp + dCMBheat_dzp + eps_Lya_cont +
               eps_Lya_inj) *
              p.dzp;
    }
    if (Tk < 0.0) {
        Tk = p.Trad;
    }

    double J_alpha_tot = p.use_lya_heating ? (r_lya_c + r_lya_i + r_xlya) : (r_lya_ci + r_xlya);

    double T_inv = 1.0 / Tk;
    double T_inv_sq = T_inv * T_inv;
    double xc_fast = (1.0 + curr_delta) * p.xc_inverse *
                     ((1.0 - x_e) * p.No_val * p21_kappa_10(Tk) +
                      x_e * p.Nb0_val * p21_kappa_10_elec(Tk) +
                      x_e * p.No_val * p21_kappa_10_pH(Tk));
    double xi_power = p.Ts_prefactor * cbrt((1.0 + curr_delta) * (1.0 - x_e) * T_inv_sq);
    double xi2 = xi_power * xi_power;
    double xi3 = xi2 * xi_power;
    double xa_tilde_fast_arg =
        p.xa_tilde_prefactor * J_alpha_tot / (1.0 + 2.98394 * xi_power + 1.53583 * xi2 + 3.85289 * xi3);

    double TS_fast;
    if (J_alpha_tot > 1.0e-20) {
        TS_fast = p.Trad;
        double TSold_fast = 0.0;
        int iter = 0;
        while (fabs(TS_fast - TSold_fast) / TS_fast > 1.0e-3 && iter < 1000) {
            TSold_fast = TS_fast;
            double ts_inv = 1.0 / TS_fast;
            double xa_tilde_fast =
                (1.0 - 0.0631789 * T_inv + 0.115995 * T_inv_sq - 0.401403 * T_inv * ts_inv +
                 0.336463 * T_inv_sq * ts_inv) *
                xa_tilde_fast_arg;
            TS_fast = (xCMB + xa_tilde_fast + xc_fast) /
                      (xCMB * p.Trad_inv +
                       xa_tilde_fast * (T_inv + 0.405535 * T_inv * ts_inv - 0.405535 * T_inv_sq) +
                       xc_fast * T_inv);
            iter++;
        }
    } else {
        TS_fast = (xCMB + xc_fast) / (xCMB * p.Trad_inv + xc_fast * T_inv);
    }

    TS_fast = fabs(TS_fast);
    float ts_f = (float)TS_fast;
    if (!isfinite(ts_f)) {
        atomicOr(nonfinite_flag, 1);
    }
    ts_out[idx] = ts_f;
    tk_out[idx] = (float)Tk;
    xe_out[idx] = (float)x_e;
    if (p.use_mcgs) {
        j21_out[idx] = (float)r_lyw;
    }
}

bool ensure_lya_heating_uploaded(cudaStream_t stream, const double **d_ec_out,
                                 const double **d_ei_out, int *nT_out, int *ngp_out) {
    const double *h_ec = nullptr, *h_ei = nullptr;
    int nT = 0, ngp = 0, gen = 0;
    if (!p21_get_lya_heating_tables(&h_ec, &h_ei, &nT, &ngp, &gen)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_lya_mu);
    if (!g_d_dEC || !g_d_dEI || g_lya_gen != gen || g_lya_nT != nT || g_lya_ngp != ngp) {
        p21_cuda_pool_free(g_d_dEC, stream);
        p21_cuda_pool_free(g_d_dEI, stream);
        g_d_dEC = nullptr;
        g_d_dEI = nullptr;
        size_t bytes = (size_t)nT * (size_t)nT * (size_t)ngp * sizeof(double);
        g_d_dEC = (double *)p21_cuda_pool_alloc(bytes, stream);
        g_d_dEI = (double *)p21_cuda_pool_alloc(bytes, stream);
        if (!g_d_dEC || !g_d_dEI ||
            !P21_CUDA_OK(cudaMemcpyAsync(g_d_dEC, h_ec, bytes, cudaMemcpyHostToDevice, stream)) ||
            !P21_CUDA_OK(cudaMemcpyAsync(g_d_dEI, h_ei, bytes, cudaMemcpyHostToDevice, stream))) {
            p21_cuda_pool_free(g_d_dEC, stream);
            p21_cuda_pool_free(g_d_dEI, stream);
            g_d_dEC = nullptr;
            g_d_dEI = nullptr;
            g_lya_gen = -1;
            return false;
        }
        g_lya_nT = nT;
        g_lya_ngp = ngp;
        g_lya_gen = gen;
    }
    *d_ec_out = g_d_dEC;
    *d_ei_out = g_d_dEI;
    *nT_out = g_lya_nT;
    *ngp_out = g_lya_ngp;
    return true;
}

}  // namespace

void p21_cuda_spin_temp_cleanup(void) {
    cudaStream_t stream = p21_cuda_stream();
    std::lock_guard<std::mutex> lock(g_lya_mu);
    p21_cuda_pool_free(g_d_dEC, stream);
    p21_cuda_pool_free(g_d_dEI, stream);
    g_d_dEC = nullptr;
    g_d_dEI = nullptr;
    g_lya_nT = 0;
    g_lya_ngp = 0;
    g_lya_gen = -1;
}

extern "C" int p21_cuda_init_first_ts(TsBox *box, float *dens, float z, double xe_recfast,
                                      double tk_recfast, double cT_ad) {
    if (!box || !dens || !box->kinetic_temp_neutral || !box->xray_ionised_fraction ||
        !box->spin_temperature || !p21_cuda_get_enabled()) {
        return -1;
    }
    if (simulation_options_global->HII_DIM <= 1 || !p21_cuda_ensure_init()) {
        return -1;
    }

    size_t total = (size_t)simulation_options_global->HII_DIM *
                   (size_t)simulation_options_global->HII_DIM * (size_t)HII_D_PARA;
    size_t bytes = total * sizeof(float);
    cudaStream_t stream = p21_cuda_stream();

    const float *d_dens = (const float *)p21_cuda_registry_get_or_upload(dens, bytes, stream);
    if (!d_dens) {
        return -1;
    }

    float *d_tk = (float *)p21_cuda_pool_alloc(bytes, stream);
    float *d_xe = (float *)p21_cuda_pool_alloc(bytes, stream);
    float *d_ts = (float *)p21_cuda_pool_alloc(bytes, stream);
    if (!d_tk || !d_xe || !d_ts) {
        p21_cuda_pool_free(d_tk, stream);
        p21_cuda_pool_free(d_xe, stream);
        p21_cuda_pool_free(d_ts, stream);
        return -1;
    }

    double Trad = physconst.T_cmb * (1.0 + z);
    double xcoll_pref = physconst.T_21 / Trad / physconst.A10;
    double z1_3 = pow(1.0 + z, 3.0);
    double kap_hi = kappa_10(tk_recfast, 0);
    double kap_el = kappa_10_elec(tk_recfast, 0);
    double kap_ph = kappa_10_pH(tk_recfast, 0);

    int grid = (int)((total + kBlockThreads - 1) / kBlockThreads);
    init_first_ts_kernel<<<grid, kBlockThreads, 0, stream>>>(
        total, d_dens, z, xe_recfast, tk_recfast, cT_ad, Trad, xcoll_pref, No, N_b0, z1_3, kap_hi,
        kap_el, kap_ph, d_tk, d_xe, d_ts);
    bool ok = P21_CUDA_OK(cudaGetLastError()) &&
              P21_CUDA_OK(cudaMemcpyAsync(box->kinetic_temp_neutral, d_tk, bytes,
                                          cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaMemcpyAsync(box->xray_ionised_fraction, d_xe, bytes,
                                          cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaMemcpyAsync(box->spin_temperature, d_ts, bytes,
                                          cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaStreamSynchronize(stream));
    if (ok) {
        p21_cuda_stats_add_d2h(3 * bytes);
    }

    if (!ok) {
        p21_cuda_pool_free(d_tk, stream);
        p21_cuda_pool_free(d_xe, stream);
        p21_cuda_pool_free(d_ts, stream);
        return -1;
    }

    auto own_or_free = [&](const void *hptr, float *dbuf) {
        if (!p21_cuda_registry_put_owned(hptr, dbuf, bytes, stream)) {
            p21_cuda_pool_free(dbuf, stream);
        }
    };
    own_or_free(box->kinetic_temp_neutral, d_tk);
    own_or_free(box->xray_ionised_fraction, d_xe);
    own_or_free(box->spin_temperature, d_ts);

    p21_cuda_stats_mut()->spin_temp_runs++;
    return 0;
}

extern "C" int p21_cuda_compute_ts_box(float redshift, float prev_redshift,
                                       PerturbedField *perturbed_field,
                                       RadiationFields *radiation_fields, TsBox *previous_spin_temp,
                                       InitialConditions *ini_boxes, TsBox *this_spin_temp) {
    (void)ini_boxes;
    if (!perturbed_field || !radiation_fields || !previous_spin_temp || !this_spin_temp ||
        !p21_cuda_get_enabled()) {
        return -1;
    }
    if (simulation_options_global->HII_DIM <= 1 || !p21_cuda_ensure_init()) {
        return -1;
    }

    const size_t total = (size_t)simulation_options_global->HII_DIM *
                         (size_t)simulation_options_global->HII_DIM * (size_t)HII_D_PARA;
    const size_t fbytes = total * sizeof(float);
    const size_t dbytes = total * sizeof(double);
    const int use_xray_heat = astro_options_global->USE_X_RAY_HEATING ? 1 : 0;
    const int use_cmb_heating = astro_options_global->USE_CMB_HEATING ? 1 : 0;
    const int use_lya_heating = astro_options_global->USE_LYA_HEATING ? 1 : 0;
    const int use_mcgs = astro_options_global->USE_MCGS ? 1 : 0;
    cudaStream_t stream = p21_cuda_stream();

    const float *d_dens =
        (const float *)p21_cuda_registry_get_or_upload(perturbed_field->density, fbytes, stream);
    const float *d_prev_ts = (const float *)p21_cuda_registry_get_or_upload(
        previous_spin_temp->spin_temperature, fbytes, stream);
    const float *d_prev_tk = (const float *)p21_cuda_registry_get_or_upload(
        previous_spin_temp->kinetic_temp_neutral, fbytes, stream);
    const float *d_prev_xe = (const float *)p21_cuda_registry_get_or_upload(
        previous_spin_temp->xray_ionised_fraction, fbytes, stream);

    const double *d_xion = (const double *)p21_cuda_registry_get_or_upload(
        radiation_fields->xray_ionization_rate, dbytes, stream);
    const double *d_xlya = (const double *)p21_cuda_registry_get_or_upload(
        radiation_fields->xray_lya_flux, dbytes, stream);
    const double *d_xheat =
        use_xray_heat
            ? (const double *)p21_cuda_registry_get_or_upload(radiation_fields->xray_heating_rate,
                                                              dbytes, stream)
            : nullptr;
    const double *d_lyw =
        use_mcgs
            ? (const double *)p21_cuda_registry_get_or_upload(radiation_fields->lyw_flux, dbytes,
                                                              stream)
            : nullptr;
    const double *d_lya_c =
        use_lya_heating
            ? (const double *)p21_cuda_registry_get_or_upload(
                  radiation_fields->lya_flux_continuum, dbytes, stream)
            : nullptr;
    const double *d_lya_i =
        use_lya_heating
            ? (const double *)p21_cuda_registry_get_or_upload(radiation_fields->lya_flux_injected,
                                                              dbytes, stream)
            : nullptr;
    const double *d_lya_ci =
        !use_lya_heating
            ? (const double *)p21_cuda_registry_get_or_upload(
                  radiation_fields->lya_flux_continuum_injected, dbytes, stream)
            : nullptr;

    if (!d_dens || !d_prev_ts || !d_prev_tk || !d_prev_xe || !d_xion || !d_xlya ||
        (use_xray_heat && !d_xheat) || (use_mcgs && !d_lyw) ||
        (use_lya_heating ? (!d_lya_c || !d_lya_i) : !d_lya_ci)) {
        return -1;
    }

    const double *d_dEC = nullptr, *d_dEI = nullptr;
    int lya_nT = 0, lya_ngp = 0;
    if (use_lya_heating && !ensure_lya_heating_uploaded(stream, &d_dEC, &d_dEI, &lya_nT, &lya_ngp)) {
        return -1;
    }

    SpinTempParams p{};
    p.zp = redshift;
    p.dzp = (double)redshift - (double)prev_redshift;
    p.growth_zp = dicke(redshift);
    p.hubble_zp = hubble(redshift);
    p.dgrowth_dzp = ddicke_dz(redshift);
    p.dt_dzp = dtdz(redshift);
    p.Trad = physconst.T_cmb * (1.0 + redshift);
    p.Trad_inv = 1.0 / p.Trad;
    p.Ts_prefactor = pow(1.0e-7 * (1.342881e-7 / p.hubble_zp) * No * pow(1.0 + redshift, 3.0),
                         1.0 / 3.0);

    double gamma_alpha = physconst.f_alpha *
                         pow(physconst.nu_Ly_alpha * physconst.e_charge / (physconst.c_cms / 10.0),
                             2.0);
    gamma_alpha /=
        6.0 * (physconst.m_e / 1000.0) * pow(physconst.c_cms / 100.0, 3.0) * physconst.vac_perm;
    p.xa_tilde_prefactor =
        8.0 * M_PI * pow(physconst.lambda_Ly_alpha * 1.0e-8, 2.0) * gamma_alpha * physconst.T_21;
    p.xa_tilde_prefactor /= 9.0 * physconst.A10 * p.Trad;
    p.xc_inverse = pow(1.0 + redshift, 3.0) * physconst.T_21 / (p.Trad * physconst.A10);
    p.dcomp_dzp_prefactor = (-1.51e-4) / (p.hubble_zp / Ho) / (cosmo_params_global->hlittle) *
                            pow(p.Trad, 4.0) / (1.0 + redshift);
    p.Nb_zp = N_b0 * (1.0 + redshift) * (1.0 + redshift) * (1.0 + redshift);
    p.N_zp = No * (1.0 + redshift) * (1.0 + redshift) * (1.0 + redshift);
    p.c21_tau_pref = 3.0 * physconst.h_p * physconst.A10 * physconst.c_cms * physconst.lambda_21 *
                     physconst.lambda_21 / 32.0 / M_PI / physconst.k_B;
    p.clumping_factor = astro_params_global->CLUMPING_FACTOR;
    p.h_frac = H_FRAC;
    p.he_frac = HE_FRAC;
    p.No_val = No;
    p.Nb0_val = N_b0;
    p.hp_val = physconst.h_p;
    p.kB_val = physconst.k_B;
    p.mp_val = physconst.m_p;
    p.c_cms_val = physconst.c_cms;
    p.A10_val = physconst.A10;
    p.T21_val = physconst.T_21;
    p.lambda21_val = physconst.lambda_21;
    p.nu_lya_val = physconst.nu_Ly_alpha;
    p.fract_float_err = FRACT_FLOAT_ERR;
    p.use_xray_heating = use_xray_heat;
    p.use_cmb_heating = use_cmb_heating;
    p.use_lya_heating = use_lya_heating;
    p.use_mcgs = use_mcgs;
    p.d_dEC = d_dEC;
    p.d_dEI = d_dEI;
    p.lya_nT = lya_nT;
    p.lya_ngp = lya_ngp;

    float *d_ts = (float *)p21_cuda_pool_alloc(fbytes, stream);
    float *d_tk = (float *)p21_cuda_pool_alloc(fbytes, stream);
    float *d_xe = (float *)p21_cuda_pool_alloc(fbytes, stream);
    float *d_j21 = use_mcgs ? (float *)p21_cuda_pool_alloc(fbytes, stream) : nullptr;
    int *d_nonfinite = (int *)p21_cuda_pool_alloc(sizeof(int), stream);
    if (!d_ts || !d_tk || !d_xe || (use_mcgs && !d_j21) || !d_nonfinite) {
        p21_cuda_pool_free(d_ts, stream);
        p21_cuda_pool_free(d_tk, stream);
        p21_cuda_pool_free(d_xe, stream);
        p21_cuda_pool_free(d_j21, stream);
        p21_cuda_pool_free(d_nonfinite, stream);
        return -1;
    }

    this_spin_temp->Q_HI = radiation_fields->Q_HI;

    bool ok = P21_CUDA_OK(cudaMemsetAsync(d_nonfinite, 0, sizeof(int), stream));
    if (ok) {
        int grid = (int)((total + kBlockThreads - 1) / kBlockThreads);
        compute_ts_fast_kernel<<<grid, kBlockThreads, 0, stream>>>(
            total, p, d_dens, d_xheat, d_xion, d_xlya, d_lya_ci, d_lya_c, d_lya_i, d_lyw, d_prev_ts,
            d_prev_tk, d_prev_xe, d_ts, d_tk, d_xe, d_j21, d_nonfinite);
        ok = P21_CUDA_OK(cudaGetLastError());
    }

    int h_nonfinite = 0;
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(this_spin_temp->spin_temperature, d_ts, fbytes,
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(this_spin_temp->kinetic_temp_neutral, d_tk, fbytes,
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(this_spin_temp->xray_ionised_fraction, d_xe, fbytes,
                                         cudaMemcpyDeviceToHost, stream));
        if (ok) p21_cuda_stats_add_d2h(3 * fbytes);
    }
    if (ok && use_mcgs) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(this_spin_temp->J_21_LW, d_j21, fbytes,
                                         cudaMemcpyDeviceToHost, stream));
        if (ok) p21_cuda_stats_add_d2h(fbytes);
    }
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(&h_nonfinite, d_nonfinite, sizeof(int),
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }

    p21_cuda_pool_free(d_nonfinite, stream);
    if (!ok) {
        p21_cuda_pool_free(d_ts, stream);
        p21_cuda_pool_free(d_tk, stream);
        p21_cuda_pool_free(d_xe, stream);
        p21_cuda_pool_free(d_j21, stream);
        return -1;
    }
    if (h_nonfinite != 0) {
        p21_cuda_pool_free(d_ts, stream);
        p21_cuda_pool_free(d_tk, stream);
        p21_cuda_pool_free(d_xe, stream);
        p21_cuda_pool_free(d_j21, stream);
        return InfinityorNaNError;
    }

    auto own_or_free = [&](const void *hptr, float *dbuf) {
        if (!dbuf) return;
        if (!p21_cuda_registry_put_owned(hptr, dbuf, fbytes, stream)) {
            p21_cuda_pool_free(dbuf, stream);
        }
    };
    own_or_free(this_spin_temp->spin_temperature, d_ts);
    own_or_free(this_spin_temp->kinetic_temp_neutral, d_tk);
    own_or_free(this_spin_temp->xray_ionised_fraction, d_xe);
    if (use_mcgs) {
        own_or_free(this_spin_temp->J_21_LW, d_j21);
    }

    p21_cuda_stats_mut()->spin_temp_runs++;
    return 0;
}
