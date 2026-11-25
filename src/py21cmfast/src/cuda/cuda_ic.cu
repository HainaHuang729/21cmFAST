#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "cuda_common.cuh"
#include "cuda_rng.cuh"

namespace {

struct CosmoKernelParams {
    int power_spectrum;
    int v_cb_model;
    double sound_horizon;
    double alpha_nu;
    double beta_c;
    double omhh;
    double f_nu;
    double theta_cmb;
    double sigma_norm;
    double eh_ratio_at_kmax;
    double ps_norm;
    double power_index;
    double omm;
    double omb;
    double hlittle;
    double c_kms;
    const double *class_dens_table;
    int class_dens_rows;
    const double *class_vcb_table;
    int class_vcb_rows;
};

struct HostSpline {
    std::vector<double> packed; /* [5 * n]: x, y, b, c, d */
    int n = 0;
};

/* Natural cubic spline coefficients matching gsl_interp_cspline. */
bool build_natural_cubic_spline(const double *x, const double *y, int n, HostSpline *out) {
    if (!x || !y || n < 3) {
        return false;
    }
    std::vector<double> h(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        h[i] = x[i + 1] - x[i];
        if (h[i] <= 0.0) {
            return false;
        }
    }

    /* Solve tridiagonal system for c[1..n-2] (c[0] = c[n-1] = 0). */
    std::vector<double> c(n, 0.0);
    std::vector<double> cp(n, 0.0);
    std::vector<double> dp(n, 0.0);
    for (int i = 1; i < n - 1; ++i) {
        double a_i = h[i - 1];
        double b_i = 2.0 * (h[i - 1] + h[i]);
        double c_i = (i + 1 < n - 1) ? h[i] : 0.0;
        double r_i = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
        double denom = b_i - a_i * cp[i - 1];
        cp[i] = c_i / denom;
        dp[i] = (r_i - a_i * dp[i - 1]) / denom;
    }
    for (int i = n - 2; i >= 1; --i) {
        c[i] = dp[i] - cp[i] * c[i + 1];
    }

    out->n = n;
    out->packed.assign((size_t)5 * n, 0.0);
    for (int i = 0; i < n; ++i) {
        out->packed[i] = x[i];
        out->packed[n + i] = y[i];
        out->packed[3 * n + i] = c[i];
    }
    for (int i = 0; i < n - 1; ++i) {
        out->packed[2 * n + i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0;
        out->packed[4 * n + i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
    }
    return true;
}

__device__ static inline double dev_eval_spline(const double *table, int n, double k) {
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (k < table[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    double dx = k - table[lo];
    return table[n + lo] +
           dx * (table[2 * n + lo] + dx * (table[3 * n + lo] + dx * table[4 * n + lo]));
}

__device__ static inline double dev_transfer_eh(double k, const CosmoKernelParams &p) {
    const double n_nu = 1.0;
    double q = k * (p.theta_cmb * p.theta_cmb) / p.omhh;
    double sqrt_a = sqrt(p.alpha_nu);
    double gamma_eff =
        sqrt_a + (1.0 - sqrt_a) / (1.0 + pow(0.43 * k * p.sound_horizon, 4.0));
    double q_eff = q / gamma_eff;
    double tf = log(M_E + 1.84 * p.beta_c * sqrt_a * q_eff);
    tf /= tf + q_eff * q_eff * (14.4 + 325.0 / (1.0 + 60.5 * pow(q_eff, 1.11)));
    double q_nu = 3.92 * q / sqrt(p.f_nu / n_nu);
    tf *= 1.0 + (1.2 * pow(p.f_nu, 0.64) * pow(n_nu, 0.3 + 0.6 * p.f_nu)) /
                    (pow(q_nu, -1.6) + pow(q_nu, 0.8));
    return tf;
}

__device__ static inline double dev_transfer_function(double k, const CosmoKernelParams &p) {
    switch (p.power_spectrum) {
        case POWER_SPECTRUM_EH:
            return dev_transfer_eh(k, p);
        case POWER_SPECTRUM_BBKS: {
            double gamma = p.omm * p.hlittle * exp(-p.omb - (p.omb / p.omm));
            double q = k / (p.hlittle * gamma);
            return (log(1.0 + 2.34 * q) / (2.34 * q)) *
                   pow(1.0 + 3.89 * q + pow(16.1 * q, 2.0) + pow(5.46 * q, 3.0) +
                           pow(6.71 * q, 4.0),
                       -0.25);
        }
        case POWER_SPECTRUM_EFSTATHIOU: {
            double gamma = p.omm * p.hlittle * p.hlittle;
            double aa = 6.4 / gamma;
            double bb = 3.0 / gamma;
            double cc = 1.7 / gamma;
            double nu = 1.13;
            return pow(1.0 + pow(aa * k + pow(bb * k, 1.5) + cc * cc * k * k, nu), -1.0 / nu);
        }
        case POWER_SPECTRUM_PEEBLES: {
            double gamma = p.omm * p.hlittle * exp(-p.omb - (p.omb / p.omm));
            double aa = 8.0 / (p.hlittle * gamma);
            double bb = 4.7 / pow(p.hlittle * gamma, 2.0);
            return 1.0 + aa * k + bb * k * k;
        }
        case POWER_SPECTRUM_WHITE: {
            double gamma = p.omm * p.hlittle * p.hlittle * exp(-p.omb - (p.omb / p.omm));
            double aa = 1.7 / gamma;
            double bb = 9.0 / pow(gamma, 1.5);
            double cc = 1.0 / (gamma * gamma);
            return 139.284 / (1.0 + aa * k + bb * pow(k, 1.5) + cc * k * k);
        }
        case POWER_SPECTRUM_CLASS: {
            int n = p.class_dens_rows;
            const double *tbl = p.class_dens_table;
            if (k > tbl[n - 1]) {
                return p.eh_ratio_at_kmax * dev_transfer_eh(k, p) * k * k;
            }
            return dev_eval_spline(tbl, n, k);
        }
        default:
            return 0.0;
    }
}

__device__ static inline double dev_power_in_k(double k, const CosmoKernelParams &p) {
    if (k == 0.0) {
        return 0.0;
    }
    double tf = dev_transfer_function(k, p);
    if (p.power_spectrum < POWER_SPECTRUM_CLASS) {
        tf *= k * k;
    }
    double primordial = p.ps_norm * pow(k / 0.05, p.power_index - 1.0);
    double pk = p.sigma_norm * primordial * tf * tf / (k * k * k);
    if (p.power_spectrum == POWER_SPECTRUM_CLASS && p.v_cb_model != V_CB_MODEL_NO) {
        double lr = log(k / 300.0);
        pk *= 1.0 - 0.24 * exp(-(lr * lr) / (2.0 * 0.9 * 0.9));
    }
    return pk;
}

__device__ static inline double dev_power_in_vcb(double k, const CosmoKernelParams &p) {
    if (k == 0.0) {
        return 0.0;
    }
    int n = p.class_vcb_rows;
    const double *tbl = p.class_vcb_table;
    double tv = 0.0;
    if (k > tbl[n - 1]) {
        tv = exp(log(tbl[2 * n - 1]) +
                 (log(tbl[2 * n - 1]) - log(tbl[2 * n - 2])) /
                     (log(tbl[n - 1]) - log(tbl[n - 2])) * (log(k) - log(tbl[n - 1])));
    } else {
        tv = dev_eval_spline(tbl, n, k);
    }
    double primordial = p.ps_norm * pow(k / 0.05, p.power_index - 1.0);
    return p.sigma_norm * primordial * tv * tv / (k * k * k);
}

__global__ void scale_real_kernel(const float *src, float *dst, size_t n, double factor) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = (float)((double)src[idx] * factor);
    }
}

__global__ void fill_ic_modes_slice_kernel(cufftComplex *box, const double *gauss, int first_x,
                                           int count_x, int ny, int nz, double box_len_x,
                                           double box_len_y, double box_len_z, double volume,
                                           CosmoKernelParams params) {
    int hz = nz / 2 + 1;
    size_t local_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t slice_modes = (size_t)count_x * ny * hz;
    if (local_idx >= slice_modes) {
        return;
    }

    int n_z = (int)(local_idx % hz);
    size_t q = local_idx / hz;
    int n_y = (int)(q % ny);
    int n_x = first_x + (int)(q / ny);

    double kx = p21_index_to_k(n_x, box_len_x, ny); /* hi_dim[0] == hi_dim[1] == DIM */
    double ky = p21_index_to_k(n_y, box_len_y, ny);
    double kz = p21_index_to_k(n_z, box_len_z, nz);
    double k_mag = sqrt(kx * kx + ky * ky + kz * kz);
    double p = dev_power_in_k(k_mag, params);
    double amp = sqrt(volume * p / 2.0);
    double a = gauss[2 * local_idx];
    double b = gauss[2 * local_idx + 1];

    size_t out_idx = ((size_t)n_x * ny + (size_t)n_y) * hz + (size_t)n_z;
    box[out_idx] = make_float2((float)(amp * a), (float)(amp * b));
}

__global__ void enforce_real_planes_general_kernel(cufftComplex *modes, int nx, int ny, int nz) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane_sz = (size_t)nx * ny;
    if (index >= 2 * plane_sz) {
        return;
    }
    int mx = nx / 2;
    int my = ny / 2;
    int mz = nz / 2;
    int hz = nz / 2 + 1;
    int plane_k = (int)(index % 2);
    if (plane_k && mz == 0) {
        return;
    }
    int k = plane_k ? mz : 0;
    size_t q = index / 2;
    int i = (int)(q / ny);
    int j = (int)(q % ny);
    size_t pos = ((size_t)i * ny + (size_t)j) * hz + (size_t)k;

    if (i == 0 && j == 0 && k == 0) {
        modes[0] = make_float2(0.0f, 0.0f);
    } else if ((i == 0 || i == mx) && (j == 0 || j == my)) {
        modes[pos].y = 0.0f;
    } else if (i > 0 && i < mx) {
        if (j == 0 || j == my) {
            cufftComplex v = modes[((size_t)(nx - i) * ny + (size_t)j) * hz + (size_t)k];
            modes[pos] = make_float2(v.x, -v.y);
        } else if (j < my || j > ny - my) {
            int sy = ny - j;
            cufftComplex v = modes[((size_t)(nx - i) * ny + (size_t)sy) * hz + (size_t)k];
            modes[pos] = make_float2(v.x, -v.y);
        }
    } else if ((i == 0 || i == mx) && j > 0 && j < my) {
        cufftComplex v = modes[((size_t)i * ny + (size_t)(ny - j)) * hz + (size_t)k];
        modes[pos] = make_float2(v.x, -v.y);
    }
}

__global__ void resample_hi_to_out_kernel(const float *hi_real, int nx, int ny, int nz,
                                          float *out_real, int ox, int oy, int oz,
                                          double dim_ratio, double volume, bool divide_vol,
                                          bool accum_sq) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)ox * oy * oz;
    if (idx >= total) {
        return;
    }
    int k = (int)(idx % oz);
    size_t q = idx / oz;
    int j = (int)(q % oy);
    int i = (int)(q / oy);

    int ri = p21_resample_coord(i, dim_ratio);
    int rj = p21_resample_coord(j, dim_ratio);
    int rk = p21_resample_coord(k, dim_ratio);
    float val = hi_real[p21_grid_idx(ri, rj, rk, ny, nz)];

    if (accum_sq) {
        double dv = (double)val;
        out_real[idx] += (float)(dv * dv);
    } else if (divide_vol) {
        out_real[idx] = (float)((double)val / volume);
    } else {
        out_real[idx] = val;
    }
}

__global__ void vcb_kspace_axis_kernel(const cufftComplex *saved, cufftComplex *out, int nx, int ny,
                                       int nz, double len_x, double len_y, double len_z, int axis,
                                       CosmoKernelParams params) {
    int hz = nz / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nx * ny * hz;
    if (idx >= total) {
        return;
    }
    if (idx == 0) {
        out[0] = make_float2(0.0f, 0.0f);
        return;
    }
    int n_z = (int)(idx % hz);
    size_t q = idx / hz;
    int n_y = (int)(q % ny);
    int n_x = (int)(q / ny);

    double kx = p21_index_to_k(n_x, len_x, nx);
    double ky = p21_index_to_k(n_y, len_y, ny);
    double kz = p21_index_to_k(n_z, len_z, nz);
    double k_mag = sqrt(kx * kx + ky * ky + kz * kz);
    double p = dev_power_in_k(k_mag, params);
    double p_vcb = dev_power_in_vcb(k_mag, params);
    double k_comp = (axis == 0) ? kx : ((axis == 1) ? ky : kz);
    double factor = (k_comp / k_mag) * sqrt(p_vcb / p) * params.c_kms;

    cufftComplex v = saved[idx];
    out[idx] = make_float2((float)(-(double)v.y * factor), (float)((double)v.x * factor));
}

__global__ void vcb_finalize_kernel(float *vcb, size_t n, double volume) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        vcb[idx] = (float)(sqrt((double)vcb[idx]) / volume);
    }
}

__global__ void f_gradient_kernel(const cufftComplex *in, cufftComplex *out, int nx, int ny, int nz,
                                  double len_x, double len_y, double len_z, int axis) {
    int hz = nz / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nx * ny * hz;
    if (idx >= total) {
        return;
    }
    if (idx == 0) {
        out[0] = make_float2(0.0f, 0.0f);
        return;
    }
    int n_z = (int)(idx % hz);
    size_t q = idx / hz;
    int n_y = (int)(q % ny);
    int n_x = (int)(q / ny);

    double kx = p21_index_to_k(n_x, len_x, nx);
    double ky = p21_index_to_k(n_y, len_y, ny);
    double kz = p21_index_to_k(n_z, len_z, nz);
    double k_sq = kx * kx + ky * ky + kz * kz;
    double k_comp = (axis == 0) ? kx : ((axis == 1) ? ky : kz);
    double factor = k_comp / k_sq;

    cufftComplex v = in[idx];
    out[idx] = make_float2((float)(-(double)v.y * factor), (float)((double)v.x * factor));
}

__global__ void f_laplacian_kernel(const cufftComplex *in, cufftComplex *out, int nx, int ny, int nz,
                                   double len_x, double len_y, double len_z, int ax0, int ax1) {
    int hz = nz / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nx * ny * hz;
    if (idx >= total) {
        return;
    }
    if (idx == 0) {
        out[0] = make_float2(0.0f, 0.0f);
        return;
    }
    int n_z = (int)(idx % hz);
    size_t q = idx / hz;
    int n_y = (int)(q % ny);
    int n_x = (int)(q / ny);

    double kx = p21_index_to_k(n_x, len_x, nx);
    double ky = p21_index_to_k(n_y, len_y, ny);
    double kz = p21_index_to_k(n_z, len_z, nz);
    double k_sq = kx * kx + ky * ky + kz * kz;
    double k0 = (ax0 == 0) ? kx : ((ax0 == 1) ? ky : kz);
    double k1 = (ax1 == 0) ? kx : ((ax1 == 1) ? ky : kz);
    double factor = -(k0 * k1) / k_sq;

    cufftComplex v = in[idx];
    out[idx] = make_float2((float)((double)v.x * factor), (float)((double)v.y * factor));
}

__global__ void accum_2lpt_rhs_kernel(float *rhs, const float *diag_i, const float *diag_j,
                                      const float *phi_ij, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double c_ii = (double)diag_i[idx];
        double c_jj = (double)diag_j[idx];
        double c_ij = (double)phi_ij[idx];
        float acc = rhs[idx];
        acc += (float)(c_ii * c_jj);
        acc -= (float)(c_ij * c_ij);
        rhs[idx] = acc;
    }
}

bool upload_and_register_d2h(float *host_ptr, const float *d_src, size_t nbytes,
                             cudaStream_t stream) {
    if (!host_ptr || !d_src || nbytes == 0) {
        return false;
    }
    if (!P21_CUDA_OK(cudaMemcpyAsync(host_ptr, d_src, nbytes, cudaMemcpyDeviceToHost, stream)) ||
        !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        return false;
    }
    return p21_cuda_registry_put_copy(host_ptr, d_src, nbytes, stream);
}

}  // namespace

extern "C" int p21_cuda_compute_initial_conditions(unsigned long long int random_seed,
                                                   InitialConditions *boxes) {
    if (!boxes || !p21_cuda_get_enabled() || !p21_cuda_ensure_init()) {
        return -1;
    }

    const int nx = simulation_options_global->DIM;
    const int ny = simulation_options_global->DIM;
    const int nz = D_PARA;
    const int lx = simulation_options_global->HII_DIM;
    const int ly = simulation_options_global->HII_DIM;
    const int lz = HII_D_PARA;
    const int n_threads = simulation_options_global->N_THREADS;

    if (nx < 2 || ny < 2 || nz < 2 || lx < 2 || ly < 2 || lz < 2 || n_threads < 1) {
        return -1;
    }
    if (matter_options_global->POWER_SPECTRUM < 0 ||
        matter_options_global->POWER_SPECTRUM > POWER_SPECTRUM_CLASS) {
        return -1;
    }
    if (matter_options_global->POWER_SPECTRUM == POWER_SPECTRUM_CLASS) {
        if (!cosmo_tables_global || !cosmo_tables_global->transfer_density ||
            cosmo_tables_global->transfer_density->size < 3) {
            return -1;
        }
        if (matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS &&
            (!cosmo_tables_global->transfer_vcb || cosmo_tables_global->transfer_vcb->size < 3)) {
            return -1;
        }
    }

    const double box_len_x = simulation_options_global->BOX_LEN;
    const double box_len_y = simulation_options_global->BOX_LEN;
    const double ncf = simulation_options_global->NON_CUBIC_FACTOR;
    const double box_len_z = ncf * box_len_x;
    const double volume = box_len_x * box_len_y * box_len_z;
    const double dim_ratio_hi_lo = nx / (double)lx;
    const float tophat_R = (float)(physconst.l_factor * box_len_x / (lx + 0.0));

    const bool perturb_hires = matter_options_global->PERTURB_ON_HIGH_RES;
    const int pt_x = perturb_hires ? nx : lx;
    const int pt_y = perturb_hires ? ny : ly;
    const int pt_z = perturb_hires ? nz : lz;
    const double dim_ratio_hi_pt = nx / (double)pt_x;

    float *vel_pointers[3] = {
        perturb_hires ? boxes->hires_vx : boxes->lowres_vx,
        perturb_hires ? boxes->hires_vy : boxes->lowres_vy,
        perturb_hires ? boxes->hires_vz : boxes->lowres_vz,
    };
    float *vel_pointers_2lpt[3] = {
        perturb_hires ? boxes->hires_vx_2LPT : boxes->lowres_vx_2LPT,
        perturb_hires ? boxes->hires_vy_2LPT : boxes->lowres_vy_2LPT,
        perturb_hires ? boxes->hires_vz_2LPT : boxes->lowres_vz_2LPT,
    };

    const size_t hi_pixels = (size_t)nx * ny * nz;
    const size_t lo_pixels = (size_t)lx * ly * lz;
    const size_t pt_pixels = (size_t)pt_x * pt_y * pt_z;
    const size_t hi_k_pixels = (size_t)nx * ny * (nz / 2 + 1);
    const size_t hi_bytes = hi_pixels * sizeof(float);
    const size_t lo_bytes = lo_pixels * sizeof(float);
    const size_t pt_bytes = pt_pixels * sizeof(float);
    const size_t hi_k_bytes = hi_k_pixels * sizeof(cufftComplex);

    cudaStream_t stream = p21_cuda_stream();

    bool non_zero_input = false;
    for (size_t idx = 0; idx < hi_pixels; ++idx) {
        if (boxes->hires_density[idx] != 0.0f) {
            non_zero_input = true;
            break;
        }
    }

    cufftComplex *d_kspace_saved = (cufftComplex *)p21_cuda_pool_alloc(hi_k_bytes, stream);
    cufftComplex *d_kspace_work = (cufftComplex *)p21_cuda_pool_alloc(hi_k_bytes, stream);
    float *d_real_work = (float *)p21_cuda_pool_alloc(hi_bytes, stream);
    float *d_real_lo = (float *)p21_cuda_pool_alloc(lo_bytes, stream);
    float *d_real_pt = perturb_hires ? nullptr : d_real_lo;
    double *d_class_dens = nullptr;
    double *d_class_vcb = nullptr;
    double *d_gauss = nullptr;
    float *d_diag[3] = {nullptr, nullptr, nullptr};
    float *d_phi2_rhs = nullptr;

    bool ok = d_kspace_saved && d_kspace_work && d_real_work && d_real_lo;
    const int tpb = 256;

    double cosmo[9] = {};
    p21_get_cosmo_constants(cosmo);

    HostSpline dens_spline, vcb_spline;
    if (ok && matter_options_global->POWER_SPECTRUM == POWER_SPECTRUM_CLASS) {
        if (!build_natural_cubic_spline(cosmo_tables_global->transfer_density->x_values,
                                        cosmo_tables_global->transfer_density->y_values,
                                        cosmo_tables_global->transfer_density->size,
                                        &dens_spline)) {
            ok = false;
        } else {
            d_class_dens =
                (double *)p21_cuda_pool_alloc(dens_spline.packed.size() * sizeof(double), stream);
            if (!d_class_dens ||
                !P21_CUDA_OK(cudaMemcpyAsync(d_class_dens, dens_spline.packed.data(),
                                             dens_spline.packed.size() * sizeof(double),
                                             cudaMemcpyHostToDevice, stream))) {
                ok = false;
            }
        }
        if (ok && matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS) {
            if (!build_natural_cubic_spline(cosmo_tables_global->transfer_vcb->x_values,
                                            cosmo_tables_global->transfer_vcb->y_values,
                                            cosmo_tables_global->transfer_vcb->size,
                                            &vcb_spline)) {
                ok = false;
            } else {
                d_class_vcb = (double *)p21_cuda_pool_alloc(vcb_spline.packed.size() * sizeof(double),
                                                            stream);
                if (!d_class_vcb ||
                    !P21_CUDA_OK(cudaMemcpyAsync(d_class_vcb, vcb_spline.packed.data(),
                                                 vcb_spline.packed.size() * sizeof(double),
                                                 cudaMemcpyHostToDevice, stream))) {
                    ok = false;
                }
            }
        }
    }

    CosmoKernelParams cosmo_params = {
        matter_options_global->POWER_SPECTRUM,
        matter_options_global->V_CB_MODEL,
        cosmo[0],
        cosmo[1],
        cosmo[2],
        cosmo[3],
        cosmo[4],
        cosmo[6],
        cosmo[7],
        cosmo[8],
        cosmo_tables_global->ps_norm,
        cosmo_params_global->POWER_INDEX,
        cosmo_params_global->OMm,
        cosmo_params_global->OMb,
        cosmo_params_global->hlittle,
        physconst.c_kms,
        d_class_dens,
        dens_spline.n,
        d_class_vcb,
        vcb_spline.n,
    };

    if (ok && non_zero_input) {
        if (!P21_CUDA_OK(cudaMemcpyAsync(d_real_work, boxes->hires_density, hi_bytes,
                                         cudaMemcpyHostToDevice, stream)) ||
            !p21_cuda_registry_put_copy(boxes->hires_density, d_real_work, hi_bytes, stream)) {
            ok = false;
        }
        if (ok) {
            int g = (int)((hi_pixels + tpb - 1) / tpb);
            scale_real_kernel<<<g, tpb, 0, stream>>>(d_real_work, d_real_work, hi_pixels,
                                                     volume / (double)hi_pixels);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 p21_cuda_fft_r2c(d_real_work, d_kspace_saved, nx, ny, nz, false, stream);
        }
    } else if (ok) {
        std::vector<uint32_t> seeds((size_t)n_threads);
        if (!p21_cuda_seed_rng_threads(random_seed, n_threads, seeds.data(), stream)) {
            ok = false;
        }
        int max_slice_x = (nx + n_threads - 1) / n_threads;
        size_t max_slice_modes = (size_t)max_slice_x * ny * (nz / 2 + 1);
        d_gauss = (double *)p21_cuda_pool_alloc(max_slice_modes * 2 * sizeof(double), stream);
        if (!d_gauss) {
            ok = false;
        }

        for (int t = 0; ok && t < n_threads; ++t) {
            int q = nx / n_threads;
            int rem = nx % n_threads;
            int first = t * q + (t < rem ? t : rem);
            int count_x = q + (t < rem ? 1 : 0);
            if (count_x <= 0) {
                continue;
            }
            size_t slice_modes = (size_t)count_x * ny * (nz / 2 + 1);
            if (!p21_cuda_fill_ugaussian_thread(t % 5, seeds[(size_t)t], d_gauss, slice_modes * 2,
                                                stream)) {
                ok = false;
                break;
            }
            int g = (int)((slice_modes + tpb - 1) / tpb);
            fill_ic_modes_slice_kernel<<<g, tpb, 0, stream>>>(
                d_kspace_saved, d_gauss, first, count_x, ny, nz, box_len_x, box_len_y, box_len_z,
                volume, cosmo_params);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
        p21_cuda_pool_free(d_gauss, stream);
        d_gauss = nullptr;

        if (ok) {
            size_t plane_items = (size_t)2 * nx * ny;
            int pg = (int)((plane_items + tpb - 1) / tpb);
            enforce_real_planes_general_kernel<<<pg, tpb, 0, stream>>>(d_kspace_saved, nx, ny, nz);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 P21_CUDA_OK(cudaMemcpyAsync(d_kspace_work, d_kspace_saved, hi_k_bytes,
                                             cudaMemcpyDeviceToDevice, stream)) &&
                 p21_cuda_fft_c2r(d_kspace_work, d_real_work, nx, ny, nz, false, stream);
        }
        if (ok) {
            int g = (int)((hi_pixels + tpb - 1) / tpb);
            scale_real_kernel<<<g, tpb, 0, stream>>>(d_real_work, d_real_work, hi_pixels,
                                                     1.0 / volume);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 upload_and_register_d2h(boxes->hires_density, d_real_work, hi_bytes, stream);
        }
    }

    /* Filtered lowres density */
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(d_kspace_work, d_kspace_saved, hi_k_bytes,
                                         cudaMemcpyDeviceToDevice, stream));
        if (ok && nx != lx) {
            ok = p21_cuda_filter_box(d_kspace_work, nx, ny, nz, box_len_x, ncf, FILTER_TOPHAT,
                                     tophat_R, 0.0f, 0.0f, stream);
        }
        if (ok) {
            ok = p21_cuda_fft_c2r(d_kspace_work, d_real_work, nx, ny, nz, false, stream);
        }
        if (ok) {
            int lg = (int)((lo_pixels + tpb - 1) / tpb);
            resample_hi_to_out_kernel<<<lg, tpb, 0, stream>>>(
                d_real_work, nx, ny, nz, d_real_lo, lx, ly, lz, dim_ratio_hi_lo, volume, true,
                false);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 upload_and_register_d2h(boxes->lowres_density, d_real_lo, lo_bytes, stream);
        }
    }

    /* Relative velocity (CLASS FLUCTS) */
    if (ok && matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS) {
        ok = P21_CUDA_OK(cudaMemsetAsync(d_real_lo, 0, lo_bytes, stream));
        int kg = (int)((hi_k_pixels + tpb - 1) / tpb);
        int lg = (int)((lo_pixels + tpb - 1) / tpb);
        for (int ii = 0; ok && ii < 3; ++ii) {
            vcb_kspace_axis_kernel<<<kg, tpb, 0, stream>>>(d_kspace_saved, d_kspace_work, nx, ny,
                                                           nz, box_len_x, box_len_y, box_len_z, ii,
                                                           cosmo_params);
            ok = P21_CUDA_OK(cudaGetLastError());
            if (ok && nx != lx) {
                ok = p21_cuda_filter_box(d_kspace_work, nx, ny, nz, box_len_x, ncf, FILTER_TOPHAT,
                                         tophat_R, 0.0f, 0.0f, stream);
            }
            if (ok) {
                ok = p21_cuda_fft_c2r(d_kspace_work, d_real_work, nx, ny, nz, false, stream);
            }
            if (ok) {
                resample_hi_to_out_kernel<<<lg, tpb, 0, stream>>>(
                    d_real_work, nx, ny, nz, d_real_lo, lx, ly, lz, dim_ratio_hi_lo, volume, false,
                    true);
                ok = P21_CUDA_OK(cudaGetLastError());
            }
        }
        if (ok) {
            vcb_finalize_kernel<<<lg, tpb, 0, stream>>>(d_real_lo, lo_pixels, volume);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 upload_and_register_d2h(boxes->lowres_vcb, d_real_lo, lo_bytes, stream);
        }
    }

    /* 1LPT velocities */
    if (ok) {
        int kg = (int)((hi_k_pixels + tpb - 1) / tpb);
        int ptg = (int)((pt_pixels + tpb - 1) / tpb);
        for (int ii = 0; ok && ii < 3; ++ii) {
            f_gradient_kernel<<<kg, tpb, 0, stream>>>(d_kspace_saved, d_kspace_work, nx, ny, nz,
                                                      box_len_x, box_len_y, box_len_z, ii);
            ok = P21_CUDA_OK(cudaGetLastError());
            if (ok && !perturb_hires && nx != lx) {
                ok = p21_cuda_filter_box(d_kspace_work, nx, ny, nz, box_len_x, ncf, FILTER_TOPHAT,
                                         tophat_R, 0.0f, 0.0f, stream);
            }
            if (ok) {
                ok = p21_cuda_fft_c2r(d_kspace_work, d_real_work, nx, ny, nz, false, stream);
            }
            if (ok) {
                float *dst_dev = perturb_hires ? (float *)d_kspace_work : d_real_pt;
                if (perturb_hires) {
                    scale_real_kernel<<<ptg, tpb, 0, stream>>>(d_real_work, dst_dev, pt_pixels,
                                                             1.0 / volume);
                } else {
                    resample_hi_to_out_kernel<<<ptg, tpb, 0, stream>>>(
                        d_real_work, nx, ny, nz, dst_dev, pt_x, pt_y, pt_z, dim_ratio_hi_pt,
                        volume, true, false);
                }
                ok = P21_CUDA_OK(cudaGetLastError()) &&
                     upload_and_register_d2h(vel_pointers[ii], dst_dev, pt_bytes, stream);
            }
        }
    }

    /* 2LPT velocities */
    if (ok && matter_options_global->PERTURB_ALGORITHM == PERTURB_ALGORITHM_2LPT) {
        for (int a = 0; a < 3; ++a) {
            d_diag[a] = (float *)p21_cuda_pool_alloc(hi_bytes, stream);
            if (!d_diag[a]) {
                ok = false;
            }
        }
        d_phi2_rhs = (float *)p21_cuda_pool_alloc(hi_bytes, stream);
        if (!d_phi2_rhs || !P21_CUDA_OK(cudaMemsetAsync(d_phi2_rhs, 0, hi_bytes, stream))) {
            ok = false;
        }

        int kg = (int)((hi_k_pixels + tpb - 1) / tpb);
        int hg = (int)((hi_pixels + tpb - 1) / tpb);
        for (int a = 0; ok && a < 3; ++a) {
            f_laplacian_kernel<<<kg, tpb, 0, stream>>>(d_kspace_saved, d_kspace_work, nx, ny, nz,
                                                       box_len_x, box_len_y, box_len_z, a, a);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 p21_cuda_fft_c2r(d_kspace_work, d_diag[a], nx, ny, nz, false, stream);
        }

        if (ok && !perturb_hires) {
            float *hires_2lpt[3] = {boxes->hires_vx_2LPT, boxes->hires_vy_2LPT,
                                    boxes->hires_vz_2LPT};
            for (int a = 0; ok && a < 3; ++a) {
                if (hires_2lpt[a]) {
                    ok = upload_and_register_d2h(hires_2lpt[a], d_diag[a], hi_bytes, stream);
                }
            }
        }

        const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
        for (int p_idx = 0; ok && p_idx < 3; ++p_idx) {
            int a = pairs[p_idx][0];
            int b = pairs[p_idx][1];
            f_laplacian_kernel<<<kg, tpb, 0, stream>>>(d_kspace_saved, d_kspace_work, nx, ny, nz,
                                                       box_len_x, box_len_y, box_len_z, a, b);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 p21_cuda_fft_c2r(d_kspace_work, d_real_work, nx, ny, nz, false, stream);
            if (ok) {
                accum_2lpt_rhs_kernel<<<hg, tpb, 0, stream>>>(d_phi2_rhs, d_diag[a], d_diag[b],
                                                              d_real_work, hi_pixels);
                ok = P21_CUDA_OK(cudaGetLastError());
            }
        }

        if (ok) {
            double norm_2lpt = 1.0 / (volume * volume * (double)hi_pixels);
            scale_real_kernel<<<hg, tpb, 0, stream>>>(d_phi2_rhs, d_phi2_rhs, hi_pixels, norm_2lpt);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 p21_cuda_fft_r2c(d_phi2_rhs, d_kspace_saved, nx, ny, nz, false, stream);
        }

        int ptg = (int)((pt_pixels + tpb - 1) / tpb);
        for (int ii = 0; ok && ii < 3; ++ii) {
            f_gradient_kernel<<<kg, tpb, 0, stream>>>(d_kspace_saved, d_kspace_work, nx, ny, nz,
                                                      box_len_x, box_len_y, box_len_z, ii);
            ok = P21_CUDA_OK(cudaGetLastError());
            if (ok && !perturb_hires && nx != lx) {
                ok = p21_cuda_filter_box(d_kspace_work, nx, ny, nz, box_len_x, ncf, FILTER_TOPHAT,
                                         tophat_R, 0.0f, 0.0f, stream);
            }
            if (ok) {
                ok = p21_cuda_fft_c2r(d_kspace_work, d_real_work, nx, ny, nz, false, stream);
            }
            if (ok) {
                if (perturb_hires) {
                    ok = upload_and_register_d2h(vel_pointers_2lpt[ii], d_real_work, pt_bytes,
                                                 stream);
                } else {
                    resample_hi_to_out_kernel<<<ptg, tpb, 0, stream>>>(
                        d_real_work, nx, ny, nz, d_real_pt, pt_x, pt_y, pt_z, dim_ratio_hi_pt,
                        volume, false, false);
                    ok = P21_CUDA_OK(cudaGetLastError()) &&
                         upload_and_register_d2h(vel_pointers_2lpt[ii], d_real_pt, pt_bytes,
                                                 stream);
                }
            }
        }
    }

    p21_cuda_pool_free(d_phi2_rhs, stream);
    for (int a = 0; a < 3; ++a) {
        p21_cuda_pool_free(d_diag[a], stream);
    }
    p21_cuda_pool_free(d_class_vcb, stream);
    p21_cuda_pool_free(d_class_dens, stream);
    p21_cuda_pool_free(d_real_lo, stream);
    p21_cuda_pool_free(d_real_work, stream);
    p21_cuda_pool_free(d_kspace_work, stream);
    p21_cuda_pool_free(d_kspace_saved, stream);

    if (ok && P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        p21_cuda_stats_mut()->ic_runs++;
        return 0;
    }
    return -1;
}
