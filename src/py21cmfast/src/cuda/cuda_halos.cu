#include <math.h>
#include <string.h>

#include "cuda_common.cuh"
#include "cuda_halo_props.cuh"

namespace {

constexpr int kBlockThreads = 256;

__global__ void perturb_halos_and_props_kernel(
    size_t n_halos, P21HaloScalingParams hp, double boxlen, double boxlen_z, double vdf,
    double vdf2, int vel_nx, int vel_ny, int vel_nz, int lo_nx, int lo_ny, int lo_nz,
    int is_2lpt, int use_reion_fb, int uses_recomb, const float *vx, const float *vy,
    const float *vz, const float *vx2, const float *vy2, const float *vz2,
    const float *mturn_acg_grid, const float *mturn_mcg_grid, const float *in_masses,
    const float *in_coords, const float *in_star_rng, const float *in_sfr_rng,
    const float *in_xray_rng, float *out_coords, float *out_masses, float *out_star_acg,
    float *out_sfr_acg, float *out_nion, float *out_star_mcg, float *out_sfr_mcg,
    float *out_fesc_wsfr, float *out_xray) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_halos) {
        return;
    }

    const int vel_dim[3] = {vel_nx, vel_ny, vel_nz};
    const double box_size[3] = {boxlen, boxlen, boxlen_z};
    const double cell_size_inv_v = (double)vel_nx / boxlen;

    double pos[3] = {
        (double)in_coords[3 * idx + 0],
        (double)in_coords[3 * idx + 1],
        (double)in_coords[3 * idx + 2],
    };
    int ipos[3];
    p21_pos_to_index(pos, cell_size_inv_v, ipos);
    for (int axis = 0; axis < 3; ++axis) {
        ipos[axis] = p21_wrap_coord(ipos[axis], vel_dim[axis]);
    }
    size_t grid_idx = p21_grid_idx(ipos[0], ipos[1], ipos[2], vel_ny, vel_nz);

    pos[0] += (double)vx[grid_idx] * vdf;
    pos[1] += (double)vy[grid_idx] * vdf;
    pos[2] += (double)vz[grid_idx] * vdf;
    if (is_2lpt) {
        pos[0] -= (double)vx2[grid_idx] * vdf2;
        pos[1] -= (double)vy2[grid_idx] * vdf2;
        pos[2] -= (double)vz2[grid_idx] * vdf2;
    }
    p21_wrap_position(pos, box_size);

    float cx = (float)pos[0];
    float cy = (float)pos[1];
    float cz = (float)pos[2];
    out_coords[3 * idx + 0] = cx;
    out_coords[3 * idx + 1] = cy;
    out_coords[3 * idx + 2] = cz;

    double hmass = (double)in_masses[idx];
    if (hmass == 0.0) {
        out_masses[idx] = 0.0f;
        out_star_acg[idx] = 0.0f;
        out_sfr_acg[idx] = 0.0f;
        out_nion[idx] = 0.0f;
        if (hp.use_mcgs) {
            out_star_mcg[idx] = 0.0f;
            out_sfr_mcg[idx] = 0.0f;
        }
        if (uses_recomb) {
            out_fesc_wsfr[idx] = 0.0f;
        }
        if (hp.use_ts_fluct) {
            out_xray[idx] = 0.0f;
        }
        return;
    }

    double cell_len_lo = boxlen / (double)lo_nx;
    double halo_pos[3] = {
        (double)cx / cell_len_lo,
        (double)cy / cell_len_lo,
        (double)cz / cell_len_lo,
    };

    double M_turn_acg = hp.mturn_acg_homogeneous;
    double M_turn_mcg = 0.0;
    if (use_reion_fb) {
        M_turn_acg = pow(10.0, p21_cic_read_float(mturn_acg_grid, halo_pos, lo_nx, lo_ny, lo_nz));
    }
    if (hp.use_mcgs) {
        M_turn_mcg = pow(10.0, p21_cic_read_float(mturn_mcg_grid, halo_pos, lo_nx, lo_ny, lo_nz));
    }

    P21HaloPropsOut props{};
    p21_set_halo_properties_dev(hmass, M_turn_acg, M_turn_mcg, hp, (double)in_star_rng[idx],
                                (double)in_sfr_rng[idx], (double)in_xray_rng[idx], &props);

    out_masses[idx] = (float)props.halo_mass;
    out_star_acg[idx] = (float)props.stellar_mass_acg;
    out_sfr_acg[idx] = (float)props.sfr_acg;
    out_nion[idx] = (float)props.n_ion;
    if (hp.use_mcgs) {
        out_star_mcg[idx] = (float)props.stellar_mass_mcg;
        out_sfr_mcg[idx] = (float)props.sfr_mcg;
    }
    if (uses_recomb) {
        out_fesc_wsfr[idx] = (float)props.fesc_weighted_sfr;
    }
    if (hp.use_ts_fluct) {
        out_xray[idx] = (float)props.xray_luminosity;
    }
}

}  // namespace

extern "C" int p21_cuda_compute_perturbed_halo_catalog(float redshift, InitialConditions *boxes,
                                                       TsBox *prev_ts, IonizedBox *prev_ion,
                                                       HaloCatalog *halos,
                                                       PerturbedHaloCatalog *halos_perturbed) {
    if (!p21_cuda_get_enabled() || !boxes || !halos || !halos_perturbed) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    halos_perturbed->n_halos = halos->n_halos;
    if (halos->n_halos == 0) {
        p21_cuda_stats_mut()->halo_perturb_runs++;
        return 0;
    }
    if (!halos->halo_masses || !halos->halo_coords || !halos->star_rng || !halos->sfr_rng ||
        !halos->xray_rng) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    const bool on_hires = matter_options_global->PERTURB_ON_HIGH_RES;
    P21DevVelFields velf{};
    if (!p21_upload_vel_fields(boxes, on_hires, stream, &velf)) {
        return -1;
    }

    ScalingConstants consts;
    set_scaling_constants(redshift, &consts, true);

    float *d_mturn_acg = nullptr, *d_mturn_mcg = nullptr;
    double mturn_avgs[2] = {0.0, 0.0};
    if (!p21_cuda_build_mturn_grids(&consts, boxes, prev_ts, prev_ion, stream, &d_mturn_acg,
                                    &d_mturn_mcg, mturn_avgs)) {
        return -1;
    }

    const size_t nh = (size_t)halos->n_halos;
    const size_t sbytes = nh * sizeof(float);
    const size_t cbytes = 3ULL * nh * sizeof(float);

    const float *d_in_hm =
        (const float *)p21_cuda_registry_get_or_upload(halos->halo_masses, sbytes, stream);
    const float *d_in_hc =
        (const float *)p21_cuda_registry_get_or_upload(halos->halo_coords, cbytes, stream);
    const float *d_in_sr =
        (const float *)p21_cuda_registry_get_or_upload(halos->star_rng, sbytes, stream);
    const float *d_in_sfrr =
        (const float *)p21_cuda_registry_get_or_upload(halos->sfr_rng, sbytes, stream);
    const float *d_in_xr =
        (const float *)p21_cuda_registry_get_or_upload(halos->xray_rng, sbytes, stream);

    const int use_reion_fb = astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK ? 1 : 0;
    const int use_mcgs = astro_options_global->USE_MCGS ? 1 : 0;
    const int uses_recomb = uses_recombination(astro_options_global->RECOMB_MODEL) ? 1 : 0;
    const int use_ts_fluct = astro_options_global->USE_TS_FLUCT ? 1 : 0;

    float *d_out_coords = (float *)p21_cuda_pool_alloc(cbytes, stream);
    float *d_out_hm = (float *)p21_cuda_pool_alloc(sbytes, stream);
    float *d_out_sa = (float *)p21_cuda_pool_alloc(sbytes, stream);
    float *d_out_sfa = (float *)p21_cuda_pool_alloc(sbytes, stream);
    float *d_out_nion = (float *)p21_cuda_pool_alloc(sbytes, stream);
    float *d_out_sm = use_mcgs ? (float *)p21_cuda_pool_alloc(sbytes, stream) : nullptr;
    float *d_out_sfm = use_mcgs ? (float *)p21_cuda_pool_alloc(sbytes, stream) : nullptr;
    float *d_out_fesc = uses_recomb ? (float *)p21_cuda_pool_alloc(sbytes, stream) : nullptr;
    float *d_out_xray = use_ts_fluct ? (float *)p21_cuda_pool_alloc(sbytes, stream) : nullptr;

    bool ok = d_in_hm && d_in_hc && d_in_sr && d_in_sfrr && d_in_xr && d_out_coords && d_out_hm &&
              d_out_sa && d_out_sfa && d_out_nion && (!use_mcgs || (d_out_sm && d_out_sfm)) &&
              (!uses_recomb || d_out_fesc) && (!use_ts_fluct || d_out_xray);

    if (ok) {
        const float gf = dicke(redshift);
        const float disp_2lpt = -(3.0f / 7.0f) * gf * gf;
        const float init_gf = dicke(simulation_options_global->INITIAL_REDSHIFT);
        const float init_disp_2lpt = -(3.0f / 7.0f) * init_gf * init_gf;
        const double vdf = (double)gf - (double)init_gf;
        const double vdf2 = (double)disp_2lpt - (double)init_disp_2lpt;

        const double boxlen = simulation_options_global->BOX_LEN;
        const double boxlen_z = boxlen * simulation_options_global->NON_CUBIC_FACTOR;
        const int lo_nx = simulation_options_global->HII_DIM;
        const int lo_nz = HII_D_PARA;

        P21HaloScalingParams hp = p21_make_halo_scaling_params(&consts);
        int blocks = (int)((nh + kBlockThreads - 1) / kBlockThreads);
        perturb_halos_and_props_kernel<<<blocks, kBlockThreads, 0, stream>>>(
            nh, hp, boxlen, boxlen_z, vdf, vdf2, velf.dim[0], velf.dim[1], velf.dim[2], lo_nx,
            lo_nx, lo_nz, velf.is_2lpt ? 1 : 0, use_reion_fb, uses_recomb, velf.v[0], velf.v[1],
            velf.v[2], velf.v2[0], velf.v2[1], velf.v2[2], d_mturn_acg, d_mturn_mcg, d_in_hm,
            d_in_hc, d_in_sr, d_in_sfrr, d_in_xr, d_out_coords, d_out_hm, d_out_sa, d_out_sfa,
            d_out_nion, d_out_sm, d_out_sfm, d_out_fesc, d_out_xray);
        ok = P21_CUDA_OK(cudaGetLastError());
    }

    if (ok) {
        auto copy_d2h = [&](float *hptr, const float *dptr, size_t nbytes) {
            if (!ok || !hptr || !dptr) return;
            ok = P21_CUDA_OK(cudaMemcpyAsync(hptr, dptr, nbytes, cudaMemcpyDeviceToHost, stream));
            if (ok) p21_cuda_stats_add_d2h(nbytes);
        };
        copy_d2h(halos_perturbed->halo_coords, d_out_coords, cbytes);
        copy_d2h(halos_perturbed->halo_masses, d_out_hm, sbytes);
        copy_d2h(halos_perturbed->stellar_masses_acg, d_out_sa, sbytes);
        copy_d2h(halos_perturbed->sfr_acg, d_out_sfa, sbytes);
        copy_d2h(halos_perturbed->n_ion, d_out_nion, sbytes);
        if (use_mcgs) {
            copy_d2h(halos_perturbed->stellar_masses_mcg, d_out_sm, sbytes);
            copy_d2h(halos_perturbed->sfr_mcg, d_out_sfm, sbytes);
        }
        if (uses_recomb) {
            copy_d2h(halos_perturbed->fesc_weighted_sfr, d_out_fesc, sbytes);
        }
        if (use_ts_fluct) {
            copy_d2h(halos_perturbed->xray_luminosity, d_out_xray, sbytes);
        }
        if (ok) ok = P21_CUDA_OK(cudaStreamSynchronize(stream));
    }

    p21_cuda_pool_free(d_mturn_acg, stream);
    p21_cuda_pool_free(d_mturn_mcg, stream);
    p21_cuda_pool_free(d_out_coords, stream);
    p21_cuda_pool_free(d_out_hm, stream);
    p21_cuda_pool_free(d_out_sa, stream);
    p21_cuda_pool_free(d_out_sfa, stream);
    p21_cuda_pool_free(d_out_nion, stream);
    p21_cuda_pool_free(d_out_sm, stream);
    p21_cuda_pool_free(d_out_sfm, stream);
    p21_cuda_pool_free(d_out_fesc, stream);
    p21_cuda_pool_free(d_out_xray, stream);

    if (!ok) {
        return -1;
    }
    p21_cuda_stats_mut()->halo_perturb_runs++;
    return 0;
}
