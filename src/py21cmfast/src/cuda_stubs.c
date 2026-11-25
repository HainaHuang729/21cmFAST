#ifndef USE_CUDA

#include <string.h>

#include "21cmFAST.h"

static int g_p21_cuda_enabled = 0;
static int g_p21_cuda_fast_halo_sampler = 0;

int p21_cuda_is_compiled(void) { return 0; }

int p21_cuda_is_available(void) { return 0; }

void p21_cuda_set_enabled(int enabled) { g_p21_cuda_enabled = enabled ? 1 : 0; }

int p21_cuda_get_enabled(void) { return g_p21_cuda_enabled; }

void p21_cuda_set_fast_halo_sampler(int enabled) {
    g_p21_cuda_fast_halo_sampler = enabled ? 1 : 0;
}

int p21_cuda_get_fast_halo_sampler(void) { return g_p21_cuda_fast_halo_sampler; }

void p21_cuda_reset_stats(void) {}

void p21_cuda_get_stats(P21CudaStats *out) {
    if (out) {
        memset(out, 0, sizeof(*out));
        out->device_id = -1;
        out->enabled = g_p21_cuda_enabled;
    }
}

void p21_cuda_cleanup(void) {}

void p21_cuda_invalidate_host_ptr(const void *ptr) { (void)ptr; }

void p21_cuda_registry_clear(void) {}

int p21_cuda_compute_initial_conditions(unsigned long long int random_seed,
                                        InitialConditions *boxes) {
    (void)random_seed;
    (void)boxes;
    return -1;
}

int p21_cuda_compute_perturbed_field(float redshift, InitialConditions *boxes,
                                     PerturbedField *perturbed_field) {
    (void)redshift;
    (void)boxes;
    (void)perturbed_field;
    return -1;
}

int p21_cuda_compute_brightness_temp(float redshift, TsBox *spin_temp, IonizedBox *ionized_box,
                                     PerturbedField *perturb_field, BrightnessTemp *box) {
    (void)redshift;
    (void)spin_temp;
    (void)ionized_box;
    (void)perturb_field;
    (void)box;
    return -1;
}

int p21_cuda_setup_radiation_mean_xe(TsBox *previous_spin_temp, double *x_e_ave_out) {
    (void)previous_spin_temp;
    (void)x_e_ave_out;
    return -1;
}

int p21_cuda_setup_radiation_tables(TsBox *previous_spin_temp, RadiationFieldsSetup *rad_setup) {
    (void)previous_spin_temp;
    (void)rad_setup;
    return -1;
}

int p21_cuda_fill_freqint_tables(double zp, int num_R, const double *lower_int_limits,
                                 double hubble_zp, RadiationFieldsSetup *rad_setup) {
    (void)zp;
    (void)num_R;
    (void)lower_int_limits;
    (void)hubble_zp;
    (void)rad_setup;
    return -1;
}

int p21_cuda_update_radiation_fields(float redshift, EmissivityFields *emissivity_fields, int R_ct,
                                     double R_star, PerturbedField *perturbed_field,
                                     TsBox *previous_spin_temp, RadiationFieldsSetup *rad_setup,
                                     RadiationFields *radiation_fields) {
    (void)redshift;
    (void)emissivity_fields;
    (void)R_ct;
    (void)R_star;
    (void)perturbed_field;
    (void)previous_spin_temp;
    (void)rad_setup;
    (void)radiation_fields;
    return -1;
}

int p21_cuda_compute_radiation_shells(float redshift, int n_shells, const int *R_ct_arr,
                                      double R_star, const double *interp_params,
                                      EmissivityFields **ef_desc_arr,
                                      EmissivityFields **ef_prog_arr,
                                      RadiationFieldsSetup *rad_setup,
                                      PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                                      RadiationFields *radiation_fields) {
    (void)redshift;
    (void)n_shells;
    (void)R_ct_arr;
    (void)R_star;
    (void)interp_params;
    (void)ef_desc_arr;
    (void)ef_prog_arr;
    (void)rad_setup;
    (void)perturbed_field;
    (void)previous_spin_temp;
    (void)radiation_fields;
    return -1;
}

int p21_cuda_init_first_ts(TsBox *box, float *dens, float z, double xe_recfast, double tk_recfast,
                           double cT_ad) {
    (void)box;
    (void)dens;
    (void)z;
    (void)xe_recfast;
    (void)tk_recfast;
    (void)cT_ad;
    return -1;
}

int p21_cuda_compute_ts_box(float redshift, float prev_redshift, PerturbedField *perturbed_field,
                            RadiationFields *radiation_fields, TsBox *previous_spin_temp,
                            InitialConditions *ini_boxes, TsBox *this_spin_temp) {
    (void)redshift;
    (void)prev_redshift;
    (void)perturbed_field;
    (void)radiation_fields;
    (void)previous_spin_temp;
    (void)ini_boxes;
    (void)this_spin_temp;
    return -1;
}

P21CudaEmissivityCtx *p21_cuda_emissivity_begin(
    double redshift, InitialConditions *ini_boxes, PerturbedField *perturbed_field,
    TsBox *previous_spin_temp, IonizedBox *previous_ionize_box, const ScalingConstants *consts,
    double log10_mturn_averages[2], double *min_dens_out, double *max_dens_out) {
    (void)redshift;
    (void)ini_boxes;
    (void)perturbed_field;
    (void)previous_spin_temp;
    (void)previous_ionize_box;
    (void)consts;
    (void)log10_mturn_averages;
    (void)min_dens_out;
    (void)max_dens_out;
    return NULL;
}

int p21_cuda_emissivity_eval_and_finish(P21CudaEmissivityCtx *ctx, const ScalingConstants *ev_consts,
                                        const P21EmissivityAverages *uhmf_averages,
                                        EmissivityFields *emissivity_fields) {
    (void)ctx;
    (void)ev_consts;
    (void)uhmf_averages;
    (void)emissivity_fields;
    return -1;
}

void p21_cuda_emissivity_abort(P21CudaEmissivityCtx *ctx) { (void)ctx; }

P21CudaIonCtx *p21_cuda_ion_begin(const P21IonBoxParams *params, int n_radii,
                                  PerturbedField *perturbed_field,
                                  PerturbedField *previous_perturbed_field,
                                  IonizedBox *previous_ionize_box, TsBox *spin_temp,
                                  EmissivityFields *emissivity_fields,
                                  InitialConditions *ini_boxes) {
    (void)params;
    (void)n_radii;
    (void)perturbed_field;
    (void)previous_perturbed_field;
    (void)previous_ionize_box;
    (void)spin_temp;
    (void)emissivity_fields;
    (void)ini_boxes;
    return NULL;
}

int p21_cuda_ion_calc_mcrit(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                            double *log10_mturn_acg_avg, double *log10_mturn_mcg_avg) {
    (void)ctx;
    (void)params;
    (void)log10_mturn_acg_avg;
    (void)log10_mturn_mcg_avg;
    return -1;
}

int p21_cuda_ion_set_fully_neutral(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                   double xion_recfast_z, IonizedBox *box, double *global_xH_out) {
    (void)ctx;
    (void)params;
    (void)xion_recfast_z;
    (void)box;
    (void)global_xH_out;
    return -1;
}

int p21_cuda_ion_prepare_r2c(P21CudaIonCtx *ctx, const P21IonBoxParams *params) {
    (void)ctx;
    (void)params;
    return -1;
}

int p21_cuda_ion_filter_and_extrema(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                    const P21IonRadiusSpec *rspec, double *min_dens,
                                    double *max_dens, double *prev_min_dens,
                                    double *prev_max_dens) {
    (void)ctx;
    (void)params;
    (void)rspec;
    (void)min_dens;
    (void)max_dens;
    (void)prev_min_dens;
    (void)prev_max_dens;
    return -1;
}

int p21_cuda_ion_calc_cond_nion(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                P21IonRadiusSpec *rspec, int need_prev_ion) {
    (void)ctx;
    (void)params;
    (void)rspec;
    (void)need_prev_ion;
    return -1;
}

int p21_cuda_ion_find_regions(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                              const P21IonRadiusSpec *rspec, double nion_uncond_acg,
                              double nion_uncond_mcg, double nion_limit_acg, double nion_limit_mcg,
                              float prev_homo_nrec) {
    (void)ctx;
    (void)params;
    (void)rspec;
    (void)nion_uncond_acg;
    (void)nion_uncond_mcg;
    (void)nion_limit_acg;
    (void)nion_limit_mcg;
    (void)prev_homo_nrec;
    return -1;
}

int p21_cuda_ion_set_ionized_temps(P21CudaIonCtx *ctx, const P21IonBoxParams *params) {
    (void)ctx;
    (void)params;
    return -1;
}

int p21_cuda_ion_global_means(P21CudaIonCtx *ctx, double *global_xH_out, float *global_gamma12_out) {
    (void)ctx;
    (void)global_xH_out;
    (void)global_gamma12_out;
    return -1;
}

int p21_cuda_ion_recomb_inhomo(P21CudaIonCtx *ctx, const P21IonBoxParams *params) {
    (void)ctx;
    (void)params;
    return -1;
}

int p21_cuda_ion_finish(P21CudaIonCtx *ctx, const P21IonBoxParams *params, IonizedBox *box) {
    (void)ctx;
    (void)params;
    (void)box;
    return -1;
}

void p21_cuda_ion_end(P21CudaIonCtx *ctx) { (void)ctx; }

int p21_cuda_fill_mhr_table(int nz, float del_z, int ng, double lng_min, float del_lng,
                            double n_h0, double y_he, double alpha_b_1e4, const double *a_mhr,
                            const double *c_mhr, const double *beta_mhr, double *rr_table_out) {
    (void)nz;
    (void)del_z;
    (void)ng;
    (void)lng_min;
    (void)del_lng;
    (void)n_h0;
    (void)y_he;
    (void)alpha_b_1e4;
    (void)a_mhr;
    (void)c_mhr;
    (void)beta_mhr;
    (void)rr_table_out;
    return -1;
}

P21CudaLagEmissCtx *p21_cuda_lag_emissivity_begin(
    double redshift, InitialConditions *ini_boxes, TsBox *previous_spin_temp,
    IonizedBox *previous_ionize_box, const ScalingConstants *consts, int need_integral,
    double log10_mturn_averages[2], double *min_dens_out, double *max_dens_out) {
    (void)redshift;
    (void)ini_boxes;
    (void)previous_spin_temp;
    (void)previous_ionize_box;
    (void)consts;
    (void)need_integral;
    (void)log10_mturn_averages;
    (void)min_dens_out;
    (void)max_dens_out;
    return NULL;
}

int p21_cuda_lag_emissivity_add_halos(P21CudaLagEmissCtx *ctx, InitialConditions *ini_boxes,
                                      HaloCatalog *halos, const ScalingConstants *consts) {
    (void)ctx;
    (void)ini_boxes;
    (void)halos;
    (void)consts;
    return -1;
}

int p21_cuda_lag_emissivity_add_integral(P21CudaLagEmissCtx *ctx, InitialConditions *ini_boxes,
                                         const ScalingConstants *ev_consts,
                                         const P21EmissivityAverages *uhmf_averages) {
    (void)ctx;
    (void)ini_boxes;
    (void)ev_consts;
    (void)uhmf_averages;
    return -1;
}

int p21_cuda_lag_emissivity_finish(P21CudaLagEmissCtx *ctx, EmissivityFields *emissivity_fields) {
    (void)ctx;
    (void)emissivity_fields;
    return -1;
}

void p21_cuda_lag_emissivity_abort(P21CudaLagEmissCtx *ctx) { (void)ctx; }

int p21_cuda_compute_perturbed_halo_catalog(float redshift, InitialConditions *boxes,
                                            TsBox *prev_ts, IonizedBox *prev_ion,
                                            HaloCatalog *halos,
                                            PerturbedHaloCatalog *halos_perturbed) {
    (void)redshift;
    (void)boxes;
    (void)prev_ts;
    (void)prev_ion;
    (void)halos;
    (void)halos_perturbed;
    return -1;
}

P21CudaDexmCtx *p21_cuda_dexm_run(float redshift, InitialConditions *boxes, float M_MIN,
                                  float Delta_R, unsigned long long int *total_halo_num_out) {
    (void)redshift;
    (void)boxes;
    (void)M_MIN;
    (void)Delta_R;
    (void)total_halo_num_out;
    return NULL;
}

int p21_cuda_dexm_finish(P21CudaDexmCtx *ctx, HaloCatalog *halos_dexm,
                         float *halo_overlap_box_out) {
    (void)ctx;
    (void)halos_dexm;
    (void)halo_overlap_box_out;
    return -1;
}

void p21_cuda_dexm_abort(P21CudaDexmCtx *ctx) { (void)ctx; }

int p21_cuda_fill_dndm_tables(int nx, double xmin, double xmax, double ymin, double ymax,
                              double growth_out, double param, int from_catalog, double *nhalo_out,
                              double *fcoll_out) {
    (void)nx;
    (void)xmin;
    (void)xmax;
    (void)ymin;
    (void)ymax;
    (void)growth_out;
    (void)param;
    (void)from_catalog;
    (void)nhalo_out;
    (void)fcoll_out;
    return -1;
}

int p21_cuda_fill_dndm_inverse_table(int nx, int np, double xmin, double xmax, double lnM_min,
                                     double growth_out, double param, int from_catalog,
                                     double min_lp, double **z_arr_out) {
    (void)nx;
    (void)np;
    (void)xmin;
    (void)xmax;
    (void)lnM_min;
    (void)growth_out;
    (void)param;
    (void)from_catalog;
    (void)min_lp;
    (void)z_arr_out;
    return -1;
}

int p21_cuda_sample_halo_grids(unsigned long long int seed, double redshift, float *dens_field,
                               float *halo_overlap_box, HaloCatalog *halofield_large,
                               HaloCatalog *halofield_out, const P21HaloSamplingParams *hsp,
                               int *overflow_out, unsigned long long int *thread_counts_out) {
    (void)seed;
    (void)redshift;
    (void)dens_field;
    (void)halo_overlap_box;
    (void)halofield_large;
    (void)halofield_out;
    (void)hsp;
    (void)overflow_out;
    (void)thread_counts_out;
    return -1;
}

int p21_cuda_sample_halo_progenitors(unsigned long long int seed, double z_in, double z_out,
                                     HaloCatalog *halofield_in, HaloCatalog *halofield_out,
                                     const P21HaloSamplingParams *hsp, int *overflow_out,
                                     unsigned long long int *thread_counts_out) {
    (void)seed;
    (void)z_in;
    (void)z_out;
    (void)halofield_in;
    (void)halofield_out;
    (void)hsp;
    (void)overflow_out;
    (void)thread_counts_out;
    return -1;
}

#else

typedef int p21_cuda_stubs_unused_translation_unit_t;

#endif /* USE_CUDA */
