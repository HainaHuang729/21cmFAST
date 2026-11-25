#ifndef P21_CUDA_API_H
#define P21_CUDA_API_H

#include "../OutputStructs.h"
#include "../scaling_relations.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct P21EmissivityAverages {
    double halo_number;
    double halo_mass_density;
    double stellar_mass_density_acg;
    double stellar_mass_density_mcg;
    double sfrd_acg;
    double sfrd_mcg;
    double n_ion;
    double xray_emissivity;
    double fesc_weighted_sfrd;
} P21EmissivityAverages;

typedef struct P21IonBoxParams {
    double redshift;
    double stored_redshift;
    double prev_redshift;
    double growth_factor;
    double prev_growth_factor;
    double photoncons_adjustment_factor;
    double dz;
    double fabs_dtdz;
    int fix_mean;
    int mass_dep_zeta;
    int filter_recombinations;
    int lagrangian_source_grids;
    int need_mcgs_nion;
    int hii_filter;
    double T_re;
    double mfp_meandens;
    double gamma_prefactor;
    double nion_prefactor_acg;
    double nion_prefactor_mcg;
    double TK_nofluct;
    double adia_TK_term;
    double M_min;
    double vcb_const;
    double mturn_acg_homogeneous;
    double atomic_cooling_threshold;
} P21IonBoxParams;

typedef struct P21IonRadiusSpec {
    double R;
    double M_max_R;
    double ln_M_max_R;
    double sigma_maxmass;
    int R_index;
    double nion_conditional_ave_acg;
    double nion_conditional_ave_mcg;
} P21IonRadiusSpec;

typedef struct P21CudaEmissivityCtx P21CudaEmissivityCtx;
typedef struct P21CudaLagEmissCtx P21CudaLagEmissCtx;
typedef struct P21CudaIonCtx P21CudaIonCtx;
typedef struct P21CudaDexmCtx P21CudaDexmCtx;

typedef struct P21HaloSamplingParams {
    int from_catalog;
    double z_in;
    double z_out;
    double growth_in;
    double growth_out;
    double M_min;
    double lnM_min;
    double sigma_min;
    double M_max_tables;
    double lnM_max_tb;
    double M_cond;
    double lnM_cond;
    double sigma_cond;
    double corr_star;
    double corr_sfr;
    double corr_xray;
} P21HaloSamplingParams;

int p21_cuda_is_compiled(void);
int p21_cuda_is_available(void);
void p21_cuda_set_enabled(int enabled);
int p21_cuda_get_enabled(void);
void p21_cuda_set_fast_halo_sampler(int enabled);
int p21_cuda_get_fast_halo_sampler(void);
void p21_cuda_reset_stats(void);
void p21_cuda_get_stats(P21CudaStats *out);
void p21_cuda_cleanup(void);
void p21_cuda_invalidate_host_ptr(const void *ptr);
void p21_cuda_registry_clear(void);

int p21_cuda_compute_initial_conditions(unsigned long long int random_seed,
                                        InitialConditions *boxes);

int p21_cuda_compute_perturbed_field(float redshift, InitialConditions *boxes,
                                     PerturbedField *perturbed_field);

int p21_cuda_compute_brightness_temp(float redshift, TsBox *spin_temp, IonizedBox *ionized_box,
                                     PerturbedField *perturb_field, BrightnessTemp *box);

int p21_cuda_setup_radiation_mean_xe(TsBox *previous_spin_temp, double *x_e_ave_out);
int p21_cuda_setup_radiation_tables(TsBox *previous_spin_temp, RadiationFieldsSetup *rad_setup);
int p21_cuda_fill_freqint_tables(double zp, int num_R, const double *lower_int_limits,
                                 double hubble_zp, RadiationFieldsSetup *rad_setup);
int p21_cuda_update_radiation_fields(float redshift, EmissivityFields *emissivity_fields, int R_ct,
                                     double R_star, PerturbedField *perturbed_field,
                                     TsBox *previous_spin_temp, RadiationFieldsSetup *rad_setup,
                                     RadiationFields *radiation_fields);
int p21_cuda_compute_radiation_shells(float redshift, int n_shells, const int *R_ct_arr,
                                      double R_star, const double *interp_params,
                                      EmissivityFields **ef_desc_arr,
                                      EmissivityFields **ef_prog_arr,
                                      RadiationFieldsSetup *rad_setup,
                                      PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                                      RadiationFields *radiation_fields);

int p21_cuda_init_first_ts(TsBox *box, float *dens, float z, double xe_recfast, double tk_recfast,
                           double cT_ad);
int p21_cuda_compute_ts_box(float redshift, float prev_redshift, PerturbedField *perturbed_field,
                            RadiationFields *radiation_fields, TsBox *previous_spin_temp,
                            InitialConditions *ini_boxes, TsBox *this_spin_temp);

P21CudaEmissivityCtx *p21_cuda_emissivity_begin(
    double redshift, InitialConditions *ini_boxes, PerturbedField *perturbed_field,
    TsBox *previous_spin_temp, IonizedBox *previous_ionize_box, const ScalingConstants *consts,
    double log10_mturn_averages[2], double *min_dens_out, double *max_dens_out);
int p21_cuda_emissivity_eval_and_finish(P21CudaEmissivityCtx *ctx, const ScalingConstants *ev_consts,
                                        const P21EmissivityAverages *uhmf_averages,
                                        EmissivityFields *emissivity_fields);
void p21_cuda_emissivity_abort(P21CudaEmissivityCtx *ctx);

P21CudaIonCtx *p21_cuda_ion_begin(const P21IonBoxParams *params, int n_radii,
                                  PerturbedField *perturbed_field,
                                  PerturbedField *previous_perturbed_field,
                                  IonizedBox *previous_ionize_box, TsBox *spin_temp,
                                  EmissivityFields *emissivity_fields,
                                  InitialConditions *ini_boxes);
int p21_cuda_ion_calc_mcrit(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                            double *log10_mturn_acg_avg, double *log10_mturn_mcg_avg);
int p21_cuda_ion_set_fully_neutral(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                   double xion_recfast_z, IonizedBox *box, double *global_xH_out);
int p21_cuda_ion_prepare_r2c(P21CudaIonCtx *ctx, const P21IonBoxParams *params);
int p21_cuda_ion_filter_and_extrema(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                    const P21IonRadiusSpec *rspec, double *min_dens,
                                    double *max_dens, double *prev_min_dens,
                                    double *prev_max_dens);
int p21_cuda_ion_calc_cond_nion(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                                P21IonRadiusSpec *rspec, int need_prev_ion);
int p21_cuda_ion_find_regions(P21CudaIonCtx *ctx, const P21IonBoxParams *params,
                              const P21IonRadiusSpec *rspec, double nion_uncond_acg,
                              double nion_uncond_mcg, double nion_limit_acg, double nion_limit_mcg,
                              float prev_homo_nrec);
int p21_cuda_ion_set_ionized_temps(P21CudaIonCtx *ctx, const P21IonBoxParams *params);
int p21_cuda_ion_global_means(P21CudaIonCtx *ctx, double *global_xH_out, float *global_gamma12_out);
int p21_cuda_ion_recomb_inhomo(P21CudaIonCtx *ctx, const P21IonBoxParams *params);
int p21_cuda_ion_finish(P21CudaIonCtx *ctx, const P21IonBoxParams *params, IonizedBox *box);
void p21_cuda_ion_end(P21CudaIonCtx *ctx);
int p21_cuda_fill_mhr_table(int nz, float del_z, int ng, double lng_min, float del_lng,
                            double n_h0, double y_he, double alpha_b_1e4, const double *a_mhr,
                            const double *c_mhr, const double *beta_mhr, double *rr_table_out);

/* Phase 3: Lagrangian emissivity, PerturbedHaloCatalog, DEXM-ESF, dNdM tables & fast sampler */
P21CudaLagEmissCtx *p21_cuda_lag_emissivity_begin(
    double redshift, InitialConditions *ini_boxes, TsBox *previous_spin_temp,
    IonizedBox *previous_ionize_box, const ScalingConstants *consts, int need_integral,
    double log10_mturn_averages[2], double *min_dens_out, double *max_dens_out);
int p21_cuda_lag_emissivity_add_halos(P21CudaLagEmissCtx *ctx, InitialConditions *ini_boxes,
                                      HaloCatalog *halos, const ScalingConstants *consts);
int p21_cuda_lag_emissivity_add_integral(P21CudaLagEmissCtx *ctx, InitialConditions *ini_boxes,
                                         const ScalingConstants *ev_consts,
                                         const P21EmissivityAverages *uhmf_averages);
int p21_cuda_lag_emissivity_finish(P21CudaLagEmissCtx *ctx, EmissivityFields *emissivity_fields);
void p21_cuda_lag_emissivity_abort(P21CudaLagEmissCtx *ctx);

int p21_cuda_compute_perturbed_halo_catalog(float redshift, InitialConditions *boxes,
                                            TsBox *prev_ts, IonizedBox *prev_ion,
                                            HaloCatalog *halos,
                                            PerturbedHaloCatalog *halos_perturbed);

P21CudaDexmCtx *p21_cuda_dexm_run(float redshift, InitialConditions *boxes, float M_MIN,
                                  float Delta_R, unsigned long long int *total_halo_num_out);
int p21_cuda_dexm_finish(P21CudaDexmCtx *ctx, HaloCatalog *halos_dexm, float *halo_overlap_box_out);
void p21_cuda_dexm_abort(P21CudaDexmCtx *ctx);

int p21_cuda_fill_dndm_tables(int nx, double xmin, double xmax, double ymin, double ymax,
                              double growth_out, double param, int from_catalog, double *nhalo_out,
                              double *fcoll_out);
int p21_cuda_fill_dndm_inverse_table(int nx, int np, double xmin, double xmax, double lnM_min,
                                     double growth_out, double param, int from_catalog,
                                     double min_lp, double **z_arr_out);

int p21_cuda_sample_halo_grids(unsigned long long int seed, double redshift, float *dens_field,
                               float *halo_overlap_box, HaloCatalog *halofield_large,
                               HaloCatalog *halofield_out, const P21HaloSamplingParams *hsp,
                               int *overflow_out, unsigned long long int *thread_counts_out);
int p21_cuda_sample_halo_progenitors(unsigned long long int seed, double z_in, double z_out,
                                     HaloCatalog *halofield_in, HaloCatalog *halofield_out,
                                     const P21HaloSamplingParams *hsp, int *overflow_out,
                                     unsigned long long int *thread_counts_out);

#ifdef __cplusplus
}
#endif

#endif /* P21_CUDA_API_H */
