#ifndef _RECOMB_H
#define _RECOMB_H

double splined_recombination_rate(double z_eff, double gamma12_bg);
/*initializes the lookup table for the PDF density integral in MHR00 model at redshift z*/
void init_MHR();
void free_MHR();
int p21_get_mhr_rr_table(const double **rr_table_out, int *nz_out, int *ng_out,
                         double *del_z_out, double *lng_min_out, double *del_lng_out,
                         int *generation_out);

#endif
