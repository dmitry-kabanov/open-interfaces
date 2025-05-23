// clang-format Language: C
#pragma once

#include <oif/api.h>
#include <oif/config_dict.h>

/**
 * Signature of the right-hand side function f for the system of ODEs.
 */
typedef int (*oif_ivp_rhs_fn_t)(double t, OIFArrayF64 *y, OIFArrayF64 *ydot, void *user_data);

/**
 * Set right hand side of the system of ordinary differential equations.
 */
int
oif_ivp_set_rhs_fn(oif_ivp_rhs_fn_t rhs);

/**
 * Set initial value y(t0) = y0.
 */
int
oif_ivp_set_initial_value(OIFArrayF64 *y0, double t0);

/**
 * Set user data that can be used to pass additional information
 * to the right-hand side function.
 */
int
oif_ivp_set_user_data(void *user_data);

/**
 * Integrate to time `t` and write the solution to `y`.
 */
int
oif_ivp_integrate(double t, OIFArrayF64 *y);

/**
 * Set integrator and optionally its parameters.
 */
int
oif_ivp_set_integrator(const char *integrator_name, OIFConfigDict *config);

/**
 * Print statistics about integration process.
 */
int
oif_ivp_print_stats();
