# %%

import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import os
import theano

# %%

measurements_path = os.path.join("data", "patient_measurements.csv")
parameters_path = os.path.join("data", "patient_parameters.csv")

all_measurements_df = pd.read_csv(measurements_path)
all_parameters_df = pd.read_csv(parameters_path)

# %%

# Choosing one patient (patient 1)
measurements_df = all_measurements_df[all_measurements_df["patient_index"] == 0]
parameters_df = all_parameters_df[all_parameters_df["patient_index"] == 0]

# %%

# ANALYZING SINGLE PATIENT

# SINGLE PATIENT MODEL WITHOUT TREND

with pm.Model() as single_patient_no_trend_model:

    baseline_prior = pm.Normal("baseline", mu=10, sigma=3)
    treatment_effect_prior = pm.Normal("treatment_effect", mu=0, sigma=0)
    sigma_prior = pm.HalfCauchy("sigma", beta=10, testval=1.0)

    # likelihood is not a well defined distribution
    # so using prebuilt parts does not work. Writing
    # custom function calculating the log likelihood

    def likelihood(baseline, treatment, sigma):
        def logp_(value):

            return (-1 / 2.0) * (
                value[0].shape[0] * theano.tensor.log(2 * np.pi)
                + value[0].shape[0] * theano.tensor.log(sigma ** 2)
                + (1 / sigma ** 2)
                * ((value[0] - (baseline + treatment * value[1])) ** 2).sum(axis=0)
            )

        return logp_

    like = pm.DensityDist(
        "y",
        likelihood(
            baseline=baseline_prior,
            treatment=treatment_effect_prior,
            sigma=sigma_prior,
        ),
        observed=[measurements_df["measurement"], measurements_df["treatment"]],
    )

    trace = pm.sample(1000, tune=500, cores=3)

    pm.traceplot(trace, ["baseline", "treatment_effect", "sigma"])

    # posteriors should look reasonable
    pm.plot_posterior(trace)
    plt.show()

    # check if your variables have reasonable credible intervals,
    # and Gelman–Rubin scores close to 1
    pm.forestplot(trace)
    plt.show()

    # check if your chains are impaired by high autocorrelation.
    # Also remember that thinning your chains is a waste of time
    # at best, and deluding yourself at worst
    pm.autocorrplot(trace)
    plt.show()

    # ideally the energy and marginal energy distributions should
    # look very similar. Long tails in the distribution of energy levels
    # indicates deteriorated sampler efficiency.
    pm.energyplot(trace)
    plt.show()

    # summary statistics
    # Look out for:
    #
    # the ^R values (a.k.a. the Gelman–Rubin statistic,
    # a.k.a. the potential scale reduction factor, a.k.a. the PSRF):
    # are they all close to 1? If not, something is horribly wrong.
    # Consider respecifying or reparameterizing your model.
    # You can also inspect these in the forest plot.
    #
    # the sign and magnitude of the inferred values: do they
    # make sense, or are they unexpected and unreasonable?
    # This could indicate a poorly specified model.
    # (E.g. parameters of the unexpected sign that have low
    # uncertainties might indicate that your model needs
    # interaction terms.)
    pm.summary(trace)

# %%

# SINGLE PATIENT MODEL WITH TREND

with pm.Model() as single_patient_with_trend_model:

    baseline_prior = pm.Normal("baseline", mu=10, sigma=3)
    treatment_effect_prior = pm.Normal("treatment_effect", mu=0, sigma=1)
    trend_prior = pm.Normal("trend", mu=0, sigma=1)
    sigma_prior = pm.HalfCauchy("sigma", beta=10, testval=1.0)

    # likelihood is not a well defined distribution
    # so using prebuilt parts does not work. Writing
    # custom function calculating the log likelihood

    def likelihood(baseline, treatment, trend, sigma):
        def logp_(value):

            return (-1 / 2.0) * (
                value[0].shape[0] * theano.tensor.log(2 * np.pi)
                + value[0].shape[0] * theano.tensor.log(sigma ** 2)
                + (1 / sigma ** 2)
                * (
                    (value[0] - (baseline + trend * value[2] + treatment * value[1]))
                    ** 2
                ).sum(axis=0)
            )

        return logp_

    like = pm.DensityDist(
        "y",
        likelihood(
            baseline=baseline_prior,
            treatment=treatment_effect_prior,
            trend=trend_prior,
            sigma=sigma_prior,
        ),
        observed=[
            measurements_df["measurement"],
            measurements_df["treatment"],
            measurements_df["measurement_index"],
        ],
    )

    trace = pm.sample(1000, tune=500, cores=3)

    pm.traceplot(trace, ["baseline", "treatment_effect", "trend", "sigma"])
    plt.show()

    # posteriors should look reasonable
    pm.plot_posterior(trace)
    plt.show()

    # check if your variables have reasonable credible intervals,
    # and Gelman–Rubin scores close to 1
    pm.forestplot(trace)
    plt.show()

    # check if your chains are impaired by high autocorrelation.
    # Also remember that thinning your chains is a waste of time
    # at best, and deluding yourself at worst
    pm.autocorrplot(trace)
    plt.show()

    # ideally the energy and marginal energy distributions should
    # look very similar. Long tails in the distribution of energy levels
    # indicates deteriorated sampler efficiency.
    pm.energyplot(trace)
    plt.show()

    # summary statistics
    # Look out for:
    #
    # the ^R values (a.k.a. the Gelman–Rubin statistic,
    # a.k.a. the potential scale reduction factor, a.k.a. the PSRF):
    # are they all close to 1? If not, something is horribly wrong.
    # Consider respecifying or reparameterizing your model.
    # You can also inspect these in the forest plot.
    #
    # the sign and magnitude of the inferred values: do they
    # make sense, or are they unexpected and unreasonable?
    # This could indicate a poorly specified model.
    # (E.g. parameters of the unexpected sign that have low
    # uncertainties might indicate that your model needs
    # interaction terms.)
    pm.summary(trace)

# %%

# ANALYZING ALL PATIENTS AT ONCE

# one row of parameters per patient
patients_n = len(all_parameters_df)
# used for subsetting parameters
patient_index = all_measurements_df["patient_index"]

# %%

# UNPOOLED MODEL WITHOUT TREND

with pm.Model() as all_patients_no_trend_model:

    # separate parameter for each patient
    baseline_prior = pm.Normal("baseline", mu=10, sigma=3, shape=patients_n)
    treatment_effect_prior = pm.Normal(
        "treatment_effect", mu=0, sigma=1, shape=patients_n
    )
    sigma_prior = pm.HalfCauchy("sigma", beta=10, testval=1.0, shape=patients_n)

    # likelihood is not a well defined distribution
    # so using prebuilt parts does not work. Writing
    # custom function calculating the log likelihood

    def likelihood(baseline, treatment, sigma):
        def logp_(value):

            return (-1 / 2.0) * (
                value[0].shape[0] * theano.tensor.log(2 * np.pi)
                + value[0].shape[0] * theano.tensor.log(sigma[patient_index] ** 2)
                + (1 / sigma[patient_index] ** 2)
                * (
                    (
                        value[0]
                        - (
                            baseline[patient_index]
                            + treatment[patient_index] * value[1]
                        )
                    )
                    ** 2
                ).sum(axis=0)
            ).sum(axis=0)

        return logp_

    like = pm.DensityDist(
        "y",
        likelihood(
            baseline=baseline_prior,
            treatment=treatment_effect_prior,
            sigma=sigma_prior,
        ),
        observed=[all_measurements_df["measurement"], all_measurements_df["treatment"]],
    )

    trace = pm.sample(1000, tune=500, cores=3)

    pm.traceplot(trace, ["baseline", "treatment_effect", "sigma"])
    plt.show()

# %%

# UNPOOLED MODEL WITH TREND

with pm.Model() as all_patients_with_trend_model:

    # separate parameter for each patient
    baseline_prior = pm.Normal("baseline", mu=10, sigma=3, shape=patients_n)
    treatment_effect_prior = pm.Normal(
        "treatment_effect", mu=0, sigma=1, shape=patients_n
    )
    trend_prior = pm.Normal("trend", mu=0, sigma=1, shape=patients_n)
    sigma_prior = pm.HalfCauchy("sigma", beta=10, testval=1.0, shape=patients_n)

    # likelihood is not a well defined distribution
    # so using prebuilt parts does not work. Writing
    # custom function calculating the log likelihood

    def likelihood(baseline, treatment, trend, sigma):
        def logp_(value):

            return (-1 / 2.0) * (
                value[0].shape[0] * theano.tensor.log(2 * np.pi)
                + value[0].shape[0] * theano.tensor.log(sigma[patient_index] ** 2)
                + (1 / sigma[patient_index] ** 2)
                * (
                    (
                        value[0]
                        - (
                            baseline[patient_index]
                            + trend[patient_index] * value[2]
                            + treatment[patient_index] * value[1]
                        )
                    )
                    ** 2
                ).sum(axis=0)
            ).sum(axis=0)

        return logp_

    like = pm.DensityDist(
        "y",
        likelihood(
            baseline=baseline_prior,
            treatment=treatment_effect_prior,
            trend=trend_prior,
            sigma=sigma_prior,
        ),
        observed=[
            all_measurements_df["measurement"],
            all_measurements_df["treatment"],
            all_measurements_df["measurement_index"],
        ],
    )

    trace = pm.sample(1000, tune=500, cores=3)

    pm.traceplot(trace, ["baseline", "treatment_effect", "trend", "sigma"])
    plt.show()

    # posteriors should look reasonable
    pm.plot_posterior(trace)
    plt.show()

    # check if your variables have reasonable credible intervals,
    # and Gelman–Rubin scores close to 1
    pm.forestplot(trace)
    plt.show()

    # check if your chains are impaired by high autocorrelation.
    # Also remember that thinning your chains is a waste of time
    # at best, and deluding yourself at worst
    pm.autocorrplot(trace)
    plt.show()

    # ideally the energy and marginal energy distributions should
    # look very similar. Long tails in the distribution of energy levels
    # indicates deteriorated sampler efficiency.
    pm.energyplot(trace)
    plt.show()

    # summary statistics
    # Look out for:
    #
    # the ^R values (a.k.a. the Gelman–Rubin statistic,
    # a.k.a. the potential scale reduction factor, a.k.a. the PSRF):
    # are they all close to 1? If not, something is horribly wrong.
    # Consider respecifying or reparameterizing your model.
    # You can also inspect these in the forest plot.
    #
    # the sign and magnitude of the inferred values: do they
    # make sense, or are they unexpected and unreasonable?
    # This could indicate a poorly specified model.
    # (E.g. parameters of the unexpected sign that have low
    # uncertainties might indicate that your model needs
    # interaction terms.)
    pm.summary(trace)

# %%

# HIERARCHICAL MODELS

# %%

# HIERARCHICAL MODEL WITHOUT TREND

with pm.Model() as hierarchical_no_trend_model:

    # population priors
    population_baseline_mean_prior = pm.Normal(
        "population_baseline_mean", mu=10, sigma=3
    )
    population_baseline_sd_prior = pm.HalfCauchy(
        "population_baseline_sd", beta=10, testval=1
    )

    population_treatment_effect_mean_prior = pm.Normal(
        "population_treatment_effect_mean", mu=0, sigma=1
    )
    population_treatment_effect_sd_prior = pm.HalfCauchy(
        "population_treatment_effect_sd", beta=10, testval=1
    )

    population_measurement_error_beta_prior = pm.HalfCauchy(
        "population_measurement_error_beta", beta=10, testval=1
    )

    # separate parameter for each patient
    baseline_prior = pm.Normal(
        "baseline",
        mu=population_baseline_mean_prior,
        sigma=population_baseline_sd_prior,
        shape=patients_n,
    )
    treatment_effect_prior = pm.Normal(
        "treatment_effect",
        mu=population_treatment_effect_mean_prior,
        sigma=population_treatment_effect_sd_prior,
        shape=patients_n,
    )
    sigma_prior = pm.HalfCauchy(
        "sigma",
        beta=population_measurement_error_beta_prior,
        testval=1.0,
        shape=patients_n,
    )

    # likelihood is not a well defined distribution
    # so using prebuilt parts does not work. Writing
    # custom function calculating the log likelihood

    def likelihood(baseline, treatment, sigma):
        def logp_(value):

            return (-1 / 2.0) * (
                value[0].shape[0] * theano.tensor.log(2 * np.pi)
                + value[0].shape[0] * theano.tensor.log(sigma[patient_index] ** 2)
                + (1 / sigma[patient_index] ** 2)
                * (
                    (
                        value[0]
                        - (
                            baseline[patient_index]
                            + treatment[patient_index] * value[1]
                        )
                    )
                    ** 2
                ).sum(axis=0)
            ).sum(axis=0)

        return logp_

    like = pm.DensityDist(
        "y",
        likelihood(
            baseline=baseline_prior,
            treatment=treatment_effect_prior,
            sigma=sigma_prior,
        ),
        observed=[all_measurements_df["measurement"], all_measurements_df["treatment"]],
    )

    trace = pm.sample(1000, tune=500, cores=3)

    pm.traceplot(
        trace,
        [
            "baseline",
            "treatment_effect",
            "sigma",
            "population_baseline_mean",
            "population_baseline_sd",
            "population_treatment_effect_mean",
            "population_treatment_effect_sd",
            "population_measurement_error_beta",
        ],
    )
    plt.show()

    # posteriors should look reasonable
    pm.plot_posterior(trace)
    plt.show()

    # check if your variables have reasonable credible intervals,
    # and Gelman–Rubin scores close to 1
    pm.forestplot(trace)
    plt.show()

    # check if your chains are impaired by high autocorrelation.
    # Also remember that thinning your chains is a waste of time
    # at best, and deluding yourself at worst
    pm.autocorrplot(trace)
    plt.show()

    # ideally the energy and marginal energy distributions should
    # look very similar. Long tails in the distribution of energy levels
    # indicates deteriorated sampler efficiency.
    pm.energyplot(trace)
    plt.show()

    # summary statistics
    # Look out for:
    #
    # the ^R values (a.k.a. the Gelman–Rubin statistic,
    # a.k.a. the potential scale reduction factor, a.k.a. the PSRF):
    # are they all close to 1? If not, something is horribly wrong.
    # Consider respecifying or reparameterizing your model.
    # You can also inspect these in the forest plot.
    #
    # the sign and magnitude of the inferred values: do they
    # make sense, or are they unexpected and unreasonable?
    # This could indicate a poorly specified model.
    # (E.g. parameters of the unexpected sign that have low
    # uncertainties might indicate that your model needs
    # interaction terms.)
    pm.summary(trace)

# %%

# HIERARCHICAL MODEL WITH TREND

with pm.Model() as hierarchical_with_trend_model:

    # population priors
    population_baseline_mean_prior = pm.Normal(
        "population_baseline_mean", mu=10, sigma=3
    )
    population_baseline_sd_prior = pm.HalfCauchy(
        "population_baseline_sd", beta=10, testval=1
    )

    population_treatment_effect_mean_prior = pm.Normal(
        "population_treatment_effect_mean", mu=0, sigma=1
    )
    population_treatment_effect_sd_prior = pm.HalfCauchy(
        "population_treatment_effect_sd", beta=10, testval=1
    )

    population_measurement_error_beta_prior = pm.HalfCauchy(
        "population_measurement_error_beta", beta=10, testval=1
    )

    population_trend_mean_prior = pm.Normal("population_trend_mean", mu=0, sigma=0.3)
    population_trend_sd_prior = pm.HalfCauchy("population_trend_sd", beta=2, testval=1)

    # separate parameter for each patient
    baseline_prior = pm.Normal(
        "baseline",
        mu=population_baseline_mean_prior,
        sigma=population_baseline_sd_prior,
        shape=patients_n,
    )
    treatment_effect_prior = pm.Normal(
        "treatment_effect",
        mu=population_treatment_effect_mean_prior,
        sigma=population_treatment_effect_sd_prior,
        shape=patients_n,
    )
    sigma_prior = pm.HalfCauchy(
        "sigma",
        beta=population_measurement_error_beta_prior,
        testval=1.0,
        shape=patients_n,
    )
    trend_prior = pm.Normal(
        "trend",
        mu=population_trend_mean_prior,
        sigma=population_trend_sd_prior,
        shape=patients_n,
    )

    # likelihood is not a well defined distribution
    # so using prebuilt parts does not work. Writing
    # custom function calculating the log likelihood

    def likelihood(baseline, treatment, trend, sigma):
        def logp_(value):

            return (-1 / 2.0) * (
                value[0].shape[0] * theano.tensor.log(2 * np.pi)
                + value[0].shape[0] * theano.tensor.log(sigma[patient_index] ** 2)
                + (1 / sigma[patient_index] ** 2)
                * (
                    (
                        value[0]
                        - (
                            baseline[patient_index]
                            + trend[patient_index] * value[2]
                            + treatment[patient_index] * value[1]
                        )
                    )
                    ** 2
                ).sum(axis=0)
            ).sum(axis=0)

        return logp_

    like = pm.DensityDist(
        "y",
        likelihood(
            baseline=baseline_prior,
            treatment=treatment_effect_prior,
            trend=trend_prior,
            sigma=sigma_prior,
        ),
        observed=[
            all_measurements_df["measurement"],
            all_measurements_df["treatment"],
            all_measurements_df["measurement_index"],
        ],
    )

    trace = pm.sample(1000, tune=500, cores=3)

    pm.traceplot(
        trace,
        [
            "baseline",
            "treatment_effect",
            "sigma",
            "population_baseline_mean",
            "population_baseline_sd",
            "population_treatment_effect_mean",
            "population_treatment_effect_sd",
            "population_measurement_error_beta",
            "population_trend_mean",
            "population_trend_sd",
        ],
    )

    # posteriors should look reasonable
    pm.plot_posterior(trace)
    plt.show()

    # check if your variables have reasonable credible intervals,
    # and Gelman–Rubin scores close to 1
    pm.forestplot(trace)
    plt.show()

    # check if your chains are impaired by high autocorrelation.
    # Also remember that thinning your chains is a waste of time
    # at best, and deluding yourself at worst
    pm.autocorrplot(trace)
    plt.show()

    # ideally the energy and marginal energy distributions should
    # look very similar. Long tails in the distribution of energy levels
    # indicates deteriorated sampler efficiency.
    pm.energyplot(trace)
    plt.show()

    # summary statistics
    # Look out for:
    #
    # the ^R values (a.k.a. the Gelman–Rubin statistic,
    # a.k.a. the potential scale reduction factor, a.k.a. the PSRF):
    # are they all close to 1? If not, something is horribly wrong.
    # Consider respecifying or reparameterizing your model.
    # You can also inspect these in the forest plot.
    #
    # the sign and magnitude of the inferred values: do they
    # make sense, or are they unexpected and unreasonable?
    # This could indicate a poorly specified model.
    # (E.g. parameters of the unexpected sign that have low
    # uncertainties might indicate that your model needs
    # interaction terms.)
    pm.summary(trace)
