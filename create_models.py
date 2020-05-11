# %%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns
import theano

# %%

# Seed used for posterior sampling
np.random.seed(123)

# %%

measurements_path = os.path.join("data", "patient_measurements.csv")
parameters_path = os.path.join("data", "patient_parameters.csv")

all_measurements_df = pd.read_csv(measurements_path)
all_parameters_df = pd.read_csv(parameters_path)
# convert treatment order from string to list
all_parameters_df["treatment_order"] = pd.eval(all_parameters_df["treatment_order"])

# %%

# Choosing one patient (patient 1)
measurements_df = all_measurements_df[all_measurements_df["patient_index"] == 0]
parameters_df = all_parameters_df[all_parameters_df["patient_index"] == 0]

# %%

# ANALYZING SINGLE PATIENT

# SINGLE PATIENT MODEL WITHOUT TREND


def draw_posterior_checks(model, treatment_order, observations_per_treatment):

    with model as model:
        post_pred = pm.sample_posterior_predictive(trace, samples=1000)

    treatment2_indexer = np.repeat(treatment_order, observations_per_treatment)
    treatment1_indexer = np.abs(treatment2_indexer - 1)
    # convert to boolean indexer
    treatment2_indexer = np.array(treatment2_indexer, dtype=bool)
    treatment1_indexer = np.array(treatment1_indexer, dtype=bool)

    _, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))

    # Treatment 1
    sns.distplot(
        post_pred["y"][:, treatment1_indexer].mean(axis=1),
        label="Posterior Predictive Means",
        ax=ax1,
    )
    ax1.axvline(
        measurements_df[measurements_df["treatment"] == 0]["measurement"].mean(),
        ls="--",
        color="orange",
        label="True Mean of Treatment 1 in Data",
    )
    ax1.axvline(
        parameters_df["treatment1"][0],
        ls="--",
        color="r",
        label="True Treatment 1 Value",
    )
    ax1.legend()

    # Treatment 2
    sns.distplot(
        post_pred["y"][:, treatment2_indexer].mean(axis=1),
        label="Posterior Predictive Means",
        ax=ax2,
    )
    ax2.axvline(
        measurements_df[measurements_df["treatment"] == 1]["measurement"].mean(),
        ls="--",
        color="orange",
        label="True Mean of Treatment 2 in Data",
    )
    ax2.axvline(
        parameters_df["treatment2"][0],
        ls="--",
        color="r",
        label="True Treatment 2 Value",
    )

    # Treatment difference
    sns.distplot(
        post_pred["y"][:, treatment1_indexer].mean(axis=1)
        - post_pred["y"][:, treatment2_indexer].mean(axis=1),
        label="Posterior Predictive Means",
        ax=ax3,
    )
    ax3.axvline(
        measurements_df[measurements_df["treatment"] == 0]["measurement"].mean()
        - measurements_df[measurements_df["treatment"] == 1]["measurement"].mean(),
        ls="--",
        color="orange",
        label="True Mean Difference in Treatments in Data",
    )
    ax3.axvline(
        parameters_df["treatment1"][0] - parameters_df["treatment2"][0],
        ls="--",
        color="r",
        label="True Difference in Treatments",
    )
    ax3.legend()

    plt.show()


def create_simulations(treatment1, treatment2, sigma, trend, size):

    # We get these straight from the environment. No possibility to
    # pass them trough as parameters all the way from DensityDist
    #
    # treatment_order
    # observations_per_treatment

    treatment1_array = np.array(
        list(
            pd.core.common.flatten(
                [
                    [treatment1 * abs(indicator - 1)] * observations_per_treatment
                    for indicator in treatment_order
                ]
            )
        )
    )

    treatment2_array = np.array(
        list(
            pd.core.common.flatten(
                [
                    [treatment2 * indicator] * observations_per_treatment
                    for indicator in treatment_order
                ]
            )
        )
    )

    trend_array = np.array(
        [
            trend * measurement_index
            for measurement_index in range(
                observations_per_treatment * len(treatment_order)
            )
        ]
    )

    measurement_error_array = np.random.normal(
        loc=0, scale=sigma, size=observations_per_treatment * len(treatment_order)
    )

    measurements = (
        treatment1_array + treatment2_array + trend_array + measurement_error_array
    )

    return measurements


# defining a sampling function to be able to create posterior samples
def random(point=None, size=None):

    try:
        treatment1_, treatment2_, sigma_, trend_ = pm.distributions.draw_values(
            [model.treatment1, model.treatment2, model.sigma, model.trend], point=point
        )

    except AttributeError:
        treatment1_, treatment2_, sigma_ = pm.distributions.draw_values(
            [model.treatment1, model.treatment2, model.sigma], point=point
        )

        trend_ = 0

    size = 1 if size is None else size

    return pm.distributions.generate_samples(
        create_simulations,
        treatment1=treatment1_,
        treatment2=treatment2_,
        sigma=sigma_,
        trend=trend_,
        size=size,
    )


# %%


with pm.Model() as model:

    treatment1_prior = pm.Normal("treatment1", mu=10, sigma=10)
    treatment2_prior = pm.Normal("treatment2", mu=10, sigma=10)
    sigma_prior = pm.HalfCauchy("sigma", beta=10)

    # likelihood is not a well defined distribution
    # so using prebuilt parts does not work. Writing
    # custom function calculating the log likelihood

    def likelihood(treatment1, treatment2, sigma):
        def logp_(value):

            return (-1 / 2.0) * (
                value[0].shape[0] * theano.tensor.log(2 * np.pi)
                + value[0].shape[0] * theano.tensor.log(sigma ** 2)
                + (1 / sigma ** 2)
                * (
                    (
                        value[0]
                        - (treatment1 * abs(value[1] - 1) + treatment2 * value[1])
                    )
                    ** 2
                ).sum(axis=0)
            )

        return logp_

    like = pm.DensityDist(
        "y",
        likelihood(
            treatment1=treatment1_prior, treatment2=treatment2_prior, sigma=sigma_prior,
        ),
        observed=[measurements_df["measurement"], measurements_df["treatment"]],
        random=random,
    )

    trace = pm.sample(600, tune=400, cores=3)

    pm.traceplot(trace, ["treatment1", "treatment2", "sigma"])

    # posteriors should look reasonable
    # pm.plot_posterior(trace)
    # plt.show()

    # check if your variables have reasonable credible intervals,
    # and Gelman–Rubin scores close to 1
    # pm.forestplot(trace)
    # plt.show()

    # check if your chains are impaired by high autocorrelation.
    # Also remember that thinning your chains is a waste of time
    # at best, and deluding yourself at worst
    # pm.autocorrplot(trace)
    # plt.show()

    # ideally the energy and marginal energy distributions should
    # look very similar. Long tails in the distribution of energy levels
    # indicates deteriorated sampler efficiency.
    # pm.energyplot(trace)
    plt.show()

    # %%

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
# Checking model accuracy

# posterior sampling


treatment_order = parameters_df["treatment_order"][0]
observations_per_treatment = 4

draw_posterior_checks(model, treatment_order, observations_per_treatment)


# %%

# SINGLE PATIENT MODEL WITH TREND

with pm.Model() as model:

    treatment1_prior = pm.Normal("treatment1", mu=10, sigma=10)
    treatment2_prior = pm.Normal("treatment2", mu=10, sigma=10)
    trend_prior = pm.Normal("trend", mu=0, sigma=1)
    sigma_prior = pm.HalfCauchy("sigma", beta=10)

    # likelihood is not a well defined distribution
    # so using prebuilt parts does not work. Writing
    # custom function calculating the log likelihood

    def likelihood(treatment1, treatment2, trend, sigma):
        def logp_(value):

            return (-1 / 2.0) * (
                value[0].shape[0] * theano.tensor.log(2 * np.pi)
                + value[0].shape[0] * theano.tensor.log(sigma ** 2)
                + (1 / sigma ** 2)
                * (
                    (
                        value[0]
                        - (
                            treatment1 * abs(value[1] - 1)
                            + treatment2 * value[1]
                            + trend * value[2]
                        )
                    )
                    ** 2
                ).sum(axis=0)
            )

        return logp_

    like = pm.DensityDist(
        "y",
        likelihood(
            treatment1=treatment1_prior,
            treatment2=treatment2_prior,
            trend=trend_prior,
            sigma=sigma_prior,
        ),
        observed=[
            measurements_df["measurement"],
            measurements_df["treatment"],
            measurements_df["measurement_index"],
        ],
        random=random,
    )

    trace = pm.sample(800, tune=600, cores=3)

    pm.traceplot(trace, ["treatment1", "treatment2", "trend", "sigma"])
    plt.show()

    # posteriors should look reasonable
    # pm.plot_posterior(trace)
    # plt.show()

    # check if your variables have reasonable credible intervals,
    # and Gelman–Rubin scores close to 1
    # pm.forestplot(trace)
    # plt.show()

    # check if your chains are impaired by high autocorrelation.
    # Also remember that thinning your chains is a waste of time
    # at best, and deluding yourself at worst
    # pm.autocorrplot(trace)
    # plt.show()

    # ideally the energy and marginal energy distributions should
    # look very similar. Long tails in the distribution of energy levels
    # indicates deteriorated sampler efficiency.
    # pm.energyplot(trace)
    plt.show()

    # %%

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

treatment_order = parameters_df["treatment_order"][0]
observations_per_treatment = 4

draw_posterior_checks(model, treatment_order, observations_per_treatment)


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
    treatment1_prior = pm.Normal("treatment1", mu=10, sigma=10, shape=patients_n)
    treatment2_prior = pm.Normal("treatment2", mu=10, sigma=10, shape=patients_n)
    sigma_prior = pm.HalfCauchy("sigma", beta=10, shape=patients_n)

    # likelihood is not a well defined distribution
    # so using prebuilt parts does not work. Writing
    # custom function calculating the log likelihood

    def likelihood(treatment1, treatment2, sigma):
        def logp_(value):

            return (-1 / 2.0) * (
                value[0].shape[0] * theano.tensor.log(2 * np.pi)
                + value[0].shape[0] * theano.tensor.log(sigma[patient_index] ** 2)
                + (1 / sigma[patient_index] ** 2)
                * (
                    (
                        value[0]
                        - (
                            treatment1[patient_index] * abs(value[1] - 1)
                            + treatment2[patient_index] * value[1]
                        )
                    )
                    ** 2
                ).sum(axis=0)
            ).sum(axis=0)

        return logp_

    like = pm.DensityDist(
        "y",
        likelihood(
            treatment1=treatment1_prior, treatment2=treatment2_prior, sigma=sigma_prior,
        ),
        observed=[all_measurements_df["measurement"], all_measurements_df["treatment"]]
        # random=random,
    )

    trace = pm.sample(800, tune=600, cores=3)

    pm.traceplot(trace, ["treatment1", "treatment2", "sigma"])
    plt.show()

# %%

# UNPOOLED MODEL WITH TREND

with pm.Model() as model:

    # separate parameter for each patient
    treatment1_prior = pm.Normal("treatment1", mu=10, sigma=10, shape=patients_n)
    treatment2_prior = pm.Normal("treatment2", mu=10, sigma=10, shape=patients_n)
    # trend_prior = pm.Normal("trend", mu=0, sigma=1, shape=patients_n)
    sigma_prior = pm.HalfCauchy("sigma", beta=10, shape=patients_n)

    # likelihood is not a well defined distribution
    # so using prebuilt parts does not work. Writing
    # custom function calculating the log likelihood

    def likelihood(treatment1, treatment2, sigma):
        def logp_(value):

            return (-1 / 2.0) * (
                value[0].shape[0] * theano.tensor.log(2 * np.pi)
                + value[0].shape[0] * theano.tensor.log(sigma[patient_index] ** 2)
                + (1 / sigma[patient_index] ** 2)
                * (
                    (
                        value[0]
                        - (
                            treatment1[patient_index] * abs(value[1] - 1)
                            + treatment2[patient_index] * value[1]
                            # + trend[patient_index] * value[2]
                        )
                    )
                    ** 2
                ).sum(axis=0)
            ).sum(axis=0)

        return logp_

    like = pm.DensityDist(
        "y",
        likelihood(
            treatment1=treatment1_prior,
            treatment2=treatment2_prior,
            # trend=trend_prior,
            sigma=sigma_prior,
        ),
        observed=[
            all_measurements_df["measurement"],
            all_measurements_df["treatment"],
            # all_measurements_df["measurement_index"],
        ],
        # random=random,
    )

    trace = pm.sample(1000, tune=500, cores=3)

    pm.traceplot(trace, ["treatment1", "treatment2", "sigma"])
    plt.show()

    # posteriors should look reasonable
    # pm.plot_posterior(trace)
    # plt.show()

    # check if your variables have reasonable credible intervals,
    # and Gelman–Rubin scores close to 1
    # pm.forestplot(trace)
    # plt.show()

    # check if your chains are impaired by high autocorrelation.
    # Also remember that thinning your chains is a waste of time
    # at best, and deluding yourself at worst
    # pm.autocorrplot(trace)
    # plt.show()

    # ideally the energy and marginal energy distributions should
    # look very similar. Long tails in the distribution of energy levels
    # indicates deteriorated sampler efficiency.
    # pm.energyplot(trace)
    # plt.show()

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
    population_treatment1_mean_prior = pm.Normal(
        "population_treatment1_mean", mu=10, sigma=10
    )
    population_treatment1_sd_prior = pm.HalfCauchy("population_treatment1_sd", beta=10)

    population_treatment2_mean_prior = pm.Normal(
        "population_treatment2_mean", mu=10, sigma=10
    )
    population_treatment2_sd_prior = pm.HalfCauchy("population_treatment2_sd", beta=10)

    population_measurement_error_beta_prior = pm.HalfCauchy(
        "population_measurement_error_beta", beta=10
    )

    # separate parameter for each patient
    treatment1_prior = pm.Normal(
        "treatment1",
        mu=population_treatment1_mean_prior,
        sigma=population_treatment1_sd_prior,
        shape=patients_n,
    )
    treatment2_prior = pm.Normal(
        "treatment2",
        mu=population_treatment2_mean_prior,
        sigma=population_treatment2_sd_prior,
        shape=patients_n,
    )
    sigma_prior = pm.HalfCauchy(
        "sigma", beta=population_measurement_error_beta_prior, shape=patients_n,
    )

    # likelihood is not a well defined distribution
    # so using prebuilt parts does not work. Writing
    # custom function calculating the log likelihood

    def likelihood(treatment1, treatment2, sigma):
        def logp_(value):

            return (-1 / 2.0) * (
                value[0].shape[0] * theano.tensor.log(2 * np.pi)
                + value[0].shape[0] * theano.tensor.log(sigma[patient_index] ** 2)
                + (1 / sigma[patient_index] ** 2)
                * (
                    (
                        value[0]
                        - (
                            treatment1[patient_index] * abs(value[1] - 1)
                            + treatment2[patient_index] * value[1]
                        )
                    )
                    ** 2
                ).sum(axis=0)
            ).sum(axis=0)

        return logp_

    like = pm.DensityDist(
        "y",
        likelihood(
            treatment1=treatment1_prior, treatment2=treatment2_prior, sigma=sigma_prior,
        ),
        observed=[all_measurements_df["measurement"], all_measurements_df["treatment"]],
    )

    trace = pm.sample(1000, tune=500, cores=3)

    pm.traceplot(
        trace,
        [
            "treatment1",
            "treatment2",
            "sigma",
            "population_treatment1_mean",
            "population_treatment1_sd",
            "population_treatment2_mean",
            "population_treatment2_sd",
            "population_measurement_error_beta",
        ],
    )
    plt.show()

    # posteriors should look reasonable
    pm.plot_posterior(trace)
    plt.show()

    # check if your variables have reasonable credible intervals,
    # and Gelman–Rubin scores close to 1
    # pm.forestplot(trace)
    # plt.show()

    # check if your chains are impaired by high autocorrelation.
    # Also remember that thinning your chains is a waste of time
    # at best, and deluding yourself at worst
    # pm.autocorrplot(trace)
    # plt.show()

    # ideally the energy and marginal energy distributions should
    # look very similar. Long tails in the distribution of energy levels
    # indicates deteriorated sampler efficiency.
    # pm.energyplot(trace)
    # plt.show()

    # %%

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
    population_treatment1_mean_prior = pm.Normal(
        "population_treatment1_mean", mu=10, sigma=3
    )
    population_treatment1_sd_prior = pm.HalfCauchy("population_treatment1_sd", beta=10)

    population_treatment2_mean_prior = pm.Normal(
        "population_treatment2_mean", mu=0, sigma=1
    )
    population_treatment2_sd_prior = pm.HalfCauchy("population_treatment2_sd", beta=10)

    population_measurement_error_beta_prior = pm.HalfCauchy(
        "population_measurement_error_beta", beta=10
    )

    population_trend_mean_prior = pm.Normal("population_trend_mean", mu=0, sigma=0.3)
    population_trend_sd_prior = pm.HalfCauchy("population_trend_sd", beta=2)

    # separate parameter for each patient
    treatment1_prior = pm.Normal(
        "treatment1",
        mu=population_treatment1_mean_prior,
        sigma=population_treatment1_sd_prior,
        shape=patients_n,
    )
    treatment2_prior = pm.Normal(
        "treatment2",
        mu=population_treatment2_mean_prior,
        sigma=population_treatment2_sd_prior,
        shape=patients_n,
    )
    sigma_prior = pm.HalfCauchy(
        "sigma", beta=population_measurement_error_beta_prior, shape=patients_n,
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

    def likelihood(treatment1, treatment2, trend, sigma):
        def logp_(value):

            return (-1 / 2.0) * (
                value[0].shape[0] * theano.tensor.log(2 * np.pi)
                + value[0].shape[0] * theano.tensor.log(sigma[patient_index] ** 2)
                + (1 / sigma[patient_index] ** 2)
                * (
                    (
                        value[0]
                        - (
                            treatment1[patient_index] * abs(value[1] - 1)
                            + treatment2[patient_index] * value[1]
                            + trend[patient_index] * value[2]
                        )
                    )
                    ** 2
                ).sum(axis=0)
            ).sum(axis=0)

        return logp_

    like = pm.DensityDist(
        "y",
        likelihood(
            treatment1=treatment1_prior,
            treatment2=treatment2_prior,
            trend=trend_prior,
            sigma=sigma_prior,
        ),
        observed=[
            all_measurements_df["measurement"],
            all_measurements_df["treatment"],
            all_measurements_df["measurement_index"],
        ],
    )

    trace = pm.sample(1000, tune=800, cores=3)

    pm.traceplot(
        trace,
        [
            "treatment1",
            "treatment2",
            "sigma",
            "population_treatment1_mean",
            "population_treatment1_sd",
            "population_treatment2_mean",
            "population_treatment2_sd",
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
    # pm.forestplot(trace)
    # plt.show()

    # check if your chains are impaired by high autocorrelation.
    # Also remember that thinning your chains is a waste of time
    # at best, and deluding yourself at worst
    # pm.autocorrplot(trace)
    # plt.show()

    # ideally the energy and marginal energy distributions should
    # look very similar. Long tails in the distribution of energy levels
    # indicates deteriorated sampler efficiency.
    # pm.energyplot(trace)
    # plt.show()

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
    # pm.summary(trace)


# %%
