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


def draw_posterior_checks(predictions, treatment_order, observations_per_treatment):

    for patient, patient_treatment_order in zip(
        range(predictions.shape[1]), treatment_order
    ):

        treatment2_indexer = np.repeat(
            patient_treatment_order, observations_per_treatment
        )
        treatment1_indexer = np.abs(treatment2_indexer - 1)
        # convert to boolean indexer
        treatment2_indexer = np.array(treatment2_indexer, dtype=bool)
        treatment1_indexer = np.array(treatment1_indexer, dtype=bool)

        _, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))

        # Treatment 1
        sns.distplot(
            predictions[:, patient][:, treatment1_indexer].mean(axis=1),
            label="Posterior Predictive Means",
            ax=ax1,
        )
        ax1.axvline(
            all_measurements_df[
                (all_measurements_df["treatment"] == 0)
                & (all_measurements_df["patient_index"] == patient)
            ]["measurement"].mean(),
            ls="--",
            color="orange",
            label="Mean in Data",
        )
        ax1.axvline(
            all_parameters_df["treatment1"][patient],
            ls="--",
            color="r",
            label="Real Parameter Value",
        )
        ax1.set_title("Treatment 1")
        ax1.legend()

        # Treatment 2
        sns.distplot(
            predictions[:, patient][:, treatment2_indexer].mean(axis=1),
            label="Posterior Predictive Means",
            ax=ax2,
        )
        ax2.axvline(
            all_measurements_df[
                (all_measurements_df["treatment"] == 1)
                & (all_measurements_df["patient_index"] == patient)
            ]["measurement"].mean(),
            ls="--",
            color="orange",
            label="Mean in Data",
        )
        ax2.axvline(
            all_parameters_df["treatment2"][patient],
            ls="--",
            color="r",
            label="Real Parameter Value",
        )
        ax2.set_title("Treatment 2")

        # Treatment difference
        sns.distplot(
            predictions[:, patient][:, treatment1_indexer].mean(axis=1)
            - predictions[:, patient][:, treatment2_indexer].mean(axis=1),
            label="Posterior Predictive Means",
            ax=ax3,
        )
        ax3.axvline(
            all_measurements_df[
                (all_measurements_df["treatment"] == 0)
                & (all_measurements_df["patient_index"] == patient)
            ]["measurement"].mean()
            - all_measurements_df[
                (all_measurements_df["treatment"] == 1)
                & (all_measurements_df["patient_index"] == patient)
            ]["measurement"].mean(),
            ls="--",
            color="orange",
            label="Mean in Data",
        )
        ax3.axvline(
            all_parameters_df["treatment1"][patient]
            - all_parameters_df["treatment2"][patient],
            ls="--",
            color="r",
            label="Real Parameter Value",
        )
        ax3.set_title("Treatment 1 - Treatment 2")
        plt.show()


def create_simulations(treatment1, treatment2, sigma, trend, size):

    # We get these straight from the environment. No possibility to
    # pass them trough as parameters all the way from DensityDist
    #
    # treatment_order
    # observations_per_treatment

    def create_simulated_experiment(
        treatment1_, treatment2_, sigma_, trend_, treatment_order_
    ):

        treatment1_array = np.array(
            list(
                pd.core.common.flatten(
                    [
                        [treatment1_ * abs(indicator - 1)] * observations_per_treatment
                        for indicator in treatment_order_
                    ]
                )
            )
        )

        treatment2_array = np.array(
            list(
                pd.core.common.flatten(
                    [
                        [treatment2_ * indicator] * observations_per_treatment
                        for indicator in treatment_order_
                    ]
                )
            )
        )

        trend_array = np.array(
            [
                trend_ * measurement_index
                for measurement_index in range(
                    observations_per_treatment * len(treatment_order_)
                )
            ]
        )

        measurement_error_array = np.random.normal(
            loc=0, scale=sigma_, size=observations_per_treatment * len(treatment_order_)
        )

        measurements = (
            treatment1_array + treatment2_array + trend_array + measurement_error_array
        )

        return measurements

    if type(treatment1) is np.ndarray:

        results_list = []

        for patient, (treatment1_, treatment2_, sigma_, trend_) in enumerate(
            zip(treatment1, treatment2, sigma, trend)
        ):

            patient_treatment_order = treatment_order[patient]

            results_list.append(
                create_simulated_experiment(
                    treatment1_, treatment2_, sigma_, trend_, patient_treatment_order
                )
            )

        # returning each simulations for each patient in separate "rows"
        return np.vstack(results_list)

    else:

        patient_treatment_order = treatment_order[0]
        # we want to keep the same output format regardless
        # if there is one or multiple patients
        return np.vstack(
            [
                create_simulated_experiment(
                    treatment1, treatment2, sigma, trend, patient_treatment_order
                )
            ]
        )


# defining a sampling function to be able to create posterior samples
def random(point=None, size=None):

    # check if there is a trend defined in the model
    try:
        treatment1_, treatment2_, sigma_, trend_ = pm.distributions.draw_values(
            [model.treatment1, model.treatment2, model.sigma, model.trend], point=point
        )

    except AttributeError:
        treatment1_, treatment2_, sigma_ = pm.distributions.draw_values(
            [model.treatment1, model.treatment2, model.sigma], point=point
        )

        # if more than one patient (more than one treatment effect)
        # create a list of 0 trend values for each patient
        if type(treatment1_) is np.ndarray:
            trend_ = np.repeat(0, len(treatment1_))

        else:
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


with pm.Model() as single_patient_no_trend_model:

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

treatment_order = parameters_df["treatment_order"]
observations_per_treatment = 4

with single_patient_no_trend_model as model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
    predictions = post_pred["y"]

draw_posterior_checks(predictions, treatment_order, observations_per_treatment)

# %%

# SINGLE PATIENT MODEL WITH TREND

with pm.Model() as single_patient_with_trend_model:

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

    trace = pm.sample(700, tune=600, cores=3)

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

treatment_order = parameters_df["treatment_order"]
observations_per_treatment = 4

with single_patient_with_trend_model as model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
    predictions = post_pred["y"]

draw_posterior_checks(predictions, treatment_order, observations_per_treatment)

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
        observed=[all_measurements_df["measurement"], all_measurements_df["treatment"]],
        random=random,
    )

    trace = pm.sample(800, tune=600, cores=3)

    pm.traceplot(trace, ["treatment1", "treatment2", "sigma"])
    plt.show()

# %%

treatment_order = all_parameters_df["treatment_order"]
observations_per_treatment = 4

with all_patients_no_trend_model as model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
    predictions = post_pred["y"]

draw_posterior_checks(predictions, treatment_order, observations_per_treatment)

# %%

# UNPOOLED MODEL WITH TREND

with pm.Model() as all_patients_with_trend_model:

    # separate parameter for each patient
    treatment1_prior = pm.Normal("treatment1", mu=10, sigma=10, shape=patients_n)
    treatment2_prior = pm.Normal("treatment2", mu=10, sigma=10, shape=patients_n)
    trend_prior = pm.Normal("trend", mu=0, sigma=1, shape=patients_n)
    sigma_prior = pm.HalfCauchy("sigma", beta=10, shape=patients_n)

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
        random=random,
    )

    trace = pm.sample(800, tune=600, cores=3)

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

treatment_order = all_parameters_df["treatment_order"]
observations_per_treatment = 4

with all_patients_with_trend_model as model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
    predictions = post_pred["y"]

draw_posterior_checks(predictions, treatment_order, observations_per_treatment)

# %%

# HIERARCHICAL MODELS


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
        random=random,
    )

    trace = pm.sample(800, tune=600, cores=3)

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

# posterior sampling

treatment_order = all_parameters_df["treatment_order"]
observations_per_treatment = 4

with hierarchical_no_trend_model as model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
    predictions = post_pred["y"]

draw_posterior_checks(predictions, treatment_order, observations_per_treatment)

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
        random=random,
    )

    trace = pm.sample(800, tune=800, cores=3)

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

# posterior sampling

treatment_order = all_parameters_df["treatment_order"]
observations_per_treatment = 4

with hierarchical_with_trend_model as model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
    predictions = post_pred["y"]

draw_posterior_checks(predictions, treatment_order, observations_per_treatment)


# %%
