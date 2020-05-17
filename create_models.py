# %%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from dotenv import load_dotenv

from src.draw_posterior_checks import draw_posterior_checks
from src.likelihood import (
    single_patient_no_trend_likelihood,
    single_patient_with_trend_likelihood,
    all_patients_no_trend_likelihood,
    all_patients_with_trend_likelihood,
)
from src.create_patient_measurements import create_patient_measurements

load_dotenv()

plt.style.use("seaborn-white")

# %%

seed = int(os.getenv("SEED"))

np.random.seed(seed)

patients_n = int(os.getenv("PATIENTS_N"))
blocks_n = int(os.getenv("BLOCKS_N"))
treatment_measurements_n = int(os.getenv("TREATMENT_MEASUREMENTS_N"))

# treatment and no treatment only. No multiple treatments
total_measurements_n = blocks_n * treatment_measurements_n * 2

population_treatment1_mean = float(os.getenv("POPULATION_TREATMENT1_MEAN"))
population_treatment1_sd = float(os.getenv("POPULATION_TREATMENT1_SD"))
population_treatment2_mean = float(os.getenv("POPULATION_TREATMENT2_MEAN"))
population_treatment2_sd = float(os.getenv("POPULATION_TREATMENT2_SD"))
population_trend_mean = float(os.getenv("POPULATION_TREND_MEAN"))
population_trend_sd = float(os.getenv("POPULATION_TREND_SD"))

population_measurement_error_shape = float(
    os.getenv("POPULATION_MEASUREMENT_ERROR_SHAPE")
)
population_measurement_error_scale = float(
    os.getenv("POPULATION_MEASUREMENT_ERROR_SCALE")
)

population_autocorrelation_alpha = float(os.getenv("POPULATION_AUTOCORRELATION_ALPHA"))
population_autocorrelation_beta = float(os.getenv("POPULATION_AUTOCORRELATION_BETA"))

# %%

measurements_path = os.path.join("data", "patient_measurements.csv")
parameters_path = os.path.join("data", "patient_parameters.csv")

all_measurements_df = pd.read_csv(measurements_path)
all_parameters_df = pd.read_csv(parameters_path)
# convert treatment order from string to list
all_parameters_df["treatment_order"] = pd.eval(all_parameters_df["treatment_order"])
# used for subsetting parameters
patient_index = all_measurements_df["patient_index"]

# %%

# Choosing one patient (patient 1)
measurements_df = all_measurements_df[all_measurements_df["patient_index"] == 0]
parameters_df = all_parameters_df[all_parameters_df["patient_index"] == 0]

# %%


def create_simulations(treatment1, treatment2, sigma, trend, size):

    # We get these straight from the environment. No possibility to
    # pass them trough as parameters all the way from DensityDist
    #
    # treatment_order
    # observations_per_treatment

    if type(treatment1) is np.ndarray:

        results_list = []

        for patient, (treatment1_, treatment2_, sigma_, trend_) in enumerate(
            zip(treatment1, treatment2, sigma, trend)
        ):

            patient_treatment_order = treatment_order[patient]

            results_list.append(
                create_patient_measurements(
                    treatment_measurements_n=treatment_measurements_n,
                    treatment_indicator_list=patient_treatment_order,
                    treatment1=treatment1_,
                    treatment2=treatment2_,
                    measurement_error_sd=sigma_,
                    trend=trend_,
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
                create_patient_measurements(
                    treatment_measurements_n=treatment_measurements_n,
                    treatment_indicator_list=patient_treatment_order,
                    treatment1=treatment1,
                    treatment2=treatment2,
                    measurement_error_sd=sigma,
                    trend=trend,
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

# ANALYZING SINGLE PATIENT

# SINGLE PATIENT MODEL WITHOUT TREND

with pm.Model() as single_patient_no_trend_model:

    treatment1_prior = pm.Normal("treatment1", mu=10, sigma=10)
    treatment2_prior = pm.Normal("treatment2", mu=10, sigma=10)
    sigma_prior = pm.HalfCauchy("sigma", beta=10)

    like = pm.DensityDist(
        "y",
        single_patient_no_trend_likelihood(
            treatment1=treatment1_prior, treatment2=treatment2_prior, sigma=sigma_prior,
        ),
        observed=[measurements_df["measurement"], measurements_df["treatment"]],
        random=random,
    )

    trace = pm.sample(600, tune=400, cores=3)

    pm.traceplot(trace, ["treatment1", "treatment2", "sigma"])

    # posteriors should look reasonable
    pm.plot_posterior(trace)
    plt.show()

    pm.summary(trace)

# %%
# Checking model accuracy

# posterior sampling

treatment_order = parameters_df["treatment_order"]

with single_patient_no_trend_model as model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
    predictions = post_pred["y"]

draw_posterior_checks(
    predictions=predictions,
    measurements_df=all_measurements_df,
    parameters_df=all_parameters_df,
    treatment_order=treatment_order,
    treatment_measurements_n=treatment_measurements_n,
)

# %%

# SINGLE PATIENT MODEL WITH TREND

with pm.Model() as single_patient_with_trend_model:

    treatment1_prior = pm.Normal("treatment1", mu=10, sigma=10)
    treatment2_prior = pm.Normal("treatment2", mu=10, sigma=10)
    trend_prior = pm.Normal("trend", mu=0, sigma=1)
    sigma_prior = pm.HalfCauchy("sigma", beta=10)

    like = pm.DensityDist(
        "y",
        single_patient_with_trend_likelihood(
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

    pm.plot_posterior(trace)
    plt.show()

    pm.summary(trace)

# %%

treatment_order = parameters_df["treatment_order"]

with single_patient_with_trend_model as model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
    predictions = post_pred["y"]

draw_posterior_checks(
    predictions=predictions,
    measurements_df=all_measurements_df,
    parameters_df=all_parameters_df,
    treatment_order=treatment_order,
    treatment_measurements_n=treatment_measurements_n,
)

# %%

# ANALYZING ALL PATIENTS AT ONCE

# UNPOOLED MODEL WITHOUT TREND

with pm.Model() as all_patients_no_trend_model:

    # separate parameter for each patient
    treatment1_prior = pm.Normal("treatment1", mu=10, sigma=10, shape=patients_n)
    treatment2_prior = pm.Normal("treatment2", mu=10, sigma=10, shape=patients_n)
    sigma_prior = pm.HalfCauchy("sigma", beta=10, shape=patients_n)

    like = pm.DensityDist(
        "y",
        all_patients_no_trend_likelihood(
            treatment1=treatment1_prior,
            treatment2=treatment2_prior,
            sigma=sigma_prior,
            patient_index=patient_index,
        ),
        observed=[all_measurements_df["measurement"], all_measurements_df["treatment"]],
        random=random,
    )

    trace = pm.sample(800, tune=600, cores=3)

    pm.traceplot(trace, ["treatment1", "treatment2", "sigma"])
    plt.show()

    pm.plot_posterior(trace)
    plt.show()

    pm.summary(trace)

# %%

treatment_order = all_parameters_df["treatment_order"]

with all_patients_no_trend_model as model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
    predictions = post_pred["y"]

draw_posterior_checks(
    predictions=predictions,
    measurements_df=all_measurements_df,
    parameters_df=all_parameters_df,
    treatment_order=treatment_order,
    treatment_measurements_n=treatment_measurements_n,
)


# %%

# UNPOOLED MODEL WITH TREND

with pm.Model() as all_patients_with_trend_model:

    # separate parameter for each patient
    treatment1_prior = pm.Normal("treatment1", mu=10, sigma=10, shape=patients_n)
    treatment2_prior = pm.Normal("treatment2", mu=10, sigma=10, shape=patients_n)
    trend_prior = pm.Normal("trend", mu=0, sigma=1, shape=patients_n)
    sigma_prior = pm.HalfCauchy("sigma", beta=10, shape=patients_n)

    like = pm.DensityDist(
        "y",
        all_patients_with_trend_likelihood(
            treatment1=treatment1_prior,
            treatment2=treatment2_prior,
            trend=trend_prior,
            sigma=sigma_prior,
            patient_index=patient_index,
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

    pm.plot_posterior(trace)
    plt.show()

    pm.summary(trace)

# %%

treatment_order = all_parameters_df["treatment_order"]

with all_patients_with_trend_model as model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
    predictions = post_pred["y"]

draw_posterior_checks(
    predictions=predictions,
    measurements_df=all_measurements_df,
    parameters_df=all_parameters_df,
    treatment_order=treatment_order,
    treatment_measurements_n=treatment_measurements_n,
)

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

    like = pm.DensityDist(
        "y",
        all_patients_no_trend_likelihood(
            treatment1=treatment1_prior,
            treatment2=treatment2_prior,
            sigma=sigma_prior,
            patient_index=patient_index,
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

    pm.plot_posterior(trace)
    plt.show()

    pm.summary(trace)

# %%

# posterior sampling

treatment_order = all_parameters_df["treatment_order"]

with hierarchical_no_trend_model as model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
    predictions = post_pred["y"]

draw_posterior_checks(
    predictions=predictions,
    measurements_df=all_measurements_df,
    parameters_df=all_parameters_df,
    treatment_order=treatment_order,
    treatment_measurements_n=treatment_measurements_n,
)

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

    like = pm.DensityDist(
        "y",
        all_patients_with_trend_likelihood(
            treatment1=treatment1_prior,
            treatment2=treatment2_prior,
            trend=trend_prior,
            sigma=sigma_prior,
            patient_index=patient_index,
        ),
        observed=[
            all_measurements_df["measurement"],
            all_measurements_df["treatment"],
            all_measurements_df["measurement_index"],
        ],
        random=random,
    )

    trace = pm.sample(800, tune=1000, cores=3)

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

    pm.plot_posterior(trace)
    plt.show()

    pm.summary(trace)

# %%

# posterior sampling

treatment_order = all_parameters_df["treatment_order"]

with hierarchical_with_trend_model as model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
    predictions = post_pred["y"]

draw_posterior_checks(
    predictions=predictions,
    measurements_df=all_measurements_df,
    parameters_df=all_parameters_df,
    treatment_order=treatment_order,
    treatment_measurements_n=treatment_measurements_n,
)

# %%
