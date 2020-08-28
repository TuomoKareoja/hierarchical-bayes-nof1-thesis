# %%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from dotenv import load_dotenv

from src.draw_posterior_checks import draw_posterior_checks

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

population_measurement_error_scale = float(
    os.getenv("POPULATION_MEASUREMENT_ERROR_SCALE")
)

population_autocorrelation_alpha = float(os.getenv("POPULATION_AUTOCORRELATION_ALPHA"))
population_autocorrelation_beta = float(os.getenv("POPULATION_AUTOCORRELATION_BETA"))

visualization_path = os.path.join("figures")

measurements_path = os.path.join("data", "patient_measurements.csv")
parameters_path = os.path.join("data", "patient_parameters.csv")

measurements_df = pd.read_csv(measurements_path)
parameters_df = pd.read_csv(parameters_path)
# used for subsetting measurements
patient_index = measurements_df["patient_index"]

# %%

# SINGLE PATIENT MODEL

with pm.Model() as single_patient_no_trend_model:

    # separate priors for each treatment
    treatment1_prior = pm.Normal("treatment1", mu=10, sigma=10)
    treatment2_prior = pm.Normal("treatment2", mu=10, sigma=10)
    # common variance parameter defining the error
    gamma_prior = pm.HalfCauchy("gamma", beta=10)

    # measurements are created from both priors, with a indicator setting the
    # values to 0 if the treatment is not applied at the particular observation
    measurement_est = (
        treatment1_prior * measurements_df[patient_index == 0]["treatment1_indicator"]
        + treatment2_prior * measurements_df[patient_index == 0]["treatment2_indicator"]
    )

    # likelihood is normal distribution with the same amount of dimensions
    # as the patient has measurements and and the mean is defined by either
    # the treatment 1 prior or treatment 2 prior with the same sigma
    likelihood = pm.Normal(
        "y",
        measurement_est,
        sigma=gamma_prior,
        observed=measurements_df[patient_index == 0]["measurement"],
    )

    # running the model
    trace = pm.sample(800, tune=500, cores=3)

    pm.traceplot(trace, ["treatment1", "treatment2", "gamma"])
    plt.savefig(
        os.path.join(visualization_path, "single_patient_no_trend_traceplot.pdf"),
        bbox_inches="tight",
    )
    plt.show()
    summary_metrics_df = pd.DataFrame(pm.summary(trace))
    print(summary_metrics_df)
    # TODO only keep the most important metrics to have the table with the page
    with open(
        os.path.join(visualization_path, "single_patient_no_trend_diag_metrics.tex"),
        "w",
    ) as file:
        file.write(summary_metrics_df.drop(["mean", "sd", 'hpd_3%', 'hpd_97%'], axis=1).to_latex())

    # posteriors should look reasonable
    pm.plot_posterior(trace)
    plt.savefig(
        os.path.join(visualization_path, "single_patient_no_trend_posteriors.pdf"),
        bbox_inches="tight",
    )
    plt.show()

# %%

# posterior sampling
with single_patient_no_trend_model as model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
    predictions = post_pred["y"]

draw_posterior_checks(
    predictions=predictions,
    measurements_df=measurements_df[patient_index == 0],
    parameters_df=parameters_df[parameters_df["patient_index"] == 0],
    plot_name="single_patient_no_trend_posterior_sampling",
)

# %%

# HIERARCHICAL MODEL

with pm.Model() as hierarchical_with_trend_model:

    # population priors
    population_treatment1_mean_prior = pm.Normal(
        "population_treatment1_mean", mu=10, sigma=10
    )
    population_treatment1_sd_prior = pm.HalfCauchy("population_treatment1_sd", beta=10)

    population_treatment2_mean_prior = pm.Normal(
        "population_treatment2_mean", mu=10, sigma=10
    )
    population_treatment2_sd_prior = pm.HalfCauchy("population_treatment2_sd", beta=10)

    # TODO should the trend be capped so that nobody so get better?
    population_trend_mean_prior = pm.Normal("population_trend_mean", mu=0.1, sigma=0.3)
    population_trend_sd_prior = pm.HalfCauchy("population_trend_sd", beta=2)

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
    trend_prior = pm.Normal(
        "trend",
        mu=population_trend_mean_prior,
        sigma=population_trend_sd_prior,
        shape=patients_n,
    )

    measurement_est = (
        treatment1_prior[patient_index] * measurements_df["treatment1_indicator"]
        + treatment2_prior[patient_index] * measurements_df["treatment2_indicator"]
        + trend_prior[patient_index] * measurements_df["measurement_index"]
    )

    likelihood = pm.Normal(
        "y",
        measurement_est,
        sigma=sigma_prior[patient_index],
        observed=measurements_df["measurement"],
    )

    trace = pm.sample(800, tune=300, cores=3)

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
    plt.savefig(
        os.path.join(visualization_path, "hierarchical_with_trend_traceplot.pdf"),
        bbox_inches="tight",
    )
    plt.show()

    pm.plot_posterior(trace)
    plt.savefig(
        os.path.join(visualization_path, "hierarchical_with_trend_posteriors.pdf"),
        bbox_inches="tight",
    )
    plt.show()

    pm.summary(trace)

# %%

# posterior sampling
with hierarchical_with_trend_model as model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
    predictions = post_pred["y"]

draw_posterior_checks(
    predictions=predictions,
    measurements_df=measurements_df,
    parameters_df=parameters_df,
    plot_name="hierarchical_with_trend_posterior_sampling",
)

# %%
