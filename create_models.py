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
    trend_prior = pm.Normal("trend", mu=0.1, sigma=0.3)
    # common variance parameter defining the error
    gamma_prior = pm.HalfCauchy("gamma", beta=10)

    # measurements are created from both priors, with a indicator setting the
    # values to 0 if the treatment is not applied at the particular observation
    measurement_est = (
        treatment1_prior * measurements_df[patient_index == 0]["treatment1_indicator"]
        + treatment2_prior * measurements_df[patient_index == 0]["treatment2_indicator"]
        + trend_prior * measurements_df[patient_index == 0]["measurement_index"]
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

    difference = pm.Deterministic('difference', treatment1_prior - treatment2_prior)

    # running the model
    trace = pm.sample(draws=800, tune=700, cores=3, random_seed=[seed, seed + 1, seed + 2])

    pm.traceplot(trace, ["treatment1", "treatment2", "trend", "gamma"])
    plt.savefig(
        os.path.join(visualization_path, "single_patient_traceplot.pdf"),
        bbox_inches="tight",
    )
    plt.show()
    summary_metrics_df = pd.DataFrame(pm.summary(trace))
    print(summary_metrics_df)
    # TODO only keep the most important metrics to have the table with the page
    with open(
        os.path.join(visualization_path, "single_patient_diag_metrics.tex"),
        "w",
    ) as file:
        file.write(
            summary_metrics_df.drop(
                ["mean", "sd", "hpd_3%", "hpd_97%"], axis=1
            ).to_latex()
        )

    # posteriors should look reasonable
    pm.plot_posterior(trace)
    plt.savefig(
        os.path.join(visualization_path, "single_patient_posteriors.pdf"),
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
    plot_name="single_patient_posterior_sampling",
)

# %%

# HIERARCHICAL MODEL

with pm.Model() as hierarchical_with_trend_model:

    # population priors
    pop_treatment1_mean_prior = pm.Normal("pop_treatment1_mean", mu=10, sigma=10)
    pop_treatment1_sd_prior = pm.HalfCauchy("pop_treatment1_sd", beta=10)

    pop_treatment2_mean_prior = pm.Normal("pop_treatment2_mean", mu=10, sigma=10)
    pop_treatment2_sd_prior = pm.HalfCauchy("pop_treatment2_sd", beta=10)

    # TODO should the trend be capped so that nobody so get better?
    pop_trend_mean_prior = pm.Normal("pop_trend_mean", mu=0.1, sigma=0.3)
    pop_trend_sd_prior = pm.HalfCauchy("pop_trend_sd", beta=2)

    pop_measurement_error_beta_prior = pm.HalfCauchy(
        "pop_measurement_error_beta", beta=10
    )

    # separate parameter for each patient
    pat_treatment1 = pm.Normal(
        "treatment1",
        mu=pop_treatment1_mean_prior,
        sigma=pop_treatment1_sd_prior,
        shape=patients_n,
    )
    pat_treatment2 = pm.Normal(
        "treatment2",
        mu=pop_treatment2_mean_prior,
        sigma=pop_treatment2_sd_prior,
        shape=patients_n,
    )
    pat_sigma = pm.HalfCauchy(
        "sigma", beta=pop_measurement_error_beta_prior, shape=patients_n,
    )
    pat_trend = pm.Normal(
        "trend", mu=pop_trend_mean_prior, sigma=pop_trend_sd_prior, shape=patients_n,
    )

    measurement_means = (
        pat_treatment1[patient_index] * measurements_df["treatment1_indicator"]
        + pat_treatment2[patient_index] * measurements_df["treatment2_indicator"]
        + pat_trend[patient_index] * measurements_df["measurement_index"]
    )

    likelihood = pm.Normal(
        "y",
        measurement_means,
        sigma=pat_sigma[patient_index],
        observed=measurements_df["measurement"],
    )

    trace = pm.sample(800, tune=500, cores=3)

    pm.traceplot(
        trace, ["treatment1", "treatment2", "sigma",],
    )
    plt.savefig(
        os.path.join(
            visualization_path, "hierarchical_model_patient_level_traceplot.pdf"
        ),
        bbox_inches="tight",
    )
    plt.show()

    pm.traceplot(
        trace,
        [
            "pop_treatment1_mean",
            "pop_treatment1_sd",
            "pop_treatment2_mean",
            "pop_treatment2_sd",
            "pop_measurement_error_beta",
            "pop_trend_mean",
            "pop_trend_sd",
        ],
    )
    plt.savefig(
        os.path.join(
            visualization_path, "hierarchical_model_population_level_traceplot.pdf"
        ),
        bbox_inches="tight",
    )
    plt.show()

    summary_metrics_df = pd.DataFrame(pm.summary(trace))
    print(summary_metrics_df)
    # TODO only keep the most important metrics to have the table with the page
    with open(
        os.path.join(visualization_path, "hierarchical_model_diag_metrics.tex"),
        "w",
    ) as file:
        file.write(
            summary_metrics_df.drop(
                ["mean", "sd", "hpd_3%", "hpd_97%"], axis=1
            ).to_latex()
        )

    # TODO We need to separate this into multiple plots
    # pm.plot_posterior(trace)
    # plt.savefig(
    #     os.path.join(visualization_path, "hierarchical_with_trend_posteriors.pdf"),
    #     bbox_inches="tight",
    # )
    # plt.show()

    pm.summary(trace)

# %%

# posterior sampling
with hierarchical_with_trend_model as model:
    post_pred = pm.sample_posterior_predictive(trace, samples=500)
    predictions = post_pred["y"]

# TODO the picture has to be smaller to fit in one page
draw_posterior_checks(
    predictions=predictions,
    measurements_df=measurements_df,
    parameters_df=parameters_df,
    plot_name="hierarchical_model_posterior_sampling",
)

# %%
