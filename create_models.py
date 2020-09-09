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

population_treatment_a_mean = float(os.getenv("POPULATION_TREATMENT1_MEAN"))
population_treatment_a_sd = float(os.getenv("POPULATION_TREATMENT1_SD"))
population_treatment_b_mean = float(os.getenv("POPULATION_TREATMENT2_MEAN"))
population_treatment_b_sd = float(os.getenv("POPULATION_TREATMENT2_SD"))
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
patient_colors = ["red", "green", "blue", "orange", "brown", "black"]

# %%

# SINGLE PATIENT MODEL

with pm.Model() as single_patient_no_trend_model:

    treatment_a = pm.Normal("Treatment A", mu=10, sigma=1)
    treatment_b = pm.Normal("Treatment B", mu=10, sigma=1)
    trend = pm.Normal("Trend", mu=0.1, sigma=0.3)
    # common variance parameter defining the error
    gamma = pm.HalfCauchy("Gamma", beta=1)

    # measurements are created from both priors, with a indicator setting the
    # values to 0 if the treatment is not applied at the particular observation
    measurement_est = (
        treatment_a * measurements_df[patient_index == 0]["treatment1_indicator"]
        + treatment_b * measurements_df[patient_index == 0]["treatment2_indicator"]
        + trend * measurements_df[patient_index == 0]["measurement_index"]
    )

    # likelihood is normal distribution with the same amount of dimensions
    # as the patient has measurements and and the mean is defined by either
    # the treatment 1 prior or treatment 2 prior with the same sigma
    likelihood = pm.Normal(
        "y",
        measurement_est,
        sigma=gamma,
        observed=measurements_df[patient_index == 0]["measurement"],
    )

    difference = pm.Deterministic(
        "Treatment Difference (A-B)", treatment_a - treatment_b
    )

    # running the model
    single_patient_trace = pm.sample(
        draws=800, tune=700, cores=3, random_seed=[seed, seed + 1, seed + 2]
    )

    pm.traceplot(
        single_patient_trace,
        ["Treatment A", "Treatment B", "Trend", "Gamma"],
        divergences="top",
    )
    plt.savefig(
        os.path.join(visualization_path, "single_patient_traceplot.pdf"),
        bbox_inches="tight",
    )
    plt.show()
    summary_metrics_df = pd.DataFrame(
        pm.summary(single_patient_trace, kind="diagnostics")
    )
    print(summary_metrics_df)
    with open(
        os.path.join(visualization_path, "single_patient_diag_metrics.tex"), "w",
    ) as file:
        file.write(summary_metrics_df.to_latex())

    # posteriors should look reasonable
    pm.plot_posterior(single_patient_trace)
    plt.savefig(
        os.path.join(visualization_path, "single_patient_posteriors.pdf"),
        bbox_inches="tight",
    )
    plt.show()

# %%

# posterior sampling
with single_patient_no_trend_model as model:
    single_patient_post_pred = pm.sample_posterior_predictive(
        single_patient_trace, samples=500
    )
    single_patient_predictions = single_patient_post_pred["y"]

draw_posterior_checks(
    predictions=single_patient_predictions,
    measurements_df=measurements_df[patient_index == 0],
    parameters_df=parameters_df[parameters_df["patient_index"] == 0],
    plot_name="single_patient_posterior_sampling",
)

# %%

# TIMELINE
fig, ax = plt.subplots(figsize=(8, 4))

for sample in single_patient_predictions:

    ax.plot(
        range(len(sample)),
        sample,
        linestyle="solid",
        linewidth=1,
        color="grey",
        alpha=0.05,
    )


ax.plot(
    measurements_df[patient_index == 0]["measurement_index"],
    measurements_df[patient_index == 0]["measurement"],
    color="red",
    linestyle="solid",
    linewidth=1,
    label="Patient 1",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.ylabel("Measurement Value")
plt.xlabel("Time Index")
plt.legend(
    loc="upper left",
    ncol=2,
    handletextpad=0.5,
    borderpad=1,
    columnspacing=1,
    labelspacing=1,
)

plt.tight_layout()
plt.savefig(
    os.path.join(visualization_path, "posterior_sample_timeline_single_patient.pdf"),
    bbox_inches="tight",
)
plt.show()

# %%

# HIERARCHICAL MODEL

with pm.Model() as hierarchical_with_trend_model:

    # population priors
    pop_treatment_a_mean = pm.Normal("Population Treatment A Mean", mu=10, sigma=10)
    pop_treatment_a_sd = pm.HalfCauchy("Population Treatment A Sd", beta=10)

    pop_treatment_b_mean = pm.Normal("Population Treatment B Mean", mu=10, sigma=10)
    pop_treatment_b_sd = pm.HalfCauchy("Population Treatment B Sd", beta=10)

    pop_trend_mean = pm.Normal("Population Trend Mean", mu=0.1, sigma=0.3)
    pop_trend_sd = pm.HalfCauchy("Population Trend SD", beta=2)

    pop_gamma = pm.HalfCauchy("Population Gamma", beta=10)

    # separate parameter for each patient
    pat_treatment_a = pm.Normal(
        "Treatment A",
        mu=pop_treatment_a_mean,
        sigma=pop_treatment_a_sd,
        shape=patients_n,
    )
    pat_treatment_b = pm.Normal(
        "Treatment B",
        mu=pop_treatment_b_mean,
        sigma=pop_treatment_b_sd,
        shape=patients_n,
    )
    # TODO check what is the parameter implemented in PyMC3
    pat_gamma = pm.HalfCauchy("Gamma", beta=pop_gamma, shape=patients_n,)
    pat_trend = pm.Normal(
        "Trend", mu=pop_trend_mean, sigma=pop_trend_sd, shape=patients_n,
    )

    measurement_means = (
        pat_treatment_a[patient_index] * measurements_df["treatment1_indicator"]
        + pat_treatment_b[patient_index] * measurements_df["treatment2_indicator"]
        + pat_trend[patient_index] * measurements_df["measurement_index"]
    )

    likelihood = pm.Normal(
        "y",
        measurement_means,
        sigma=pat_gamma[patient_index],
        observed=measurements_df["measurement"],
    )

    # adding the comparison between the treatments
    pop_difference = pm.Deterministic(
        "Population Treatment Difference (A-B)",
        pop_treatment_a_mean - pop_treatment_b_mean,
    )
    pat_difference = pm.Deterministic(
        "Treatment Difference (A-B)", pat_treatment_a - pat_treatment_b
    )

    hierarchical_trace = pm.sample(
        800, tune=500, cores=3, random_seed=[seed, seed + 1, seed + 2]
    )

    pm.traceplot(
        hierarchical_trace,
        [
            "Population Treatment A Mean",
            "Population Treatment A Sd",
            "Population Treatment B Mean",
            "Population Treatment B Sd",
            "Population Trend Mean",
            "Population Trend SD",
            "Population Gamma",
        ],
    )
    plt.savefig(
        os.path.join(
            visualization_path, "hierarchical_model_population_level_traceplot.pdf"
        ),
        bbox_inches="tight",
    )
    plt.show()

    pm.traceplot(
        hierarchical_trace, ["Treatment A", "Treatment B", "Trend", "Gamma"],
    )
    plt.savefig(
        os.path.join(
            visualization_path, "hierarchical_model_patient_level_traceplot.pdf"
        ),
        bbox_inches="tight",
    )
    plt.show()

    summary_metrics_df = pd.DataFrame(
        pm.summary(hierarchical_trace, kind="diagnostics")
    )
    print(summary_metrics_df)
    # TODO only keep the most important metrics to have the table with the page
    with open(
        os.path.join(visualization_path, "hierarchical_model_diag_metrics.tex"), "w",
    ) as file:
        file.write(summary_metrics_df.to_latex())

    pm.plot_posterior(
        hierarchical_trace,
        [
            "Population Treatment A Mean",
            "Population Treatment B Mean",
            "Population Trend Mean",
            "Population Gamma",
            "Population Treatment Difference (A-B)",
        ],
    )
    plt.savefig(
        os.path.join(
            visualization_path, "hierarchical_model_population_level_posteriors.pdf"
        ),
        bbox_inches="tight",
    )
    plt.show()

    pm.plot_posterior(
        hierarchical_trace,
        ["Treatment A", "Treatment B", "Trend", "Treatment Difference (A-B)"],
    )
    plt.savefig(
        os.path.join(
            visualization_path, "hierarchical_model_patient_level_posteriors.pdf"
        ),
        bbox_inches="tight",
    )
    plt.show()


# %%

# posterior sampling
with hierarchical_with_trend_model as model:
    hierarchical_post_pred = pm.sample_posterior_predictive(
        hierarchical_trace, samples=500
    )
    hierarchical_predictions = hierarchical_post_pred["y"]

# TODO why the picture is too small?
draw_posterior_checks(
    predictions=hierarchical_predictions,
    measurements_df=measurements_df,
    parameters_df=parameters_df,
    plot_name="hierarchical_model_posterior_sampling",
)

# %%

# TIMELINE


fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(14, 8))

for patient, color, ax in zip(
    measurements_df["patient_index"].unique(), patient_colors, axs.ravel()
):

    patient_samples = hierarchical_predictions[
        :, measurements_df["patient_index"] == patient
    ]

    for sample in patient_samples:

        ax.plot(
            range(len(sample)),
            sample,
            linestyle="solid",
            linewidth=1,
            color="grey",
            alpha=0.03,
        )

    ax.plot(
        measurements_df[measurements_df["patient_index"] == patient][
            "measurement_index"
        ],
        measurements_df[measurements_df["patient_index"] == patient]["measurement"],
        color=color,
        linestyle="solid",
        linewidth=1,
        label="Patient {}".format(patient + 1),
    )
    ax.legend(
        loc="upper left",
        ncol=2,
        handletextpad=0.5,
        borderpad=1,
        columnspacing=1,
        labelspacing=1,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axs[0, 0].set_ylabel("Measurement Value")
axs[2, 0].set_xlabel("Time Index")
plt.tight_layout()
plt.savefig(
    os.path.join(
        visualization_path, "posterior_sample_timeline_hierarchical_model.pdf"
    ),
    bbox_inches="tight",
)
plt.show()


# %%

# Comparing posterior results between the models

# Unpooled model with all patients

with pm.Model() as non_hierarchical_model:

    treatment_a = pm.Normal("Treatment A", mu=10, sigma=1, shape=patients_n)
    treatment_b = pm.Normal("Treatment B", mu=10, sigma=1, shape=patients_n)
    trend = pm.Normal("Trend", mu=0.1, sigma=0.3, shape=patients_n)
    # common variance parameter defining the error
    gamma = pm.HalfCauchy("Gamma", beta=1, shape=patients_n)

    # measurements are created from both priors, with a indicator setting the
    # values to 0 if the treatment is not applied at the particular observation
    measurement_est = (
        treatment_a[patient_index] * measurements_df["treatment1_indicator"]
        + treatment_b[patient_index] * measurements_df["treatment2_indicator"]
        + trend[patient_index] * measurements_df["measurement_index"]
    )

    likelihood = pm.Normal(
        "y",
        measurement_est,
        sigma=gamma[patient_index],
        observed=measurements_df["measurement"],
    )

    difference = pm.Deterministic(
        "Treatment Difference (A-B)", treatment_a - treatment_b
    )

    # running the model
    non_hierarchical_trace = pm.sample(
        draws=800, tune=700, cores=3, random_seed=[seed, seed + 1, seed + 2]
    )

    pm.traceplot(
        non_hierarchical_trace,
        ["Treatment A", "Treatment B", "Trend", "Gamma"],
        divergences="top",
    )

    pm.plot_posterior(single_patient_trace)


# %%

# taking all values of the chains after tuning steps

# NOTE chains don't include the tuning steps
non_hierarchical_treatment_a = non_hierarchical_trace["Treatment A"].mean(axis=0)
non_hierarchical_treatment_b = non_hierarchical_trace["Treatment B"].mean(
    axis=0
)

hierarchical_treatment_a = hierarchical_trace["Treatment A"].mean(axis=0)
hierarchical_treatment_b = hierarchical_trace["Treatment B"].mean(axis=0)

# %%

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(
    111,
    xlabel="Treatment A",
    ylabel="Treatment B",
)

for patient in range(patients_n):

    ax.scatter(
        non_hierarchical_treatment_a[patient],
        non_hierarchical_treatment_b[patient],
        marker='o',
        c=patient_colors[patient],
        s=60,
        label='Patient {} non-hierarchical'.format(patient+1)
    )

    ax.scatter(
        hierarchical_treatment_a[patient],
        hierarchical_treatment_b[patient],
        marker='x',
        c=patient_colors[patient],
        s=60,
        label='Patient {} hierarchical'.format(patient+1)
    )

    ax.arrow(
        non_hierarchical_treatment_a[patient],
        non_hierarchical_treatment_b[patient],
        hierarchical_treatment_a[patient] - non_hierarchical_treatment_a[patient],
        hierarchical_treatment_b[patient] - non_hierarchical_treatment_b[patient],
        fc="k",
        ec="k",
        length_includes_head=True,
        alpha=0.2,
        head_width=0.005,
    )
ax.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(
        visualization_path, "posterior_shrinkage.pdf"
    ),
    bbox_inches="tight",
)
plt.show()


# %%
