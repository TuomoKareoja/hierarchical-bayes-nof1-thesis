# %%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

plt.style.use("seaborn-white")

# %%

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
measurements_df = pd.read_csv(measurements_path)
parameters_df = pd.read_csv(parameters_path)

visualization_folder = os.path.join("figures")

# %%

# POPULATION PARAMETER DISTRIBUTIONS

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(8, 2))

# treatment1
x = np.linspace(
    scipy.stats.norm.ppf(
        0.01, loc=population_treatment1_mean, scale=population_treatment1_sd
    ),
    scipy.stats.norm.ppf(
        0.99, loc=population_treatment1_mean, scale=population_treatment1_sd
    ),
)
y = scipy.stats.norm.pdf(
    x, loc=population_treatment1_mean, scale=population_treatment1_sd
)

ax1.plot(x, y, "r-", lw=2)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_xlabel("Treatment A Effect")
ax1.set_ylabel("Probability Density")

# TREATMENT EFFECT
x = np.linspace(
    scipy.stats.norm.ppf(
        0.01, loc=population_treatment2_mean, scale=population_treatment2_sd
    ),
    scipy.stats.norm.ppf(
        0.99, loc=population_treatment2_mean, scale=population_treatment2_sd,
    ),
)
y = scipy.stats.norm.pdf(
    x, loc=population_treatment2_mean, scale=population_treatment2_sd,
)

ax2.plot(x, y, "r-", lw=2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xlabel("Treatment B Effect")

# MEASUREMENT ERROR
x = np.linspace(
    scipy.stats.invgamma.ppf(
        0.01,
        a=population_measurement_error_shape,
        scale=population_measurement_error_scale,
    ),
    scipy.stats.invgamma.ppf(
        0.99,
        a=population_measurement_error_shape,
        scale=population_measurement_error_scale,
    ),
)
y = scipy.stats.invgamma.pdf(
    x, a=population_measurement_error_shape, scale=population_measurement_error_scale,
)

ax3.plot(
    x, y, "r-", lw=2,
)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.set_xlabel("Measurement Error")

# AUTOCORRELATION
x = np.linspace(
    scipy.stats.beta.ppf(
        0.01, a=population_autocorrelation_alpha, b=population_autocorrelation_beta
    ),
    scipy.stats.beta.ppf(
        0.99, a=population_autocorrelation_alpha, b=population_autocorrelation_beta,
    ),
)
y = scipy.stats.beta.pdf(
    x, a=population_autocorrelation_alpha, b=population_autocorrelation_beta
)

ax4.plot(x, y, "r-", lw=2)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.set_xlabel("Autocorrelation")

plt.savefig(
    os.path.join(visualization_folder, "population_parameter_distributions.pdf"),
    bbox_inches="tight",
)


# %%

# PATIENT PARAMETER DISTRIBUTIONS

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
    nrows=2, ncols=3, figsize=(8, 5.5)
)

ax1.hist(parameters_df["treatment1"])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_xlabel("Treatment A Effect")
ax1.set_ylabel("Number of Patients")

ax2.hist(parameters_df["treatment2"])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xlabel("Treatment B Effect")

ax3.hist(parameters_df["trend"])
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.set_xlabel("Trend")

ax4.hist(parameters_df["measurement_error_sd"])
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.set_xlabel("Measurement Error")

ax5.hist(parameters_df["autocorrelation"])
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)
ax5.set_xlabel("Autocorrelation")

ax6.axis("off")

plt.tight_layout()
plt.savefig(
    os.path.join(visualization_folder, "patient_parameter_distribution.pdf"),
    bbox_inches="tight",
)

# %%

# PARAMETER CORRELATIONS

parameters_nice_names_df = parameters_df[
    ["treatment1", "treatment2", "trend", "measurement_error_sd", "autocorrelation"]
]

parameters_nice_names_df.columns = [
    "Treatment 1",
    "Treatment 2",
    "Trend",
    "Measurement Error\nStandard Deviation",
    "Autocorrelation",
]

sns.set(rc={"figure.figsize": (8, 8)})
sns.set_style("ticks")
sns.pairplot(data=parameters_nice_names_df, corner=True)

plt.tight_layout()
plt.savefig(
    os.path.join(visualization_folder, "patient_parameter_relationships.pdf"),
    bbox_inches="tight",
)

# %%

# TIMELINE
fig, ax = plt.subplots(figsize=(8, 4))

for patient in measurements_df["patient_index"].unique():

    if patient == max(measurements_df["patient_index"]):
        color = "red"
        alpha = 1

    else:
        color = "grey"
        alpha = 0.3

    ax.plot(
        measurements_df[measurements_df["patient_index"] == patient][
            "measurement_index"
        ],
        measurements_df[measurements_df["patient_index"] == patient]["measurement"],
        color=color,
        linestyle="solid",
        alpha=alpha,
        linewidth=2,
    )


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.ylabel("Bad Thing")
plt.xlabel("Measurement Index")

plt.tight_layout()
plt.savefig(
    os.path.join(visualization_folder, "measurements_timeline.pdf"),
    bbox_inches="tight",
)

# %%
