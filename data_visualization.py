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

patient_colors = ["red", "green", "cyan", "orange", "brown", "black"]
population_color = "grey"
population_alpha = 0.6

# %%

# POPULATION PARAMETER DISTRIBUTIONS

fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

# treatment1
x = np.linspace(
    scipy.stats.norm.ppf(
        0.00000000000000001, loc=population_treatment1_mean, scale=population_treatment1_sd
    ),
    scipy.stats.norm.ppf(
        0.999, loc=population_treatment1_mean, scale=population_treatment1_sd
    ),
)
y = scipy.stats.norm.pdf(
    x, loc=population_treatment1_mean, scale=population_treatment1_sd
)

ax1.plot(x, y, color=population_color, alpha=population_alpha, lw=2)
ax1.set_xlim(9.2, 10.3)
ax1.set_ybound(lower=0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.set_yticklabels([])
ax1.set_xlabel("Treatment A Effect")

# add patient parameter values
for value, color, patient_idx in zip(
    parameters_df["treatment1"], patient_colors, parameters_df["patient_index"]
):
    ax1.axvline(x=value, color=color, label="Patient {}".format(patient_idx + 1))

ax1.legend(
    loc="upper left",
    ncol=1,
    handletextpad=0.5,
    borderpad=1,
    columnspacing=1,
    labelspacing=1,
)

# TREATMENT EFFECT
x = np.linspace(
    scipy.stats.norm.ppf(
        0.00000001, loc=population_treatment2_mean, scale=population_treatment2_sd
    ),
    scipy.stats.norm.ppf(
        0.999999999999999, loc=population_treatment2_mean, scale=population_treatment2_sd,
    ),
)
y = scipy.stats.norm.pdf(
    x, loc=population_treatment2_mean, scale=population_treatment2_sd,
)

ax3.plot(x, y, color=population_color, alpha=population_alpha, lw=2)
ax3.set_xlim(9.2, 10.3)
ax3.set_ybound(lower=0)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax3.set_yticklabels([])
ax3.set_xlabel("Treatment B Effect")

# add patient parameter values
for value, color in zip(parameters_df["treatment2"], patient_colors):
    ax3.axvline(x=value, color=color)

# MEASUREMENT ERROR
x = np.linspace(
    scipy.stats.halfcauchy.ppf(0.00, loc=0, scale=population_measurement_error_scale,),
    scipy.stats.halfcauchy.ppf(0.9, loc=0, scale=population_measurement_error_scale,),
)
y = scipy.stats.halfcauchy.pdf(x, loc=0, scale=population_measurement_error_scale,)

ax2.plot(x, y, color=population_color, alpha=population_alpha, lw=2)
ax2.set_xlim(0, 0.58)
ax2.set_ybound(lower=0)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.set_yticklabels([])
ax2.set_xlabel("Measurement Error")

# add patient parameter values
for value, color in zip(parameters_df["measurement_error_sd"], patient_colors):
    ax2.axvline(x=value, color=color)

# AUTOCORRELATION
x = np.linspace(
    scipy.stats.beta.ppf(
        0.001, a=population_autocorrelation_alpha, b=population_autocorrelation_beta
    ),
    scipy.stats.beta.ppf(
        0.999, a=population_autocorrelation_alpha, b=population_autocorrelation_beta,
    ),
)
y = scipy.stats.beta.pdf(
    x, a=population_autocorrelation_alpha, b=population_autocorrelation_beta
)

ax4.plot(x, y, color=population_color, alpha=population_alpha, lw=2)
# ax4.set_xlim(0.25, 0.42)
ax4.set_ybound(lower=0)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.spines["left"].set_visible(False)
ax4.set_yticklabels([])
ax4.set_xlabel("Autocorrelation")

# add patient parameter values
for value, color in zip(parameters_df["autocorrelation"], patient_colors):
    ax4.axvline(x=value, color=color)

# TREND
x = np.linspace(
    scipy.stats.norm.ppf(
        0.0001, loc=population_trend_mean, scale=population_trend_sd
    ),
    scipy.stats.norm.ppf(
        0.9999, loc=population_trend_mean, scale=population_trend_sd
    ),
)
y = scipy.stats.norm.pdf(
    x, loc=population_trend_mean, scale=population_trend_sd
)

ax5.plot(x, y, color=population_color, alpha=population_alpha, lw=2)
ax5.set_xlim(-0.005, 0.045)
ax5.set_ybound(lower=0)

ax4.spines["top"].set_visible(False)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)
ax5.spines["left"].set_visible(False)
ax5.set_yticklabels([])
ax5.set_xlabel("Trend")

# add patient parameter values
for value, color in zip(parameters_df["trend"], patient_colors):
    ax5.axvline(x=value, color=color)
    
# make the unused plot invisible
ax6.axis('off')

plt.savefig(
    os.path.join(visualization_folder, "population_parameter_distributions.pdf"),
    bbox_inches="tight",
)

# %%

# TIMELINE
fig, ax = plt.subplots(figsize=(8, 4))

for patient, color in zip(measurements_df["patient_index"].unique(), patient_colors):

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
    os.path.join(visualization_folder, "measurements_timeline.pdf"),
    bbox_inches="tight",
)

# %%

# PATIENT 1 TIMELINE
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    measurements_df[measurements_df["patient_index"] == 0][
        "measurement_index"
    ],
    measurements_df[measurements_df["patient_index"] == 0]["measurement"],
    color='red',
    linestyle="solid",
    linewidth=1,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.ylabel("Measurement Value")
plt.xlabel("Time Index")

# Add treatment periods
for i in range(0, total_measurements_n):
    if measurements_df['treatment1_indicator'][i] == 1:
        plt.axvspan(i-0.5, i+0.5, facecolor='grey', alpha=0.15)

plt.tight_layout()
plt.savefig(
    os.path.join(visualization_folder, "patient1_timeline.pdf"),
    bbox_inches="tight",
)

# %%
