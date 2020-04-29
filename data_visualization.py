# %%

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-white")

# %%

measurements_path = os.path.join("data", "patient_measurements.csv")
parameters_path = os.path.join("data", "patient_parameters.csv")
measurements_df = pd.read_csv(measurements_path)
parameters_df = pd.read_csv(parameters_path)

visualization_folder = os.path.join("figures")

# %%

# PARAMETER DISTRIBUTIONS

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
    nrows=2, ncols=3, figsize=(8, 5.5)
)

ax1.hist(parameters_df["baselevel"])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_xlabel("Baselevel")
ax1.set_ylabel("Number of Patients")

ax2.hist(parameters_df["treatment_effect"])
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xlabel("Treatment Effect")

ax3.hist(parameters_df["trend"])
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.set_xlabel("Trend")

ax4.hist(parameters_df["measurement_error_sd"])
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.set_xlabel("Measurement Error\nStandard Deviation")

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
    [
        "baselevel",
        "treatment_effect",
        "trend",
        "measurement_error_sd",
        "autocorrelation",
    ]
]
parameters_nice_names_df.columns = [
    "Baselevel",
    "Treatment Effect",
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
