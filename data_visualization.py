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

# %%

fig, ax = plt.subplots(figsize=(15, 10))

for patient in measurements_df["patient_index"].unique():

    if patient == 0:
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
        linewidth=1,
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.ylabel("Good Thing")
plt.xlabel("Measurement Index")
plt.show()
