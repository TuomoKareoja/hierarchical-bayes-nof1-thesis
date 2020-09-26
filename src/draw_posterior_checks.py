# %%
import os

import matplotlib.pyplot as plt
import seaborn as sns

visualization_path = os.path.join("figures")


def draw_posterior_checks(
    predictions, measurements_df, parameters_df, plot_name,
):
    _, axes = plt.subplots(
        nrows=len(parameters_df), ncols=3, figsize=(12, 3 * len(parameters_df))
    )

    for patient in parameters_df["patient_index"]:

        patient_preds = predictions[:, measurements_df["patient_index"] == patient]
        patient_treatment1_preds = patient_preds[
            :,
            measurements_df[measurements_df["patient_index"] == patient][
                "treatment1_indicator"
            ]
            == 1,
        ]
        patient_treatment2_preds = patient_preds[
            :,
            measurements_df[measurements_df["patient_index"] == patient][
                "treatment2_indicator"
            ]
            == 1,
        ]

        if len(parameters_df) == 1:
            ax1 = axes[0]
            ax2 = axes[1]
            ax3 = axes[2]

        else:
            ax1 = axes[patient, 0]
            ax2 = axes[patient, 1]
            ax3 = axes[patient, 2]

        # Treatment 1
        sns.distplot(
            patient_treatment1_preds.mean(axis=1),
            label="Posterior Predictive Means",
            ax=ax1,
        )
        ax1.axvline(
            measurements_df[
                (measurements_df["treatment1_indicator"] == 1)
                & (measurements_df["patient_index"] == patient)
            ]["measurement"].mean(),
            ls="--",
            color="red",
            label="Mean in Data",
        )
        # Add label for the patient number
        ax1.text(
            x=0.09,
            y=0.9,
            s="Patient {}".format(patient + 1),
            transform=ax1.transAxes,
        )
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Treatment 2
        sns.distplot(
            patient_treatment2_preds.mean(axis=1),
            label="Posterior Predictive Means",
            ax=ax2,
        )
        ax2.axvline(
            measurements_df[
                (measurements_df["treatment2_indicator"] == 1)
                & (measurements_df["patient_index"] == patient)
            ]["measurement"].mean(),
            ls="--",
            color="red",
            label="Mean in Data",
        )
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # Treatment difference
        sns.distplot(
            patient_treatment1_preds.mean(axis=1)
            - patient_treatment2_preds.mean(axis=1),
            label="Posterior Predictive Means",
            ax=ax3,
        )
        ax3.axvline(
            measurements_df[
                (measurements_df["treatment1_indicator"] == 1)
                & (measurements_df["patient_index"] == patient)
            ]["measurement"].mean()
            - measurements_df[
                (measurements_df["treatment2_indicator"] == 1)
                & (measurements_df["patient_index"] == patient)
            ]["measurement"].mean(),
            ls="--",
            color="red",
            label="Mean in Data",
        )
        ax3.axvline(
            parameters_df["treatment1"][patient] - parameters_df["treatment2"][patient],
            ls="-",
            color="green",
            label="Real Parameter Value",
        )
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        
        # drawing the legends for the first patient at the top of the graph
        if patient == 0:
            ax3.legend(bbox_to_anchor=(-1.15, 1.15, 1.0, 1.15), loc=8, ncol=3)

    if len(parameters_df) == 1:
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]

    else:
        ax1 = axes[0, 0]
        ax2 = axes[0, 1]
        ax3 = axes[0, 2]

    ax1.set_title("Treatment 1")
    ax2.set_title("Treatment 2")
    ax3.set_title("Treatment 1 - Treatment 2")
    plt.savefig(
        os.path.join(visualization_path, plot_name + str(".pdf")), bbox_inches="tight",
    )
    plt.show()


# %%
