import os

import matplotlib.pyplot as plt
import seaborn as sns

visualization_path = os.path.join("figures")


def draw_posterior_checks(
    predictions, measurements_df, parameters_df, plot_name,
):

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

        _, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))

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
        ax1.set_title("Treatment 1")

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
        ax2.set_title("Treatment 2")

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
        ax3.set_title("Treatment 1 - Treatment 2")
        ax3.legend()
        plt.savefig(
            os.path.join(visualization_path, plot_name + str(".pdf")),
            bbox_inches="tight",
        )
        plt.show()
