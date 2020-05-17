import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def draw_posterior_checks(
    predictions,
    measurements_df,
    parameters_df,
    treatment_order,
    treatment_measurements_n,
):

    for patient, patient_treatment_order in zip(
        range(predictions.shape[1]), treatment_order
    ):

        treatment2_indexer = np.repeat(
            patient_treatment_order, treatment_measurements_n
        )
        treatment1_indexer = np.abs(treatment2_indexer - 1)
        # convert to boolean indexer
        treatment2_indexer = np.array(treatment2_indexer, dtype=bool)
        treatment1_indexer = np.array(treatment1_indexer, dtype=bool)

        _, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))

        # Treatment 1
        sns.distplot(
            predictions[:, patient][:, treatment1_indexer].mean(axis=1),
            label="Posterior Predictive Means",
            ax=ax1,
        )
        ax1.axvline(
            measurements_df[
                (measurements_df["treatment"] == 0)
                & (measurements_df["patient_index"] == patient)
            ]["measurement"].mean(),
            ls="--",
            color="red",
            label="Mean in Data",
        )
        ax1.set_title("Treatment 1")

        # Treatment 2
        sns.distplot(
            predictions[:, patient][:, treatment2_indexer].mean(axis=1),
            label="Posterior Predictive Means",
            ax=ax2,
        )
        ax2.axvline(
            measurements_df[
                (measurements_df["treatment"] == 1)
                & (measurements_df["patient_index"] == patient)
            ]["measurement"].mean(),
            ls="--",
            color="red",
            label="Mean in Data",
        )
        ax2.set_title("Treatment 2")

        # Treatment difference
        sns.distplot(
            predictions[:, patient][:, treatment1_indexer].mean(axis=1)
            - predictions[:, patient][:, treatment2_indexer].mean(axis=1),
            label="Posterior Predictive Means",
            ax=ax3,
        )
        ax3.axvline(
            measurements_df[
                (measurements_df["treatment"] == 0)
                & (measurements_df["patient_index"] == patient)
            ]["measurement"].mean()
            - measurements_df[
                (measurements_df["treatment"] == 1)
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
        plt.show()
