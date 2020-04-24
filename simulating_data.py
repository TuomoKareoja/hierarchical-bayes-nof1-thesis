# %%

import pandas as pd
import numpy as np
import random
import os
import statsmodels.api as sm

# %%

seed = 123

np.random.seed(123)
random.seed(123)

# %%

data_folder = os.path.join("data")
params_file_name = "patient_parameters.csv"
measurements_file_name = "patient_measurements.csv"
params_path = os.path.join(data_folder, params_file_name)
measurements_path = os.path.join(data_folder, measurements_file_name)

# %%

# Study design
patients_n = 40
# must be an even number, because of balanced design
blocks_n = 4
treatment_measurements_n = 7
# treatment and no treatment only. No multiple treatments
total_measurements_n = blocks_n * treatment_measurements_n * 2

# %%

# Population Level Parameters

# normal distribution
population_base_level_mean = 10
population_base_level_sd = 1
population_treatment_effect_mean = 1
population_treatment_effect_sd = 0.5
population_trend_mean = 0.02
population_trend_sd = 0.01
# gamma distribution
population_measurement_error_shape = 1.5
population_measurement_error_scale = 1.5
# beta distribution
population_autocorrelation_alpha = 80
population_autocorrelation_beta = 200


# Patient Level Parameters
patient_base_level_array = np.random.normal(
    population_base_level_mean, population_base_level_sd, patients_n
)
patient_treatment_effect_array = np.random.normal(
    population_treatment_effect_mean, population_treatment_effect_sd, patients_n
)
patient_trend_array = np.random.normal(
    population_trend_mean, population_trend_sd, patients_n
)
patient_measurement_error_sd_array = np.random.gamma(
    population_measurement_error_shape, population_measurement_error_scale, patients_n
)
patient_autocorrelation_array = np.random.beta(
    population_autocorrelation_alpha, population_autocorrelation_beta, patients_n
)

# %%

# creating patient dataframe
patient_params_df = pd.DataFrame(
    {
        "patient_index": [i for i in range(patients_n)],
        "base_level": patient_base_level_array,
        "treatment_effect": patient_treatment_effect_array,
        "trend": patient_trend_array,
        "measurement_error_sd": patient_measurement_error_sd_array,
        "autocorrelation": patient_autocorrelation_array,
    }
)

# %%

# Randomizing patient treatment order


patient_treatment_orders_list = []

for patient in range(patients_n):

    treatment_order_list = []

    for i in range(blocks_n // 2):
        block = [0, 0]
        # randomizing the treatment to first or second slot in the block
        # 1 marks the effective treatment and 0 is the default
        effective_treatment_index = np.random.binomial(1, 0.5)
        block[effective_treatment_index] = 1

        treatment_order_list.extend(block)
        # add the balancing block
        treatment_order_list.extend(block[::-1])

    patient_treatment_orders_list.append(treatment_order_list)


patient_params_df["treatment_order"] = patient_treatment_orders_list

# %%

patient_measurements_df_list = []

for index, patient in patient_params_df.iterrows():

    # AUTOCORRELATED ERRORS
    # no autoregressive component
    ar = np.array([1, 0])
    # lag 1 moving average component
    ma = np.array([1, patient["autocorrelation"]])
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    ma_process_array = arma_process.generate_sample(
        nsample=total_measurements_n, scale=patient["measurement_error_sd"],
    )

    # BASE LEVEL AND TREND
    base_and_trend_array = np.array(
        [
            patient["base_level"] + patient["trend"] * measurement_index
            for measurement_index in range(total_measurements_n)
        ]
    )

    # TREATMENT EFFECT
    treatment_effects_array = np.array(
        list(
            pd.core.common.flatten(
                [
                    [patient["treatment_effect"] * indicator] * treatment_measurements_n
                    for indicator in patient["treatment_order"]
                ]
            )
        )
    )

    measurements = ma_process_array + base_and_trend_array + treatment_effects_array

    patient_df = pd.DataFrame(
        {
            "patient_index": [patient["patient_index"]] * total_measurements_n,
            "measurement_index": [
                measurement_index for measurement_index in range(total_measurements_n)
            ],
            "treatment_period_index": list(
                pd.core.common.flatten(
                    [
                        [treatment_period_index] * treatment_measurements_n
                        for treatment_period_index in range(blocks_n * 2)
                    ]
                )
            ),
            "block_index": list(
                pd.core.common.flatten(
                    [
                        [block_index] * treatment_measurements_n * 2
                        for block_index in range(blocks_n)
                    ]
                )
            ),
            "treatment": list(
                pd.core.common.flatten(
                    [
                        [indicator] * treatment_measurements_n
                        for indicator in patient["treatment_order"]
                    ]
                )
            ),
            "measurement": measurements,
        }
    )

    patient_measurements_df_list.append(patient_df)

patient_measurements_df = pd.concat(patient_measurements_df_list, ignore_index=True)

# %%

patient_params_df.to_csv(params_path, index=False)
patient_measurements_df.to_csv(measurements_path, index=False)
