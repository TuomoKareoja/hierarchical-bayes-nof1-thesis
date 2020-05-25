# %%

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from dotenv import load_dotenv

from src.create_patient_measurements import create_patient_measurements

load_dotenv()

plt.style.use("seaborn-white")

# %%

seed = int(os.getenv("SEED"))

np.random.seed(seed)
random.seed(seed)

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

data_folder = os.path.join("data")
visualization_folder = os.path.join("figures")
params_file_name = "patient_parameters.csv"
measurements_file_name = "patient_measurements.csv"
params_path = os.path.join(data_folder, params_file_name)
measurements_path = os.path.join(data_folder, measurements_file_name)

# %% Creating patient level parameters

patient_treatment1_array = np.random.normal(
    population_treatment1_mean, population_treatment1_sd, patients_n
)
patient_treatment2_array = np.random.normal(
    population_treatment2_mean, population_treatment2_sd, patients_n
)
patient_trend_array = np.random.normal(
    population_trend_mean, population_trend_sd, patients_n
)
# scipy shares the numpy random seed
patient_measurement_error_sd_array = scipy.stats.invgamma.rvs(
    a=population_measurement_error_shape,
    scale=population_measurement_error_scale,
    size=patients_n,
)
patient_autocorrelation_array = np.random.beta(
    population_autocorrelation_alpha, population_autocorrelation_beta, patients_n
)

# %%

# creating patient dataframe
patient_params_df = pd.DataFrame(
    {
        "patient_index": [i for i in range(patients_n)],
        "treatment1": patient_treatment1_array,
        "treatment2": patient_treatment2_array,
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
        block = [1, 1]
        # randomizing the second treatment to first or second slot in the block
        effective_treatment_index = np.random.binomial(1, 0.5)
        block[effective_treatment_index] = 2

        treatment_order_list.extend(block)
        # add the balancing block
        treatment_order_list.extend(block[::-1])

    patient_treatment_orders_list.append(treatment_order_list)

# creating indexers that take value 1 when treatment is given and 0 otherwise
patient_params_df["treatment_order"] = patient_treatment_orders_list

# %%

patient_measurements_df_list = []

for _, patient in patient_params_df.iterrows():

    measurements = create_patient_measurements(
        treatment_measurements_n=treatment_measurements_n,
        treatment_order=patient["treatment_order"],
        treatment1=patient["treatment1"],
        treatment2=patient["treatment2"],
        trend=patient["trend"],
        autocorrelation=patient["autocorrelation"],
        measurement_error_sd=patient["measurement_error_sd"],
    )

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
            "treatment1_indicator": list(
                pd.core.common.flatten(
                    [
                        [abs(indicator - 2)] * treatment_measurements_n
                        for indicator in patient["treatment_order"]
                    ]
                )
            ),
            "treatment2_indicator": list(
                pd.core.common.flatten(
                    [
                        [indicator - 1] * treatment_measurements_n
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

# %%
