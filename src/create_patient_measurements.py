import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv

load_dotenv()
seed = int(os.getenv("SEED"))

np.random.seed(seed)


def create_patient_measurements(
    treatment_measurements_n,
    treatment_order,
    treatment1,
    treatment2,
    measurement_error_sd,
    autocorrelation=0,
    trend=0,
):

    # TREATMENT1
    treatment1_array = np.array(
        list(
            pd.core.common.flatten(
                [
                    [treatment1 * abs(indicator - 2)] * treatment_measurements_n
                    for indicator in treatment_order
                ]
            )
        )
    )

    # TREATMENT2
    treatment2_array = np.array(
        list(
            pd.core.common.flatten(
                [
                    [treatment2 * (indicator - 1)] * treatment_measurements_n
                    for indicator in treatment_order
                ]
            )
        )
    )

    # ERROR TERM
    # no autoregressive component
    ar = np.array([1, 0])
    # lag 1 moving average component
    ma = np.array([1, autocorrelation])
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    # if autocorrelation is zero this just normally distributed error
    error_array = arma_process.generate_sample(
        nsample=len(treatment1_array), scale=measurement_error_sd,
    )

    # TREND
    trend_array = np.array(
        [
            trend * measurement_index
            for measurement_index in range(len(treatment1_array))
        ]
    )

    measurements = treatment1_array + treatment2_array + trend_array + error_array

    return measurements
