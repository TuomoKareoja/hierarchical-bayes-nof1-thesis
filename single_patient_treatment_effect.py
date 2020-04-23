# %%

import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import os
import theano

# %%

measurements_path = os.path.join("data", "patient_measurements.csv")
parameters_path = os.path.join("data", "patient_parameters.csv")

all_measurements_df = pd.read_csv(measurements_path)
all_parameters_df = pd.read_csv(parameters_path)

# %%

# Choosing one patient (patient 1)
measurements_df = all_measurements_df[all_measurements_df["patient_number"] == 1]
parameters_df = all_parameters_df[all_parameters_df["patient_number"] == 1]

# %%

with pm.Model() as baseline_model:

    baseline = pm.Normal("baseline", mu=10, sigma=5)
    sigma = pm.HalfCauchy("sigma", beta=10, testval=1.0)

    expected_value = pm.Normal(
        "y",
        mu=baseline,
        sigma=sigma,
        observed=measurements_df[measurements_df["treatment"] == 0]["measurement"],
    )

    trace = pm.sample(1000, cores=3, tune=400)

    pm.traceplot(trace, ["baseline", "sigma"])
    plt.show()

# %%

with pm.Model() as treatment_model:

    with_treatment = pm.Normal("with_treatment", mu=11, sigma=5)
    sigma = pm.HalfCauchy("sigma", beta=10, testval=1.0)

    expected_value = pm.Normal(
        "y",
        mu=with_treatment,
        sigma=sigma,
        observed=measurements_df[measurements_df["treatment"] == 1]["measurement"],
    )

    trace = pm.sample(1000, cores=3, tune=400)

    pm.traceplot(trace, ["with_treatment", "sigma"])
    plt.show()

# %%

# combination

with pm.Model() as full_model:

    baseline_prior = pm.Normal("baseline", mu=10, sigma=10)
    treatment_effect_prior = pm.Normal("treatment_effect", mu=2, sigma=3)
    sigma_prior = pm.HalfCauchy("sigma", beta=10, testval=1.0)

    # likelihood is not a well defined distribution
    # so using prebuilt parts does not work. Writing
    # custom function calculating the log likelihood

    def likelihood(baseline, treatment, sigma):
        def logp_(value):

            return (-1 / 2.0) * (
                value[0].shape[0] * theano.tensor.log(2 * np.pi)
                + value[0].shape[0] * theano.tensor.log(sigma ** 2)
                + (1 / sigma ** 2)
                * ((value[0] - (baseline + treatment * value[1])) ** 2).sum(axis=0)
            )

        return logp_

    like = pm.DensityDist(
        "like",
        likelihood(
            baseline=baseline_prior,
            treatment=treatment_effect_prior,
            sigma=sigma_prior,
        ),
        observed=[measurements_df["measurement"], measurements_df["treatment"]],
    )

    trace = pm.sample(1000, tune=500, cores=3)

    pm.traceplot(trace, ["baseline", "treatment_effect", "sigma"])
    plt.show()

# %%
