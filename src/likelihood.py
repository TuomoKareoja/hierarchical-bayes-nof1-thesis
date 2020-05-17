import theano
import numpy as np


def single_patient_no_trend_likelihood(treatment1, treatment2, sigma):
    def logp_(value):

        return (-1 / 2.0) * (
            value[0].shape[0] * theano.tensor.log(2 * np.pi)
            + value[0].shape[0] * theano.tensor.log(sigma ** 2)
            + (1 / sigma ** 2)
            * (
                (value[0] - (treatment1 * abs(value[1] - 1) + treatment2 * value[1]))
                ** 2
            ).sum(axis=0)
        )

    return logp_


def single_patient_with_trend_likelihood(treatment1, treatment2, trend, sigma):
    def logp_(value):

        return (-1 / 2.0) * (
            value[0].shape[0] * theano.tensor.log(2 * np.pi)
            + value[0].shape[0] * theano.tensor.log(sigma ** 2)
            + (1 / sigma ** 2)
            * (
                (
                    value[0]
                    - (
                        treatment1 * abs(value[1] - 1)
                        + treatment2 * value[1]
                        + trend * value[2]
                    )
                )
                ** 2
            ).sum(axis=0)
        )

    return logp_


def all_patients_no_trend_likelihood(treatment1, treatment2, sigma, patient_index):
    def logp_(value):

        return (-1 / 2.0) * (
            value[0].shape[0] * theano.tensor.log(2 * np.pi)
            + value[0].shape[0] * theano.tensor.log(sigma[patient_index] ** 2)
            + (1 / sigma[patient_index] ** 2)
            * (
                (
                    value[0]
                    - (
                        treatment1[patient_index] * abs(value[1] - 1)
                        + treatment2[patient_index] * value[1]
                    )
                )
                ** 2
            ).sum(axis=0)
        ).sum(axis=0)

    return logp_


def all_patients_with_trend_likelihood(
    treatment1, treatment2, trend, sigma, patient_index
):
    def logp_(value):

        return (-1 / 2.0) * (
            value[0].shape[0] * theano.tensor.log(2 * np.pi)
            + value[0].shape[0] * theano.tensor.log(sigma[patient_index] ** 2)
            + (1 / sigma[patient_index] ** 2)
            * (
                (
                    value[0]
                    - (
                        treatment1[patient_index] * abs(value[1] - 1)
                        + treatment2[patient_index] * value[1]
                        + trend[patient_index] * value[2]
                    )
                )
                ** 2
            ).sum(axis=0)
        ).sum(axis=0)

    return logp_
