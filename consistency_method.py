import numpy as np


def norm_sub_frequency(est_value_list: list, tolerance=1e-3, flag_p3=False):
    np_est_value_list = np.array(est_value_list)
    estimates = np.copy(np_est_value_list)
    i = 0
    while (np.fabs(sum(estimates) - 1) > tolerance) or (estimates < 0).any():
        i += 1
        if (estimates <= 0).all():
            estimates[:] = 1 / estimates.size
            break
        estimates[estimates < 0] = 0
        total = sum(estimates)
        mask = estimates > 0
        if (flag_p3 and i >= 2) or (i >= 3):
            break
        diff = (1.0 - total) / sum(mask)
        estimates[mask] += diff
    return estimates
