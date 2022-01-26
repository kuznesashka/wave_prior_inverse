import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV


def define_best_solution(
        reg_score_array: np.ndarray,
        reg_coefs_array: np.ndarray,
        consider_model_simplicity: bool = False,
        lowest_n_nonzero_direction: int = 3
):
    best_shift_for_speed = reg_score_array.argmax(axis=0)
    best_score_for_speed = reg_score_array.max(axis=0)

    speed_num = reg_coefs_array.shape[1]

    if consider_model_simplicity:
        non_zero_direction_num_array = np.sum(reg_coefs_array != 0, axis=2)
        non_zero_direction_num_for_speed = []
        for speed_i in range(speed_num):
            shift = best_shift_for_speed[speed_i]
            non_zero_direction_num_for_speed.append(non_zero_direction_num_array[shift, speed_i])

        score_argsort = (-best_score_for_speed).argsort()
        direction_argsort = (
            non_zero_direction_num_for_speed[score_argsort[0:lowest_n_nonzero_direction]]
        ).argsort()
        solution_speed_ind = score_argsort[direction_argsort[0]]
    else:
        solution_speed_ind = (-best_score_for_speed).argsort()[0]

    solution_shift_ind = best_shift_for_speed[solution_speed_ind]
    solution_score = best_score_for_speed[solution_speed_ind]
    return solution_speed_ind, solution_shift_ind, solution_score


def lasso_inverse_solve(
        signal_data: np.ndarray,
        wave_data: np.ndarray,
        fit_intercept: bool = False,
        consider_model_simplicity: bool = False,
        lowest_n_nonzero_direction: int = 3,
):
    """LASSO with positive coefficients.

    Parameters
    ----------
    signal_data : np.ndarray
        Input signal data to solve the inverse problem [channel_num x samples_num].
    wave_data : np.ndarray
        Basis waves [direction_num x speed_num x channel_num x timepoint_num].
    fit_intercept : bool = False
        If True, intercept is added to solution.
    consider_model_simplicity : bool = False
        If True, the best solution is with the highest score among `lowest_n_nonzero_direction`
        solutions with lowest number of nonzero directions.
    lowest_n_nonzero_direction : int = 3
        If `consider_model_simplicity`, the parameter to define the set of potential solutions.

    Returns
    -------
    solution_score
    solution_speed_ind
    solution_shift_ind
    solution_coefs
    solution_intercept
    """

    channel_num = signal_data.shape[0]
    data_samples_num = signal_data.shape[1]
    wave_samples_num = wave_data.shape[3]  # wave length

    direction_num = wave_data.shape[0]  # number of propagation directions
    speed_num = wave_data.shape[1]  # number of propagation speeds
    window_num = data_samples_num - wave_samples_num + 1  # number of sliding window shifts

    regression = ElasticNetCV(
        l1_ratio=1, positive=True, cv=5, max_iter=100000, fit_intercept=fit_intercept
    )

    reg_coefs_array = np.zeros([window_num, speed_num, direction_num])  # regression coefficients
    reg_intercept_array = np.zeros([window_num, speed_num])  # regression intercept
    reg_score_array = np.zeros([window_num, speed_num])  # R-squared scores
    y_pred = np.zeros([window_num, speed_num, channel_num * data_samples_num])  # predicted spikes

    for w in range(window_num):
        signal_vec = signal_data[:, w: (wave_samples_num + w)].flatten()
        for s in range(speed_num):
            wave_speed = wave_data[:, s, :, :]
            wave_speed_vec = np.zeros([direction_num, signal_vec.shape[0]])
            for d in range(direction_num):
                wave_speed_vec[d] = wave_speed[d, :, :].flatten()
            regression.fit(wave_speed_vec.T, signal_vec)
            reg_coefs_array[w, s, :] = regression.coef_
            reg_intercept_array[w, s] = regression.intercept_
            y_pred[w, s, :] = regression.predict(wave_speed_vec.T)
            if fit_intercept:
                reg_score_array[w, s] = regression.score(wave_speed_vec.T, signal_vec)
            else:
                tss = sum(signal_vec ** 2)
                rss = sum((signal_vec - y_pred[w, s, :]) ** 2)
                reg_score_array[w, s] = 1 - rss / tss

    solution_speed_ind, solution_shift_ind, solution_score = define_best_solution(
        reg_score_array=reg_score_array,
        reg_coefs_array=reg_coefs_array,
        consider_model_simplicity=consider_model_simplicity,
        lowest_n_nonzero_direction=lowest_n_nonzero_direction
    )
    solution_coefs = reg_coefs_array[solution_shift_ind, solution_speed_ind]
    solution_intercept = reg_intercept_array[solution_shift_ind, solution_speed_ind]

    return solution_score, solution_speed_ind, solution_shift_ind, solution_coefs, solution_intercept


def plot_data_predicted(signal_vec, y_pred, solution_shift_ind, solution_speed_ind, solution_score):
    plt.figure()
    plt.plot(signal_vec)
    plt.plot(y_pred[solution_shift_ind, solution_speed_ind, :])
    plt.title(['R-squared = ', str(solution_score)])
