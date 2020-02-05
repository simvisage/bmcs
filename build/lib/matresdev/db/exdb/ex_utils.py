'''
Created on Apr 11, 2016

@author: rch
'''
import numpy as np


def iron_measured_curve(y_asc, x_asc, t_asc, jump_rtol=0.0001):
    ''''Force curve with removes jumps in the stress strain curve
    due to sliding in the buttstrap clamping. The small unloading/loading
    branches are removed from the data and the smoothed curve is
    determined based on the remaining data.
    In order to smoothen out the effect of the jump the pieces of the
    F-w-curve that contain jumps in the force (=unloading/reloading) path
    are removed from the data in a range up to the double range of the jump,
    so that the smoothing does not change the shape of the F-w- curve.
    '''
    F_asc = y_asc
    eps_asc = x_asc
    time_asc = t_asc

    # get the differences of the force values between two adjacent
    # time steps
    #
    jump_arr = F_asc[1:] - F_asc[0:-1]

    # determine the criteria for a jump
    # based on the data range and the specified tolerances:
    #
    jump_crit = jump_rtol * F_asc[-1]

    # get the indices of the measurement data at which a
    # force jump exceeds (last step before the jump) the defined tolerance
    # criteria
    # i.e. negative jump that exceeds the defined tolerance magnitude
    #
    jump_idx_arr = np.where(jump_arr < -jump_crit)[0] - 1

    # index of the measurement data where the force reaches
    # the same magnitude before the sudden value drop due to the jump
    #
    jump_idx2_arr = np.zeros_like(jump_idx_arr)

    # amount of indices between the sudden value drop of the force and
    # the reloading to the same load level; delta value indicate
    # the strain range that will be removed in order to smoothen out
    # the influence of the jump in the force curve
    #
    delta_jump_idx_arr = np.zeros_like(jump_idx_arr)

    # search at which index the force reaches its old value before
    # the jump again check that this value is index wise and time
    # wise reached after the jump occurred
    #
    for n_idx, jump_idx in enumerate(jump_idx_arr):
        delta_F = F_asc - F_asc[jump_idx]
        delta_eps = eps_asc - eps_asc[jump_idx]
        delta_t = time_asc - time_asc[jump_idx]
        bool_arr_F = delta_F > 0.
        bool_arr_eps = delta_eps > 0.
        bool_arr_t = delta_t > 0.
        bool_arr = bool_arr_F * bool_arr_eps * bool_arr_t
        try:
            jump_idx2 = np.where(bool_arr)[0][1]
        except:
            break
        delta_jump_idx = jump_idx2 - jump_idx
        jump_idx2_arr[n_idx] = jump_idx2
        delta_jump_idx_arr[n_idx] = delta_jump_idx

    # remove jumps from the jump index when a succeeding jump still
    # lays within the influence range of an earlier jump
    # this can happen when jumps occur within the re-mounting
    # branch of the force
    #
    remove_idx = []
    for i in range(jump_idx2_arr.shape[0] - 1):
        if np.any(jump_idx2_arr[:i + 1] > jump_idx2_arr[i + 1]):
            remove_idx += [i + 1]

    jump_idx_arr = np.delete(jump_idx_arr, remove_idx)
    jump_idx2_arr = np.delete(jump_idx2_arr, remove_idx)
    delta_jump_idx_arr = np.delete(delta_jump_idx_arr, remove_idx)

    # specify the factor with whom the index delta range of a jump
    # (i.e. displacement range of the jump)
    # is multiplied, i.e. up to which index the values of the
    # F-w- curve are removed
    #
    jump_smooth_fact = 2

    # remove the values of the curve within the jump and the neighboring
    # region
    #
    F_asc_ironed_list = []
    eps_asc_ironed_list = []

    max_stress_idx = len(y_asc)
    jump_idx_arr_ = np.hstack(
        [np.array([0.]), jump_idx_arr, np.array([max_stress_idx])])
    delta_jump_idx_arr_ = np.hstack(
        [np.array([0]), delta_jump_idx_arr, np.array([0])])

    for i in range(jump_idx_arr_.shape[0] - 1):
        F_asc_ironed_list += \
            [F_asc[jump_idx_arr_[i] +
                   jump_smooth_fact *
                   delta_jump_idx_arr_[i]: jump_idx_arr_[i + 1]]]
        eps_asc_ironed_list += \
            [eps_asc[jump_idx_arr_[i] +
                     jump_smooth_fact *
                     delta_jump_idx_arr_[i]: jump_idx_arr_[i + 1]]]

    # remove the values of the curve within the jump
    # and the neighboring region
    #
    F_asc_ironed = np.hstack(F_asc_ironed_list)
    eps_asc_ironed = np.hstack(eps_asc_ironed_list)
    return F_asc_ironed, eps_asc_ironed
